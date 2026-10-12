import socket
import sys
import getopt
import os
import time
import json
import math

PI = 3.14159265359

data_size = 2**17

ophelp=  'Options:\n'
ophelp+= ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp+= ' --port, -p <port>    TORCS port. [3001]\n'
ophelp+= ' --id, -i <id>        ID for server. [SCR]\n'
ophelp+= ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp+= ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp+= ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp+= ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp+= ' --parameters, -f <json>  JSON string with keys: target_speed, steer_gain, centering_gain,\n'
ophelp+= '                          brake_threshold, gear_thresholds (list), traction_control (bool).\n'
ophelp+= ' --debug, -d          Output full telemetry.\n'
ophelp+= ' --help, -h           Show this help.\n'
ophelp+= ' --version, -v        Show current version.'
usage= 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage= usage + ophelp
version= "20130505-2"

# =============================================================
#  MODULAR DRIVE LOGIC WITH USER PARAMETERS
#  MISSION 1 : complete a lap in under 2 minutes
#  MISSION 2 : do it without incurring car damage
# =============================================================

# -------- USER CONFIGURABLE PARAMETERS --------
######### REQUIRED FOR PRACTICE MODE - CAN CHANGE BUT DO NOT REMOVE #########

TARGET_SPEED = 360          # km/h – max ceiling; corners capped by GRIP_K
STEER_GAIN = 10             # smooth steering, avoids oscillation
CENTERING_GAIN = 0.12       # gentle center-correction
BRAKE_THRESHOLD = 0.45      # radians (~26°) — brake on mis-alignment
GEAR_SPEEDS = [0, 28, 55, 88, 130, 175]    # shift up even earlier → top gear sooner
ENABLE_TRACTION_CONTROL = True

# Grip constant: safe_speed = sqrt(ahead_sensor * GRIP_K)
# Only track[9] (0°) is reliable on 12m-wide corkscrew.
# Pulled back 320→280: final chicane was arriving too fast and crashing.
# 200m ahead → 237 km/h,  50m → 118 km/h,  20m → 75 km/h
GRIP_K = 280.0

######### END OF REQUIRED #########
# ----------------------------------------------


# ==================== HELPER FUNCTIONS ====================

def calculate_steering(S):
    """Blend angle correction with track-center correction.

    Corkscrew fix:
    - Scale centering gain with |trackPos|: gentle near center, strong at edges.
      This prevents the centering term from fighting the banking mid-section
      while still pulling the car back firmly if it drifts wide.
    - If car is fully off-track (|trackPos| > 1.0), steer hard back onto track
      regardless of angle — recovery takes priority.
    """
    track_pos = S.get('trackPos', 0)
    angle     = S.get('angle', 0)

    # Off-track: steer directly back, ignore angle correction
    if abs(track_pos) > 1.0:
        return max(-1.0, min(1.0, -track_pos * 0.8))

    # Scale centering gain with distance from center (gentle center, firm edges)
    adaptive_centering = CENTERING_GAIN * (1.0 + abs(track_pos) * 2.0)
    steer = (angle * STEER_GAIN / PI) - (track_pos * adaptive_centering)
    return max(-1.0, min(1.0, steer))


def corner_speed_factor(S):
    """Return a 0-1 multiplier used ONLY to reduce throttle when off-track
    or badly mis-aligned. NOT used to cap straight-line target speed.

    On-track and aligned → returns 1.0 (full throttle allowed).
    Drifting toward edge  → reduces toward 0.
    Off-track             → returns 0.0 (no throttle floor).
    """
    track_pos = S.get('trackPos', 0)
    angle     = abs(S.get('angle', 0))

    # Only penalise when drifting wide or mis-aligned
    # Loosened: pos edge moved 0.7→0.8, angle sensitivity halved (1.0→0.5)
    pos_penalty   = max(0.0, 1.0 - max(0.0, abs(track_pos) - 0.8) * 3.0)
    angle_penalty = max(0.0, 1.0 - angle * 0.5)

    combined = min(pos_penalty, angle_penalty)

    # Raised floor 0.3→0.5: more throttle mid-corner when on-track
    floor = 0.0 if abs(track_pos) > 1.0 else 0.5
    return max(floor, combined)


def corner_entry_speed(S):
    """Compute safe entry speed using ONLY track[9] (0° straight ahead).

    On a 12m-wide track like corkscrew:
      ±45° sensors always read ~8.5m (side barriers) — useless for corners
      ±19° sensors always read ~18m (side barriers) — also unreliable
      Only track[9] (0°) reads the actual road ahead reliably.

    safe_speed = sqrt(ahead * GRIP_K)
      ahead=200m (long straight) → 245 km/h  → no braking
      ahead=100m (fast corner)   → 173 km/h  → light braking
      ahead= 50m (medium corner) → 122 km/h  → moderate braking
      ahead= 20m (tight chicane) →  77 km/h  → strong braking
      ahead= 10m (hairpin)       →  55 km/h  → hard braking

    Gate: only activate if ahead < 150m AND speed > safe_speed.
    Above 150m the road is clear — never restrict speed.
    """
    track = S.get('track', None)
    if not isinstance(track, list) or len(track) < 10:
        return TARGET_SPEED

    ahead = track[9]

    # Road clear ahead — no restriction (gate at 95 m: earlier warning for final chicane)
    if ahead >= 95:
        return TARGET_SPEED

    return min(TARGET_SPEED, math.sqrt(ahead * GRIP_K))


def calculate_throttle(S, R):
    """Simple speed-governed throttle.
    Target is always TARGET_SPEED — sensors don't reduce it.
    Braking is handled entirely by apply_brakes().
    corner_speed_factor only scales throttle when off-track or mis-aligned.
    """
    cf    = corner_speed_factor(S)
    speed = S.get('speedX', 0)

    if speed < TARGET_SPEED - abs(R['steer']) * 20:   # reduced steer penalty 50→20
        accel = min(1.0, R['accel'] + 0.8)   # faster ramp-up out of corners
    else:
        accel = max(0.0, R['accel'] - 0.4)

    if speed < 10:
        accel += 1.0 / (speed + 0.1)

    # Only scale down when off-track or mis-aligned
    return max(0.0, min(1.0, accel * cf))


def apply_brakes(S, R):
    """Brake using only track[9] (0°) — the only reliable sensor on 12m-wide track."""
    speed     = S.get('speedX', 0)
    track_pos = S.get('trackPos', 0)
    track     = S.get('track', None)

    # NEVER brake if already slow — prevents car getting stuck at 0 km/h
    if speed < 30:
        return 0.0

    # 1. Off-track: brake to recover position
    abs_pos = abs(track_pos)
    if abs_pos > 0.9:
        return min(0.5, (abs_pos - 0.9) * 2.0)

    # 2. Corner entry speed — based purely on track[9]
    safe_speed = corner_entry_speed(S)
    if speed > safe_speed + 5:   # 5 km/h hysteresis (tighter than before)
        overspeed = speed - safe_speed
        # Ramp: /90 — slightly gentler initial bite, same peak
        brake = min(0.8, overspeed / 90.0)
        if brake > 0.05:
            return brake

    # 3. Mis-alignment brake
    angle = abs(S.get('angle', 0))
    if angle > BRAKE_THRESHOLD:
        excess = angle - BRAKE_THRESHOLD
        return min(0.4, 0.1 + excess * 0.3)

    return 0.0


def shift_gears(S):
    """Shift up through gears based on speed thresholds.

    Guard: if the car is rolling backward (speedX < 0) keep gear 1 so the
    engine doesn't fight reverse motion — this prevents the 4 km/h deadlock
    where recover_if_stuck exits with gear=-1 and the car cannot accelerate.
    """
    speed = S.get('speedX', 0)
    if speed < 0:
        return 1
    gear = 1
    for i, spd in enumerate(GEAR_SPEEDS):
        if speed > spd:
            gear = i + 1
    return min(gear, 6)


def traction_control(S, accel):
    """Reduce throttle when rear wheels spin faster than fronts (wheelspin)."""
    if not ENABLE_TRACTION_CONTROL:
        return accel
    wsv = S.get('wheelSpinVel', [0, 0, 0, 0])
    if len(wsv) >= 4:
        spin_diff = (wsv[2] + wsv[3]) - (wsv[0] + wsv[1])
        if spin_diff > 2:
            accel -= 0.15 * min(spin_diff / 5.0, 1.0)  # proportional reduction
    return max(0.0, accel)


def recover_if_stuck(S, R):
    """Multi-phase recovery when the car is stuck or deep off-track.

    Returns True while recovery is active.

    Fix: after Phase 3 completes, run a mandatory 40-step cool-down
    (still returning True) before handing back to drive_modular.
    This prevents the 1-step gap where _timer resets to 0 but the car
    is still off-track, which caused drive_modular to immediately apply
    full throttle into the wall on the very next tick.
    """
    stuck     = S.get('stucktimer', 0)
    track_pos = S.get('trackPos', 0)
    angle     = S.get('angle', 0)
    speed     = S.get('speedX', 0)

    # Use a module-level timer independent of stucktimer
    if not hasattr(recover_if_stuck, '_timer'):
        recover_if_stuck._timer = 0

    # Trigger conditions: stuck by server OR deeply off-track
    triggered = (stuck and stuck > 50) or abs(track_pos) > 1.5

    if triggered or (0 < recover_if_stuck._timer < 280):
        # Keep running if mid-sequence even if trigger just cleared
        recover_if_stuck._timer += 1
        t = recover_if_stuck._timer
        R['brake'] = 0.0

        if t < 80:
            # Phase 1: reverse, steer hard toward track
            R['gear']  = -1
            R['steer'] = max(-1.0, min(1.0, -track_pos * 1.0))
            R['accel'] = 0.8
        elif t < 160:
            # Phase 2: still reversing, align wheels with track angle
            R['gear']  = -1
            R['steer'] = max(-1.0, min(1.0, -angle * 2.0))
            R['accel'] = 0.8
        elif t < 240:
            # Phase 3: drive forward toward center
            R['gear']  = 1
            R['steer'] = max(-1.0, min(1.0, -track_pos * 1.0))
            R['accel'] = 0.8
        else:
            # Cool-down (steps 240-279): gentle forward, let drive_modular
            # take over only once the car is clearly back on track.
            R['gear']  = 1
            R['steer'] = max(-1.0, min(1.0, -track_pos * 0.5))
            R['accel'] = 0.3
            R['brake'] = 0.0
            if t >= 280:
                recover_if_stuck._timer = 0   # full reset, ready for next incident
        return True
    else:
        # Not stuck — reset timer so next incident starts fresh
        recover_if_stuck._timer = 0
        return False


# ==================== UTILITY ====================

def clip(v, lo, hi):
    if v < lo: return lo
    elif v > hi: return hi
    else: return v


def bargraph(x, mn, mx, w, c='X'):
    '''ASCII art bar graph for telemetry display.'''
    if not w: return ''
    if x < mn: x = mn
    if x > mx: x = mx
    tx = mx - mn
    if tx <= 0: return 'backwards'
    upw = tx / float(w)
    if upw <= 0: return 'what?'
    negpu, pospu, negnonpu, posnonpu = 0, 0, 0, 0
    if mn < 0:
        if x < 0:
            negpu     = -x + min(0, mx)
            negnonpu  = -mn + x
        else:
            negnonpu  = -mn + min(0, mx)
    if mx > 0:
        if x > 0:
            pospu     = x - max(0, mn)
            posnonpu  = mx - x
        else:
            posnonpu  = mx - max(0, mn)
    nnc = int(negnonpu / upw) * '-'
    npc = int(negpu    / upw) * c
    ppc = int(pospu    / upw) * c
    pnc = int(posnonpu / upw) * '_'
    return '[%s]' % (nnc + npc + ppc + pnc)


# ==================== CLIENT ====================

class Client():
    def __init__(self, H=None, p=None, i=None, e=None, t=None, s=None, d=None, vision=False):
        self.vision = vision
        self.host        = 'localhost'
        self.port        = 3001
        self.sid         = 'SCR'
        self.maxEpisodes = 1
        self.trackname   = 'unknown'
        self.stage       = 3
        self.debug       = False
        self.maxSteps    = 100000
        self.parse_the_command_line()
        if H: self.host        = H
        if p: self.port        = p
        if i: self.sid         = i
        if e: self.maxEpisodes = e
        if t: self.trackname   = t
        if s: self.stage       = s
        if d: self.debug       = d
        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            print('Error: Could not create socket...')
            sys.exit(-1)
        self.so.settimeout(1)

        # FIX: was n_fail < 0 (off-by-one); now n_fail <= 0 → exactly 5 retries
        n_fail = 5
        while True:
            a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = '%s(init %s)' % (self.sid, a)
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                sys.exit(-1)
            sockdata = str()
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail <= 0:   # FIX: was < 0
                    print("relaunch torcs")
                    os.system('pkill torcs')
                    time.sleep(1.0)
                    if self.vision is False:
                        os.system('torcs -nofuel -nodamage -nolaptime &')
                    else:
                        os.system('torcs -nofuel -nodamage -nolaptime -vision &')
                    time.sleep(1.0)
                    os.system('sh autostart.sh')
                    n_fail = 5
                n_fail -= 1
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(
                sys.argv[1:], 'H:p:i:m:e:t:s:f:dhv',
                ['host=', 'port=', 'id=', 'steps=',
                 'episodes=', 'track=', 'stage=', 'parameters=',
                 'debug', 'help', 'version'])
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] in ('-h', '--help'):
                    print(usage); sys.exit(0)
                if opt[0] in ('-d', '--debug'):
                    self.debug = True
                if opt[0] in ('-H', '--host'):
                    self.host = opt[1]
                if opt[0] in ('-i', '--id'):
                    self.sid = opt[1]
                if opt[0] in ('-t', '--track'):
                    self.trackname = opt[1]
                if opt[0] in ('-s', '--stage'):
                    self.stage = int(opt[1])
                ######### REQUIRED FOR RACE MODE - DO NOT CHANGE OR REMOVE #########
                if opt[0] in ('-p', '--port'):
                    self.port = int(opt[1])
                ######## END OF REQUIRED #########
                if opt[0] in ('-e', '--episodes'):
                    self.maxEpisodes = int(opt[1])
                if opt[0] in ('-m', '--steps'):
                    self.maxSteps = int(opt[1])
                ######### REQUIRED FOR PRACTICE MODE - CAN CHANGE BUT DO NOT REMOVE #########
                if opt[0] in ('-f', '--parameters'):
                    global TARGET_SPEED, STEER_GAIN, CENTERING_GAIN
                    global BRAKE_THRESHOLD, GEAR_SPEEDS, ENABLE_TRACTION_CONTROL
                    j = json.loads(opt[1])
                    TARGET_SPEED          = int(j['target_speed'])
                    STEER_GAIN            = int(j['steer_gain'])
                    CENTERING_GAIN        = float(j['centering_gain'])
                    BRAKE_THRESHOLD       = float(j['brake_threshold'])
                    GEAR_SPEEDS           = [int(x) for x in j['gear_thresholds']]
                    ENABLE_TRACTION_CONTROL = bool(j['traction_control'])
                    print(f"Running with parameters:\n"
                          f"  target_speed={TARGET_SPEED}\n"
                          f"  steer_gain={STEER_GAIN}\n"
                          f"  centering_gain={CENTERING_GAIN}\n"
                          f"  brake_threshold={BRAKE_THRESHOLD}\n"
                          f"  gear_speeds={GEAR_SPEEDS}\n"
                          f"  traction_control={ENABLE_TRACTION_CONTROL}")
                ########## END OF REQUIRED #########
                if opt[0] in ('-v', '--version'):
                    print('%s %s' % (sys.argv[0], version)); sys.exit(0)
        except ValueError as why:
            print('Bad parameter \'%s\' for option %s: %s\n%s' % (
                opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)

    ######### REQUIRED FOR RACE MODE - DO NOT CHANGE OR REMOVE #########
    def get_servers_input(self):
        '''Server's input is stored in a ServerState object.'''
        if not self.so: return
        sockdata = str()

        # Counter resets on every good packet received.
        # Only shuts down after 30 *consecutive* timeouts (30 s at settimeout=1),
        # meaning a genuine server disappearance — not a normal brief gap.
        consecutive_misses = 0
        max_consecutive    = 30

        while True:
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
                consecutive_misses = 0   # reset on every successful receive
            except socket.error:
                print('.', end=' ')
                consecutive_misses += 1
                if consecutive_misses >= max_consecutive:
                    print("\nServer unreachable after %d consecutive timeouts. Shutting down." % max_consecutive)
                    self.shutdown()
                    return
                continue
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                print("Server has stopped the race on %d. You were in %d place."
                      % (self.port, self.S.d.get('racePos', 0)))
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                print("Server has restarted the race on %d." % self.port)
                self.shutdown()
                return
            elif not sockdata:
                continue
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H")
                    print(self.S)
                break
    ########## END OF REQUIRED #########

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            # FIX: emsg[0]/emsg[1] raises TypeError in Python 3
            print("Error sending to server: %s Message %s" % (emsg.strerror, str(emsg.errno)))
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())

    def shutdown(self):
        if not self.so: return
        print("Race terminated or %d steps elapsed. Shutting down %d."
              % (self.maxSteps, self.port))
        self.so.close()
        self.so = None


# ==================== SERVER STATE ====================

class ServerState():
    '''What the server is reporting right now.'''
    def __init__(self):
        self.servstr = str()
        self.d       = dict()

    def parse_server_str(self, server_string):
        self.servstr = server_string.strip()[:-1]
        sslisted = self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = destringify(w[1:])

    def __repr__(self):
        # FIX: dead code block after return removed (was unreachable)
        return self.fancyout()

    def fancyout(self):
        '''Specialty output for useful ServerState monitoring.'''
        out = str()
        sensors = [
            'stucktimer',
            'fuel',
            'distRaced',
            'distFromStart',
            'opponents',
            'wheelSpinVel',
            'z',
            'speedZ',
            'speedY',
            'speedX',
            # FIX: removed 'targetSpeed' — not a real TORCS sensor
            'rpm',
            'skid',
            'slip',
            'track',
            'trackPos',
            'angle',
        ]
        for k in sensors:
            if type(self.d.get(k)) is list:
                if k == 'track':
                    strout = str()
                    raw_tsens = ['%.1f' % x for x in self.d['track']]
                    strout += (' '.join(raw_tsens[:9]) + '_' +
                               raw_tsens[9] + '_' +
                               ' '.join(raw_tsens[10:]))
                elif k == 'opponents':
                    strout = str()
                    for osensor in self.d['opponents']:
                        if   osensor > 190: oc = '_'
                        elif osensor >  90: oc = '.'
                        elif osensor >  39: oc = chr(int(osensor / 2) + 97 - 19)
                        elif osensor >  13: oc = chr(int(osensor) + 65 - 13)
                        elif osensor >   3: oc = chr(int(osensor) + 48 - 3)
                        else:               oc = '?'
                        strout += oc
                    strout = ' -> ' + strout[:18] + ' ' + strout[18:] + ' <-'
                else:
                    strlist = [str(i) for i in self.d[k]]
                    strout  = ', '.join(strlist)
            else:
                if k == 'gear':
                    gs  = '_._._._._._._._._'
                    p   = int(self.d['gear']) * 2 + 2
                    l   = '%d' % self.d['gear']
                    if l == '-1': l = 'R'
                    if l == '0':  l = 'N'
                    strout = gs[:p] + '(%s)' % l + gs[p + 3:]
                elif k == 'damage':
                    strout = '%6.0f %s' % (self.d[k], bargraph(self.d[k], 0, 10000, 50, '~'))
                elif k == 'fuel':
                    strout = '%6.0f %s' % (self.d[k], bargraph(self.d[k], 0, 100, 50, 'f'))
                elif k == 'speedX':
                    cx = 'R' if self.d[k] < 0 else 'X'
                    strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k], -30, 300, 50, cx))
                elif k == 'speedY':
                    strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k] * -1, -25, 25, 50, 'Y'))
                elif k == 'speedZ':
                    strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k], -13, 13, 50, 'Z'))
                elif k == 'z':
                    strout = '%6.3f %s' % (self.d[k], bargraph(self.d[k], .3, .5, 50, 'z'))
                elif k == 'trackPos':
                    cx = '>' if self.d[k] < 0 else '<'
                    strout = '%6.3f %s' % (self.d[k], bargraph(self.d[k] * -1, -1, 1, 50, cx))
                elif k == 'stucktimer':
                    if self.d[k]:
                        strout = '%3d %s' % (self.d[k], bargraph(self.d[k], 0, 300, 50, "'"))
                    else:
                        strout = 'Not stuck!'
                elif k == 'rpm':
                    g = self.d.get('gear', 0)
                    g = 'R' if g < 0 else '%1d' % g
                    strout = bargraph(self.d[k], 0, 10000, 50, g)
                elif k == 'angle':
                    asyms = [
                        "  !  ", ".|'  ", "./'  ", "_.-  ", ".--  ", "..-  ",
                        "---  ", ".__  ", "-._  ", "'-.  ", "'\.  ", "'|.  ",
                        "  |  ", "  .|'", "  ./'", "  .-'", "  _.-", "  __.",
                        "  ---", "  --.", "  -._", "  -..", "  '\.", "  '|."
                    ]
                    rad   = self.d[k]
                    deg   = int(rad * 180 / PI)
                    symno = int(.5 + (rad + PI) / (PI / 12)) % (len(asyms) - 1)
                    strout = '%5.2f %3d (%s)' % (rad, deg, asyms[symno])
                elif k == 'skid':
                    # FIX: guard against missing keys
                    wsv  = self.d.get('wheelSpinVel', [0, 0, 0, 0])
                    fwrps = wsv[0] if wsv else 0
                    skid  = 0
                    if fwrps:
                        skid = .5555555555 * self.d.get('speedX', 0) / fwrps - .66124
                    strout = bargraph(skid, -.05, .4, 50, '*')
                elif k == 'slip':
                    # FIX: guard against missing keys
                    wsv  = self.d.get('wheelSpinVel', [0, 0, 0, 0])
                    slip  = 0
                    if wsv and wsv[0]:
                        slip = (wsv[2] + wsv[3]) - (wsv[0] + wsv[1])
                    strout = bargraph(slip, -5, 150, 50, '@')
                else:
                    strout = str(self.d.get(k, 'N/A'))
            out += "%s: %s\n" % (k, strout)
        return out


# ==================== DRIVER ACTION ====================

class DriverAction():
    '''What the driver is intending to do (i.e. send to the server).'''
    def __init__(self):
        self.actionstr = str()
        self.d = {
            'accel':  0.2,
            'brake':  0,
            'clutch': 0,
            'gear':   1,
            'steer':  0,
            'focus':  [-90, -45, 0, 45, 90],
            'meta':   0
        }

    def clip_to_limits(self):
        self.d['steer']  = clip(self.d['steer'],  -1, 1)
        self.d['brake']  = clip(self.d['brake'],   0, 1)
        self.d['accel']  = clip(self.d['accel'],   0, 1)
        self.d['clutch'] = clip(self.d['clutch'],  0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0
        if self.d['meta'] not in [0, 1]:
            self.d['meta'] = 0
        if (type(self.d['focus']) is not list or
                min(self.d['focus']) < -180 or max(self.d['focus']) > 180):
            self.d['focus'] = 0

    def __repr__(self):
        self.clip_to_limits()
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if type(v) is not list:
                out += '%.3f' % v
            else:
                out += ' '.join([str(x) for x in v])
            out += ')'
        return out
        # FIX: removed unreachable second `return out+'\n'`

    def fancyout(self):
        out = str()
        od  = self.d.copy()
        od.pop('gear',  '')
        od.pop('meta',  '')
        od.pop('focus', '')
        for k in sorted(od):
            if k in ('clutch', 'brake', 'accel'):
                strout = '%6.3f %s' % (od[k], bargraph(od[k], 0, 1, 50, k[0].upper()))
            elif k == 'steer':
                strout = '%6.3f %s' % (od[k], bargraph(od[k] * -1, -1, 1, 50, 'S'))
            else:
                strout = str(od[k])
            out += "%s: %s\n" % (k, strout)
        return out


# ==================== MISC UTILS ====================

def destringify(s):
    '''Convert a string (or list of strings) to float value(s).'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]


# ==================== MAIN DRIVE FUNCTION ====================
# MISSION 1: sub-2-minute lap  → high TARGET_SPEED + aggressive gears
# MISSION 2: no damage         → corner look-ahead, early braking, stuck recovery

def drive_modular(c):
    S, R = c.S.d, c.R.d

    # 1. Stuck recovery – overrides everything else if triggered
    if recover_if_stuck(S, R):
        return

    # 2. Steering
    R['steer'] = calculate_steering(S)

    # 3. Braking
    R['brake'] = apply_brakes(S, R)

    # 4. Throttle — always recalculate fresh; never let brake accumulate accel down
    R['accel'] = calculate_throttle(S, R)
    # Zero throttle whenever any brake is applied (prevents simultaneous throttle+brake)
    if R['brake'] > 0.0:
        R['accel'] = 0.0

    # 5. Traction control
    R['accel'] = traction_control(S, R['accel'])

    # 6. Gears
    R['gear'] = shift_gears(S)


# ==================== MAIN ====================
if __name__ == "__main__":
    #### DO NOT HARDCODE THE PORT
    C = Client()
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        if not C.so:      # server sent shutdown/restart or became unreachable
            break
        drive_modular(C)
        C.respond_to_server()
    C.shutdown()
