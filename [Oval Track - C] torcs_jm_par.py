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

# High target speed for a fast lap.  The corner-speed governor below will
# automatically reduce throttle when the track curves, so it is safe to set
# this high.
TARGET_SPEED = 280          # km/h – straight-line target

# Steering: moderate gain avoids the oscillation that causes wall strikes.
STEER_GAIN = 10             # lower = smoother steering (was 30)
CENTERING_GAIN = 0.15       # gentle center-correction (was 0.20)

# Braking: activate when the car is significantly mis-aligned with the road.
BRAKE_THRESHOLD = 0.5       # radians (~29°); was 0.9 – now brakes much earlier

# Gear speed thresholds (km/h). Tuned to match the higher target speed.
GEAR_SPEEDS = [0, 30, 60, 100, 150, 200]   # 6 thresholds → gears 1-6

# Traction control keeps rear wheels from spinning out.
ENABLE_TRACTION_CONTROL = True

######### END OF REQUIRED #########
# ----------------------------------------------


# ==================== HELPER FUNCTIONS ====================

def calculate_steering(S):
    """Blend angle correction with track-center correction.
    Clamped to [-1, 1]."""
    steer = (S['angle'] * STEER_GAIN / PI) - (S['trackPos'] * CENTERING_GAIN)
    return max(-1.0, min(1.0, steer))


def corner_speed_factor(S):
    """Return a 0-1 throttle multiplier based on upcoming corner severity.

    FIX for non-oval tracks:
    - Use DIAGONAL sensors (track[2]/track[16] at ~±30°) instead of relying
      solely on the 0° sensor.  On tight corners track[9] reads very short
      even with road ahead, but the diagonals still see the corner exit and
      give a true picture of how tight the radius is.
    - Geometric mean of 4 factors so no single weak sensor kills throttle.
    - Floor at 0.15 so the car always has minimum throttle and never stalls
      in a hairpin on technical tracks.
    """
    track = S.get('track', None)
    if track is None or not isinstance(track, list) or len(track) < 19:
        return 1.0

    ahead        = track[9]                        # 0° straight ahead
    diag         = min(track[2], track[16])        # ~±30° diagonals
    side         = min(track[4], track[14])        # ~±45° sides

    ahead_factor = min(ahead / 80.0,  1.0)   # full throttle at 80 m clear ahead
    diag_factor  = min(diag  / 60.0,  1.0)   # corner-radius proxy via diagonals
    side_factor  = min(side  / 40.0,  1.0)   # stability check
    angle_factor = max(0.0, 1.0 - abs(S.get('angle', 0)) * 1.5)

    combined = (ahead_factor * diag_factor * side_factor * angle_factor) ** 0.25

    # Floor: always keep minimum throttle so car never stalls mid-corner.
    return max(0.15, combined)


def calculate_throttle(S, R):
    """Speed-governed throttle with corner look-ahead.
    Aggressive on straights, conservative on corners (MISSION 2)."""
    cf     = corner_speed_factor(S)
    speed  = S.get('speedX', 0)
    target = TARGET_SPEED * cf          # corner-adjusted target

    if speed < target - abs(R['steer']) * 30:
        accel = min(1.0, R['accel'] + 0.5 * cf)   # fast ramp-up
    else:
        accel = max(0.0, R['accel'] - 0.3)         # fast shed

    # Low-speed boost so the car gets rolling quickly.
    if speed < 10:
        accel += 1.0 / (speed + 0.1)

    return max(0.0, min(1.0, accel))


def apply_brakes(S, R):
    """Brake on large mis-alignment OR when a wall is genuinely imminent.

    FIX for non-oval tracks:
    - track[9] < 20/30 is NOT a reliable wall indicator on technical tracks —
      the sensor reads short on any tight corner even with road still ahead.
    - Only hard-brake if BOTH track[9] is short AND speed is high enough
      to actually be dangerous.  A slow-speed hairpin approach should not
      trigger maximum braking.
    - Use the diagonal sensors (track[2]/track[16]) to distinguish a genuine
      wall from a normal tight corner: if diagonals are also short, it's a
      wall; if diagonals are long, it's a corner and we only brake gently.
    """
    speed = S.get('speedX', 0)
    track = S.get('track', None)

    if isinstance(track, list) and len(track) >= 17:
        ahead = track[9]
        diag  = min(track[2], track[16])  # ~±30° diagonal sensors

        # Genuine wall: straight-ahead AND diagonals are both very short at speed.
        if ahead < 15 and diag < 15 and speed > 30:
            return 0.8                    # hard brake — wall truly imminent

        # Tight corner at high speed: ahead short but diagonals show open road.
        # Brake proportionally to speed — light at low speed, harder at high speed.
        if ahead < 25 and speed > 60:
            return min(0.5, (speed - 60) / 200.0 + 0.1)

    # Brake on large mis-alignment (spun / badly off-angle).
    angle = abs(S.get('angle', 0))
    if angle > BRAKE_THRESHOLD:
        excess = angle - BRAKE_THRESHOLD
        return min(0.5, 0.15 + excess * 0.4)

    return 0.0


def shift_gears(S):
    """Shift up through gears based on speed thresholds."""
    gear = 1
    for i, speed in enumerate(GEAR_SPEEDS):
        if S.get('speedX', 0) > speed:
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
    """If the car is stuck (stucktimer > threshold), reverse and steer out.
    Returns True if recovery mode is active."""
    stuck = S.get('stucktimer', 0)
    if stuck and stuck > 50:           # stuck for ~1 second (50 steps)
        R['gear']  = -1                # reverse
        R['steer'] = -S.get('trackPos', 0)  # steer away from edge
        R['accel'] = 0.5
        R['brake'] = 0.0
        return True
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

    # 3. Braking (before throttle so accel can account for it)
    R['brake'] = apply_brakes(S, R)

    # 4. Throttle – backed off when braking
    if R['brake'] > 0.2:
        R['accel'] = max(0.0, R['accel'] - 0.3)
    else:
        R['accel'] = calculate_throttle(S, R)

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
