"""Handshake demo: a seated person offers their hand; the iiwa sees it with its head
camera, reaches out, the Aero Hand grips it, they shake, and it lets go.

  python handshake.py             # interactive viewer + control window (Start/Stop)
  python handshake.py --check     # headless run of a fixed episode script: detection,
                                  # trigger and handshake metrics, handshake_frames.png

The person (human_agent.py) behaves on their own: rests, offers a hand, waves, hesitates,
or offers and pulls back. The robot detects the hand with a D435-like head camera
(perception.py) and responds only to a hand that is offered: inside the meeting zone and
held still. The partner hand (partner_hand.py) is an anatomical right hand with a soft
palm; it is welded to a mocap target the person moves, and to the robot hand while they
shake.
"""

import argparse
import os

import mujoco
import numpy as np

from head_camera import add_head_camera
from human_agent import HumanAgent
from human_body import HumanBody, add_human_body
from partner_hand import PRESS_TARGET, add_partner_hand
from partner_hand import grip_targets as partner_grip_targets
from perception import HandDetector
from tactile import TactileSensors, add_tactile, heat_rgb_u8

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(HERE, "scene.xml")
MEDIA = os.path.join(HERE, "media")      # check outputs, images and videos

# Arm posture for the handshake (IK reference): elbow up and out to the robot's right,
# wrist well away from the iiwa's wrist singularity (joint 6 near 0). The earlier posture
# had joint 6 at -0.1..-0.2: for hands offered high and far out, a 22 mm shake then needed
# joint 5 at up to 3 rad/s and the hand lagged into the person's. This one keeps the
# Jacobian's smallest singular value >= 0.12 over the whole offer zone (was 0.03).
SHAKE_Q_REF = np.array([-1.06, 0.76, -1.48, 1.82, 1.06, 0.75, -0.11])
# Home: same elbow family, arm tucked low on the robot's right: out of the head camera's
# view of where the person offers their hand (hides ~2% of it; the original home, hand
# 0.6 m forward, hid 26-100%) and out of the hand's way.
ARM_HOME = np.array([-1.53, 0.76, -1.89, 1.92, 1.06, 1.15, 0.51])
MOUNT = "hand_tetheria_mount"
PRESS_ACT = "act_partner_press"

# Handshake pose of the hand bracket, in world coordinates.
# Bracket axes: +x = finger direction, -y = thumb side, +z = palm normal.
# Handshake: fingers forward (+x), thumb up (+z), palm facing the robot's left (+y),
# tilted 15 deg so the fingers point slightly down, like a person's would.
_tilt = np.radians(15)
SHAKE_ROT = np.array([[np.cos(_tilt), 0, np.sin(_tilt)],
                      [0, 1, 0],
                      [-np.sin(_tilt), 0, np.cos(_tilt)]]) @ np.array(
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]])
PALM_CENTER_LOCAL = np.array([0.11, 0.01, -0.01])   # palm face centre in bracket frame
# Where the palms meet. 0.68 m out (was 0.62): at 0.62 the arm's elbow was within 0.11
# rad of its limit and hands offered closer or lower than nominal were unreachable
# (85% of the offer zone); from 0.68 the whole zone is reachable.
SHAKE_PALM_WORLD = np.array([0.68, 0.0, 0.55])
SHAKE_MOUNT_POS = SHAKE_PALM_WORLD - SHAKE_ROT @ PALM_CENTER_LOCAL

# Partner hand geometry lives in partner_hand.py (its frame = robot bracket frame at the
# shake pose). It comes in from this far along PARTNER_APPROACH_DIR.
PARTNER_APPROACH = 0.25                              # m
# Approach direction in the bracket frame (+x along the robot fingers, +z off the palm):
# coming mostly from the palm side (the person brings their palm in from beside the robot
# hand) lets it settle onto the palm without sweeping through the open fingers.
PARTNER_APPROACH_DIR = np.array([0.146, 0.0, 0.989])   # mostly from the palm side (~82 deg)

# Hand actuators (tendon lengths): high = open, low = closed.
# Order: index, middle, ring, pinky, thumb abduction, thumb flex 1, thumb flex 2.
HAND_OPEN = np.array([0.11, 0.11, 0.11, 0.11, -0.1, 0.038, 0.112])     # thumb out of the way
# Fingers: full close, contact stops them. Thumb: abducted to 1.4 rad so it rises off the
# palm and rests on the back of the partner's hand (at 0.5 it stayed at palm height and
# jammed into the partner's thumb). Robust for abduction 1.3-1.5.
HAND_GRIP = np.array([0.059, 0.059, 0.059, 0.059, 1.4, 0.028, 0.085])
HAND_FORCE_LIMIT = 20.0   # N per tendon actuator (assumed servo limit; source model has none)

SHAKE_AMPLITUDE = 0.022   # m, vertical: a gentle pump (was 35 mm)
SHAKE_FREQ = 1.25         # Hz (was 1.6): peak acceleration ~1.4 m/s^2 instead of ~3.5
SHAKE_CYCLES = 3

# Robot motion (s): home -> reach (joint space) -> approach (straight line into the hand).
T_REACH, T_APPROACH = 2.2, 1.2
PRE_DIST = 0.15       # m, the reach ends this far short of the hand, on its palm side
T_RETREAT, T_RETURN = 1.0, 2.2
COOLDOWN = 1.0        # s at home before responding again
ABORT_DIST = 0.06     # m, hand moved this far from where the robot committed -> abort
LOST_ABORT = 1.0      # s without a hand track while reaching -> abort
ADAPT_TIME = 0.3      # s, the person's hand settles onto the robot's at the start of partner_grip
CONTACT_N = 1.5       # N on the palm Velostat: someone is actually holding the robot's hand
CONTACT_WAIT = 0.8    # s into partner_grip without that -> nobody there, back off

# Once the robot's hand has arrived (s). The person closes their hand fully first
# (partner_grip), then the robot wraps (grip), then they shake. Letting go mirrors it:
# the robot opens first (release), then the person (partner_release).
ENGAGE = [("partner_grip", 1.2), ("grip", 1.6), ("shake", SHAKE_CYCLES / SHAKE_FREQ),
          ("hold", 0.5), ("release", 0.8), ("partner_release", 0.8)]
GRIPPED = ("grip", "shake", "hold", "release")             # partner hand fully closed


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(SCENE_XML)
    # Gravity compensation so the position-controlled arm holds its targets.
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    add_tactile(spec)   # Velostat palm matrix + Hall pads (see tactile.py)
    # Partner hand: a free body (frame = robot bracket frame at the shake pose) held by
    # two soft welds that the script switches between:
    #   partner_to_target: to a mocap target -> scripted approach / retreat
    #   partner_to_robot:  to the robot bracket -> while gripped, the person's relaxed arm
    #                      simply goes along with the shake (no fighting through contacts)
    spec.worldbody.add_body(name="partner_target", mocap=True)
    # The free root carries only the welds' anchor (the hand hangs off it on a slide), so
    # it gets a small explicit inertia: the forearm's share of the person's arm.
    partner = spec.worldbody.add_body(name="partner", gravcomp=1.0, mass=0.2,
                                      inertia=[2e-4, 2e-4, 2e-4], explicitinertial=True)
    partner.add_freejoint(name="partner_free")
    add_partner_hand(spec, partner)
    add_human_body(spec)        # seated person around the partner hand (visual only)
    add_head_camera(spec)       # robot "head": D435-like RGB-D camera
    # The Aero Hand's contacts are soft (tuned for 10 ms steps and light touches); a
    # firm handshake would sink through them. Give the partner's geoms priority and
    # stiff contact parameters so every robot<->partner contact uses these. The palm's
    # softness comes from its spring-loaded tissue pads (partner_hand.py), not from
    # mushy contacts, so the contacts themselves can all be firm.
    for geom in spec.geoms:
        if geom.name.startswith("partner"):
            geom.priority = 1
            geom.solref = [0.005, 1.0]
            geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    # Realistic tendon force limit for the Aero Hand servos (the source model has none,
    # which lets a closing finger push with hundreds of newtons).
    for act in spec.actuators:
        if act.name.startswith("hand_right_") and act.name != "hand_right_thumb_A_cmc_abd":
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-HAND_FORCE_LIMIT, HAND_FORCE_LIMIT]
    data = np.zeros(12)
    data[6] = 1.0     # identity relative pose
    data[10] = 5.0    # torquescale: rotationally stiff, so the gripped hand doesn't wobble
    for name, other in (("partner_to_target", "partner_target"), ("partner_to_robot", MOUNT)):
        weld = spec.add_equality(name=name, type=mujoco.mjtEq.mjEQ_WELD,
                                 objtype=mujoco.mjtObj.mjOBJ_BODY, name1="partner",
                                 name2=other, solref=[0.004, 1.0])
        weld.data = data
        weld.active = name == "partner_to_target"
    # Home keyframe: the tucked arm, plus the partner's free joint (the Session moves the
    # hand onto the person's thigh); every other partner joint starts at zero.
    key = spec.key("home")
    key.qpos = list(ARM_HOME) + list(key.qpos)[7:]
    key.ctrl = list(ARM_HOME) + list(key.ctrl)[7:]
    start = approach_pos(1.0)
    n_q = sum({mujoco.mjtJoint.mjJNT_HINGE: 1, mujoco.mjtJoint.mjJNT_SLIDE: 1}.get(j.type, 0)
              for j in spec.joints if j.name.startswith("partner_"))
    n_act = sum(a.name.startswith("act_partner_") for a in spec.actuators)
    key.qpos = list(key.qpos) + list(start) + list(mat2quat(SHAKE_ROT)) + [0.0] * n_q
    key.ctrl = list(key.ctrl) + [0.0] * n_act
    # Mocap bodies: partner_target, then the person's right upper arm (re-posed every step).
    key.mpos = list(start) + [0.0, 0.0, 0.0]
    key.mquat = list(mat2quat(SHAKE_ROT)) + [1.0, 0.0, 0.0, 0.0]
    return spec.compile()


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


SHAKE_QUAT = mat2quat(SHAKE_ROT)


def solve_ik(m, d_ik, q_init, pos, quat, q_ref, iters=60):
    """Damped least-squares IK on the 7 arm joints for the bracket body pose."""
    bid = m.body(MOUNT).id
    q = q_init.copy()
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    lo, hi = m.jnt_range[:7].T
    for _ in range(iters):
        d_ik.qpos[:7] = q
        mujoco.mj_kinematics(m, d_ik)
        mujoco.mj_comPos(m, d_ik)
        err_p = pos - d_ik.xpos[bid]
        err_r = np.zeros(3)
        mujoco.mju_subQuat(err_r, quat, d_ik.xquat[bid])
        err_r = d_ik.xmat[bid].reshape(3, 3) @ err_r   # body frame -> world (Jacobian frame)
        err = np.concatenate([err_p, err_r])
        if np.linalg.norm(err_p) < 1e-4 and np.linalg.norm(err_r) < 1e-3:
            break
        mujoco.mj_jacBody(m, d_ik, jacp, jacr, bid)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        JJt = J @ J.T + 1e-4 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        # Null-space pull toward a comfortable posture.
        N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
        dq += N @ (0.1 * (q_ref - q))
        q = np.clip(q + dq, lo + 0.02, hi - 0.02)
    return q


def pose_error(m, d_ik, q, pos, quat):
    d_ik.qpos[:7] = q
    mujoco.mj_kinematics(m, d_ik)
    bid = m.body(MOUNT).id
    err_r = np.zeros(3)
    mujoco.mju_subQuat(err_r, quat, d_ik.xquat[bid])
    return np.linalg.norm(pos - d_ik.xpos[bid]), np.linalg.norm(err_r)


def solve_ik_global(m, d_ik, pos, quat, q_ref, seeds=200, rng_seed=0):
    """Multi-start IK: keeps the exact solution closest to q_ref (for a calm, repeatable pose)."""
    rng = np.random.default_rng(rng_seed)
    lo, hi = m.jnt_range[:7].T
    best, best_cost = None, np.inf
    for i in range(seeds):
        q0 = q_ref if i == 0 else rng.uniform(lo * 0.8, hi * 0.8)
        q = solve_ik(m, d_ik, q0, pos, quat, q_ref, 200)
        ep, er = pose_error(m, d_ik, q, pos, quat)
        if ep > 1e-3 or er > 1e-2:
            continue
        # Prefer staying near the reference and away from joint limits.
        margin = np.min(np.minimum(q - lo, hi - q))
        cost = np.linalg.norm(q - q_ref) - 0.5 * min(margin, 0.5)
        if cost < best_cost:
            best, best_cost = q, cost
    if best is None:
        raise RuntimeError("IK failed: handshake pose unreachable")
    return best


def approach_pos(a):
    """Partner position at approach fraction a (0 = shaking pose, 1 = start)."""
    direction = SHAKE_ROT @ (PARTNER_APPROACH_DIR / np.linalg.norm(PARTNER_APPROACH_DIR))
    return SHAKE_MOUNT_POS + PARTNER_APPROACH * a * direction


def shake_offset(s):
    """Vertical shake displacement at shake-phase progress s in [0, 1]."""
    # Min-jerk ramps over the first and last half-stroke: the pumping eases in and out
    # with continuous acceleration (sqrt(sin) jerked at the start and end).
    ramp = 2 * SHAKE_CYCLES
    envelope = smooth(s * ramp) * smooth((1.0 - s) * ramp)
    return SHAKE_AMPLITUDE * envelope * np.sin(2 * np.pi * SHAKE_CYCLES * s)


def nlerp(qa, qb, w):
    """Normalised quaternion interpolation (fine for the small angles used here)."""
    qb = qb if np.dot(qa, qb) >= 0 else -qb
    q = (1 - w) * qa + w * qb
    return q / np.linalg.norm(q)


def smooth(s):
    """Min-jerk 0..1 profile."""
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# Direction the partner sat from the robot hand during the tuned approach (world). The robot
# now does the moving, so it comes in along the reverse of this.
APPROACH_WORLD = SHAKE_ROT @ _unit(PARTNER_APPROACH_DIR)


def engage_phase(t):
    """(name, progress) within the ENGAGE timeline at time t since the robot arrived."""
    for name, dur in ENGAGE:
        if t < dur:
            return name, t / dur
        t -= dur
    return None, 1.0


def engage_start(name):
    t = 0.0
    for n, dur in ENGAGE:
        if n == name:
            return t
        t += dur
    raise KeyError(name)


ENGAGE_DUR = dict(ENGAGE)


class Robot:
    """Arm + hand controller.

    home -> (hand offered, robot enabled) -> reach (joint space, to 15 cm short of the hand)
    -> approach (straight line into the hand) -> partner_grip .. partner_release (the tuned
    handshake, centred on where the hand was detected) -> retreat (straight line back)
    -> return (joint space to home) -> home.
    Abort: while reaching or approaching, if the visible hand moves more than ABORT_DIST
    from where the robot committed, or is lost for LOST_ABORT s.
    """

    def __init__(self, m):
        self.m = m
        self.d_ik = mujoco.MjData(m)
        self.q_shake = solve_ik_global(m, self.d_ik, SHAKE_MOUNT_POS, SHAKE_QUAT, SHAKE_Q_REF)
        self.eq_target = m.equality("partner_to_target").id
        self.eq_robot = m.equality("partner_to_robot").id
        self.press = m.actuator(PRESS_ACT).id
        self.partner_grip = [i for i in range(m.nu)
                             if m.actuator(i).name.startswith("act_partner_") and i != self.press]
        self.enabled = False
        self.stopping = False
        self.state, self.t = "home", 0.0
        self.q_cmd = ARM_HOME.copy()
        self.q_from = ARM_HOME.copy()
        self.target = self.committed = self.pre = None
        self.q_pre = None
        self.et, self.jump = 0.0, None        # time in the engaged timeline; stop-time jump
        self.human_from = None                # partner pose when the robot arrived
        self.events = []                      # (time, kind, info) for --check
        self.handshakes = 0
        self.arrived_home = False
        self.dz, self.now, self.lost_since, self.retry_at = 0.0, 0.0, None, 0.0
        self.q_ref = self.q_shake.copy()      # IK posture reference (set per meeting)
        self.partner_on, self.contact = False, False

    # --- enable / disable (Start / Stop) ---
    def enable(self):
        self.enabled, self.stopping = True, False

    def disable(self):
        """Stop responding; finish gracefully from wherever the robot is."""
        if not self.enabled:
            return
        self.enabled = False
        if self.state in ("reach", "approach"):
            self._abort("stopped")
        elif self.state in ENGAGE_DUR:
            self.stopping = True
            name, s = engage_phase(self.et)
            mirror = {"partner_grip": "partner_release", "grip": "release"}
            if name in mirror:      # mirrored phase at 1-s has the same pose (min-jerk)
                self.et = engage_start(mirror[name]) + (1 - s) * ENGAGE_DUR[mirror[name]]
            elif name == "shake":   # finish the current half-stroke, then let go
                k = np.ceil(s * 2 * SHAKE_CYCLES - 1e-9) / (2 * SHAKE_CYCLES)
                self.jump = (engage_start("shake") + k * ENGAGE_DUR["shake"], engage_start("release"))
            elif name == "hold":
                self.et = engage_start("release")

    # --- helpers ---
    def _go(self, state):
        self.state, self.t = state, 0.0

    def _ik(self, pos, q_init, iters=20, q_ref=None):
        """IK with a null-space pull toward q_ref: the posture planned for this meeting
        (q_meet) once the robot has committed. Pulling toward the nominal posture instead
        swings the elbow mid-handshake for off-nominal offers, and the hand lags into the
        person's hand."""
        q_ref = self.q_ref if q_ref is None else q_ref
        return solve_ik(self.m, self.d_ik, q_init, pos, SHAKE_QUAT, q_ref, iters)

    def _plan(self, target):
        """Pre-grasp and meeting configurations for a hand-frame target; None if unreachable."""
        pre = target - APPROACH_WORLD * PRE_DIST
        q_pre = self._ik(pre, self.q_shake, 200, self.q_shake)
        q_meet = self._ik(target, q_pre, 200, q_pre)
        for q, p in ((q_pre, pre), (q_meet, target)):
            ep, er = pose_error(self.m, self.d_ik, q, p, SHAKE_QUAT)
            if ep > 2e-3 or er > 2e-2:
                return None
        return pre, q_pre, q_meet

    def _commit(self, t_sim, target, q_now):
        if t_sim < self.retry_at:
            return
        plan = self._plan(target)
        if plan is None:            # out of reach: don't respond; check again shortly
            self.events.append((t_sim, "unreachable", target.copy()))
            self.retry_at = t_sim + 0.5
            return
        self.pre, self.q_pre, self.q_ref = plan
        self.target, self.committed = target.copy(), target.copy()
        self.q_from = q_now.copy()
        self.lost_since = None
        self.events.append((t_sim, "commit", target.copy()))
        self._go("reach")

    def _abort(self, why):
        self.events.append((self.now, "abort", why))
        if self.state == "approach":
            self.retreat_from = self._cart_pos()
            self._go("retreat")
        else:
            self.q_from = self.q_cmd.copy()
            self._go("return")

    def _cart_pos(self):
        s = smooth(self.t / T_APPROACH)
        return self.pre + s * (self.target - self.pre)

    def _watch_hand(self, det, t_sim):
        """Abort if the hand left. Small, early corrections refine the target."""
        est = det.estimate
        if est is None:
            self.lost_since = self.lost_since if self.lost_since is not None else t_sim
            if t_sim - self.lost_since > LOST_ABORT:
                self._abort("hand lost")
            return
        self.lost_since = None
        clear = det.last.visible and not det.last.occluded
        if clear and np.linalg.norm(est - self.committed) > ABORT_DIST:
            self._abort("hand moved away")
        elif self.state == "reach" and self.t < 0.6 * T_REACH and clear and \
                np.linalg.norm(est - self.target) > 0.005 and np.linalg.norm(est - self.committed) < 0.03:
            self.target = est.copy()
            self.pre = self.target - APPROACH_WORLD * PRE_DIST
            plan = self._plan(self.target)
            if plan is not None:
                self.pre, self.q_pre, self.q_ref = plan

    # --- partner coupling while engaged ---
    @property
    def drives_partner(self):
        """The person has taken the robot's hand (simulation coupling: the person's hand,
        fingers and press follow the handshake script while engaged)."""
        return self.state in ENGAGE_DUR and self.partner_on

    def partner_pose(self, dz=0.0):
        """Mocap target for the partner hand while engaged: it adapts to the robot's hand
        during the first ADAPT_TIME of partner_grip, then simply follows the bracket pose."""
        goal = self.target + np.array([0, 0, dz])
        w = smooth(self.et / ADAPT_TIME)
        if w >= 1.0 or self.human_from is None:
            return goal, SHAKE_QUAT
        p0, q0 = self.human_from
        q0 = q0 if np.dot(q0, SHAKE_QUAT) >= 0 else -q0
        q = (1 - w) * q0 + w * SHAKE_QUAT
        return p0 + w * (goal - p0), q / np.linalg.norm(q)

    def welds(self):
        """(partner_to_target active, partner_to_robot active)."""
        if not self.drives_partner:
            return True, False
        name, _ = engage_phase(self.et)
        to_robot = not (name == "partner_grip" and self.et < ADAPT_TIME)
        return name not in ("shake", "hold"), to_robot

    # --- control tick ---
    def update(self, d, det, human, dt, palm_total=0.0):
        self.t += dt
        self.now = d.time
        self.arrived_home = False
        hand, level, dz = HAND_OPEN, 0.0, 0.0
        q = self.q_cmd
        if self.state == "home":
            q = ARM_HOME
            if self.enabled and self.t > COOLDOWN and det.offered:
                self._commit(d.time, det.estimate, d.qpos[:7])
        if self.state == "reach":
            self._watch_hand(det, d.time)
        if self.state == "reach":
            q = self.q_from + smooth(self.t / T_REACH) * (self.q_pre - self.q_from)
            if self.t >= T_REACH:
                self._go("approach")
        if self.state == "approach":
            self._watch_hand(det, d.time)
        if self.state == "approach":
            q = self._ik(self._cart_pos(), self.q_cmd)
            if self.t >= T_APPROACH:
                self.human_from = (human.pos.copy(), human.quat.copy())
                self.partner_on = human.engage(d.time, self.target)
                self.contact = False
                self.events.append((d.time, "arrive", None))
                self.et, self.jump = 0.0, None
                self._go("partner_grip")
        if self.state in ENGAGE_DUR:
            if self.t > 0:      # advance the engaged timeline
                self.et += dt
                if self.jump and self.et >= self.jump[0]:
                    self.et, self.jump = self.jump[1], None
            name, s = engage_phase(self.et)
            if not self.contact and palm_total > CONTACT_N:
                self.contact = True
                self.events.append((d.time, "engage", None))
            if not self.contact and name == "partner_grip" and self.et > CONTACT_WAIT:
                # The robot felt nothing on its palm: nobody took its hand. Back off.
                self.events.append((d.time, "abort", "no contact"))
                if self.partner_on:
                    human.release(d.time, self.target, SHAKE_QUAT)
                self.partner_on = False
                self.retreat_from = self.target.copy()
                self._go("retreat")
                name = "aborted"
            if name == "aborted":
                pass
            elif name is None:  # handshake done: let go of the person, back off
                if self.partner_on:
                    human.release(d.time, self.target, SHAKE_QUAT)
                self.partner_on = False
                if not self.stopping:
                    self.handshakes += 1
                self.events.append((d.time, "done", None))
                self.retreat_from = self.target.copy()
                self._go("retreat")
            else:
                self.state = name
                if name == "grip":
                    hand = HAND_OPEN + smooth(s) * (HAND_GRIP - HAND_OPEN)
                elif name in ("shake", "hold"):
                    hand = HAND_GRIP
                    if name == "shake":
                        dz = shake_offset(s)
                elif name == "release":
                    hand = HAND_GRIP + smooth(s) * (HAND_OPEN - HAND_GRIP)
                level = 1.0 if name in GRIPPED else {"partner_grip": smooth(s),
                                                     "partner_release": 1.0 - smooth(s)}.get(name, 0.0)
                q = self.q_cmd
                if dz or self.et < 2 * dt:
                    q = self._ik(self.target + np.array([0, 0, dz]), self.q_cmd)
                self.dz = dz
        if self.state == "retreat":
            s = smooth(self.t / T_RETREAT)
            q = self._ik(self.retreat_from + s * (self.pre - self.retreat_from), self.q_cmd)
            if self.t >= T_RETREAT:
                self.q_from = q.copy()
                self._go("return")
        if self.state == "return":
            q = self.q_from + smooth(self.t / T_RETURN) * (ARM_HOME - self.q_from)
            if self.t >= T_RETURN:
                self._go("home")
                self.stopping = False
                self.arrived_home = True
                q = ARM_HOME
        self.q_cmd = np.array(q, float)
        d.ctrl[:7] = self.q_cmd
        d.ctrl[7:14] = hand
        if self.drives_partner or self.state not in ENGAGE_DUR:
            lv = level if self.drives_partner else 0.0
            d.ctrl[self.partner_grip] = partner_grip_targets(lv)
            d.ctrl[self.press] = lv * PRESS_TARGET   # the person presses their palm in
        if not self.drives_partner:
            self.dz = 0.0


def reset(m, d):
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    d.ctrl[7:14] = HAND_OPEN
    mujoco.mj_forward(m, d)


CTRL_EVERY = 10     # physics steps per control/sensor update (20 ms -> 50 Hz)
DETECT_EVERY = 5    # control ticks per camera frame (10 Hz)


class Session:
    """Model + robot controller + scripted person + head-camera hand detector + tactile
    skin, stepped together.

    The person behaves on their own (human_agent.py). The robot watches with its head
    camera and, while enabled, reaches out when it sees a hand offered.
      autostart=True  (record/check): robot enabled from the start.
      autostart=False (viewer): robot waits at home, not responding, until start().
    stop() disables it; an ongoing handshake winds down gracefully first.
    """

    def __init__(self, autostart=True, script=None, seed=0):
        self.m = m = build_model()
        self.d = d = mujoco.MjData(m)
        self.robot = Robot(m)
        self.skin = TactileSensors(m)
        self.body = HumanBody(m)
        self.human = HumanAgent(m, SHAKE_MOUNT_POS, SHAKE_QUAT, script=script, seed=seed)
        self.detector = HandDetector(m, SHAKE_MOUNT_POS, seed=seed)
        self.ctrl_dt = CTRL_EVERY * m.opt.timestep
        reset(m, d)
        self.free = m.jnt_qposadr[m.joint("partner_free").id]
        # Calibrate the detector with the hand at the nominal offer pose (robot at home).
        d.qpos[self.free:self.free + 3] = SHAKE_MOUNT_POS
        d.qpos[self.free + 3:self.free + 7] = SHAKE_QUAT
        mujoco.mj_forward(m, d)
        self.body.update(d)
        mujoco.mj_forward(m, d)
        self.calib_pixels = self.detector.calibrate(d, SHAKE_MOUNT_POS)
        # Start with the hand resting on the thigh.
        d.qpos[self.free:self.free + 3] = self.human.rest_pos
        d.qpos[self.free + 3:self.free + 7] = self.human.rest_quat
        d.mocap_pos[0], d.mocap_quat[0] = self.human.rest_pos, self.human.rest_quat
        mujoco.mj_forward(m, d)
        self.body.update(d)
        mujoco.mj_forward(m, d)
        self.i = 0
        if autostart:
            self.robot.enable()
        self.frame = self.skin.read(d, 0.0)

    # --- start / stop (the robot's responsiveness) ---
    @property
    def running(self):
        return self.robot.enabled

    @property
    def stopping(self):
        return self.robot.stopping

    @property
    def phase(self):
        return self.robot.state

    @property
    def status(self):
        r = self.robot
        robot = r.state + (" (stopping)" if r.stopping else "") + ("" if r.enabled or r.stopping else " [off]")
        return f"robot: {robot}   hand: {self.detector.state}   person: {self.human.activity}"

    def start(self):
        self.robot.enable()

    def stop(self):
        self.robot.disable()

    def toggle(self):
        if self.robot.enabled:
            self.stop()
        else:
            self.start()

    def step(self):
        m, d, r = self.m, self.d, self.robot
        if self.i % CTRL_EVERY == 0:
            tick = self.i // CTRL_EVERY
            if tick % DETECT_EVERY == 0:
                self.detector.update(d, DETECT_EVERY * self.ctrl_dt)
            r.update(d, self.detector, self.human, self.ctrl_dt, self.frame.palm_total)
            if r.arrived_home:
                self.skin.rezero()            # unloaded, between handshakes
            self.frame = self.skin.read(d, self.ctrl_dt)
        if r.drives_partner:
            pos, quat = r.partner_pose(r.dz)
        else:
            pos, quat = self.human.update(d.time, r.state in ("reach", "approach"))
        d.mocap_pos[0], d.mocap_quat[0] = pos, quat
        d.eq_active[r.eq_target], d.eq_active[r.eq_robot] = r.welds()
        mujoco.mj_step(m, d)
        self.body.update(d)                   # keep the upper arm on the partner's elbow
        self.i += 1
        return self.i % CTRL_EVERY == 0       # True right after a sensor update

    def close(self):
        self.detector.close()


# --- head-camera display ---------------------------------------------------------------

STATE_RGB = {"none": (150, 150, 150), "tracking": (240, 200, 40), "offered": (60, 210, 90)}


def _resize(img, w, h):
    return img[np.arange(h) * img.shape[0] // h][:, np.arange(w) * img.shape[1] // w]


def _box(img, v0, v1, u0, u1, rgb, t=2):
    h, w = img.shape[:2]
    v0, v1, u0, u1 = max(0, v0), min(h - 1, v1), max(0, u0), min(w - 1, u1)
    img[v0:v0 + t, u0:u1 + 1] = rgb
    img[max(v1 - t + 1, 0):v1 + 1, u0:u1 + 1] = rgb
    img[v0:v1 + 1, u0:u0 + t] = rgb
    img[v0:v1 + 1, max(u1 - t + 1, 0):u1 + 1] = rgb


def head_view(ses, rgb, w=640, h=360):
    """Head camera for display: RGB (with the hand estimate) over colour-mapped depth (with
    the detection box). Colour = detector state: grey none, yellow tracking, green offered,
    blue once the robot has committed."""
    from head_camera import D435, depth_rgb_u8

    det, r = ses.detector, ses.robot
    col = STATE_RGB[det.state]
    if r.state not in ("home",) and r.target is not None and r.state != "return":
        col = (70, 140, 255)
    top = _resize(rgb, w, h).copy()
    est = det.estimate
    if est is not None:
        px = det.project_rgb(ses.d, est)
        if px is not None:
            u = int(px[0] * w / D435["rgb_size"][1])
            v = int(px[1] * h / D435["rgb_size"][0])
            _box(top, v - 10, v + 10, u - 10, u + 10, col, 2)
    if det.last.depth is not None:
        bot = _resize(depth_rgb_u8(det.last.depth), w, h).copy()
    else:
        bot = np.zeros((h, w, 3), np.uint8)
    if det.last.visible and det.last.bbox is not None:
        dh, dw = D435["depth_size"]
        v0, v1, u0, u1 = det.last.bbox
        _box(bot, v0 * h // dh, v1 * h // dh, u0 * w // dw, u1 * w // dw, col, 2)
    return np.vstack([top, bot])


class ControlWindow:
    """Tk window with Start/Stop, the robot's head-camera view (RGB over depth, with the
    hand detection) and the live tactile readout panel (the same panel as the --record
    --tactile video). Tk runs on the main thread; the MuJoCo passive viewer has its own
    thread, so both windows stay responsive. Keys (this window): Space/Enter = start/stop."""

    CAM_W = 384   # px, width of the head-camera column (RGB over depth, 384 x 216 each)

    def __init__(self, ses, panel=True, headcam=True, height=720, width=560, cam_hz=5.0):
        import tkinter as tk

        try:    # Windows: be DPI-aware before Tk starts (the MuJoCo viewer turns it on later,
                # which otherwise leaves Tk placing windows in mixed scaled/unscaled pixels)
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass
        self.tk, self.ses = tk, ses
        self.readout = None
        self.root = tk.Tk()
        self.root.title("Handshake control" + (" + tactile readout" if panel else ""))
        # Dock at the top-right and stay above the (often maximised) MuJoCo viewer.
        # ("-0+0" = flush with the right edge; computing x from winfo_screenwidth() breaks
        # under Windows display scaling.)
        self.root.geometry("-0+0")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#171a21")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        bar = tk.Frame(self.root, bg="#171a21", padx=10, pady=8)
        bar.pack(fill="x")
        btn = dict(width=9, font=("Segoe UI", 11, "bold"), relief="flat", bd=0,
                   activeforeground="white", fg="white", cursor="hand2")
        self.b_start = tk.Button(bar, text="Start", bg="#2e8b57", activebackground="#3aa86a",
                                 command=ses.start, **btn)
        self.b_stop = tk.Button(bar, text="Stop", bg="#b8433f", activebackground="#d0524d",
                                command=ses.stop, **btn)
        self.b_start.pack(side="left", padx=(0, 8))
        self.b_stop.pack(side="left")
        self.status = tk.Label(bar, text="", bg="#171a21", fg="#eceef2", anchor="w",
                               font=("Segoe UI", 10), padx=12)
        self.status.pack(side="left", fill="x", expand=True)
        for key in ("<space>", "<Return>"):
            self.root.bind(key, lambda e: ses.toggle())

        self.label = None
        if panel:
            from readout import ReadoutRenderer
            self.readout = ReadoutRenderer(ses.m, height, 0, width)
        self.headcam, self.head_img, self.cam_dt, self.next_cam = headcam, None, 1.0 / cam_hz, 0.0
        self.rgb = None
        if panel or headcam:
            self.label = tk.Label(self.root, bd=0)
            self.label.pack()
        self.photo = None
        self.placed = False
        self.open = True

    def show(self):
        if not self.open:
            return
        ses = self.ses
        self.status.configure(text=ses.status)
        on = ses.running
        self.b_start.configure(state="disabled" if on else "normal")
        self.b_stop.configure(state="normal" if on else "disabled")
        parts = []
        if self.headcam:
            if self.rgb is None or ses.d.time >= self.next_cam:
                self.rgb = ses.detector.camera.render_rgb(ses.d)
                self.next_cam = ses.d.time + self.cam_dt
            cam = head_view(ses, self.rgb, self.CAM_W, self.CAM_W * 9 // 16)
            if self.readout:
                cam = np.vstack([cam, np.full((720 - cam.shape[0], self.CAM_W, 3), (23, 26, 33), np.uint8)])
            parts.append(cam)
        if self.readout:
            parts.append(self.readout.render_panel(ses.frame, ses.phase))
        if parts:
            rgb = np.ascontiguousarray(np.hstack(parts))
            h, w, _ = rgb.shape
            ppm = b"P6 %d %d 255\n" % (w, h) + rgb.tobytes()
            self.photo = self.tk.PhotoImage(data=ppm, format="PPM")
            self.label.configure(image=self.photo)
            if not self.placed:     # re-anchor once the window has its full size
                self.root.update_idletasks()
                self.root.geometry("-0+0")
                self.placed = True

    def pump(self):
        if self.open:
            try:
                self.root.update()
            except self.tk.TclError:
                self.open = False

    def close(self):
        if self.open:
            self.open = False
            self.root.destroy()
        if self.readout:
            self.readout.close()
            self.readout = None


def run_viewer(panel=True, headcam=True, autostart=False, ui_hz=20.0, seed=None):
    import time

    import mujoco.viewer

    ses = Session(autostart=autostart, seed=seed)
    m, d = ses.m, ses.d
    win = ControlWindow(ses, panel, headcam)
    toggles = []   # Enter in the 3D viewer (key callback runs on the viewer thread)

    def on_key(key):
        if key == 257:   # GLFW_KEY_ENTER
            toggles.append(1)

    next_ui = 0.0
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
        v.cam.lookat[:] = [0.55, -0.05, 0.55]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.9, 120, -15
        v.opt.sitegroup[4] = 1   # show the tactile taxels/pads, tinted by force
        print("The person offers their hand now and then. Start: the robot responds to "
              "offered hands; Stop: it stops (Space/Enter in the control window, Enter in the viewer)")
        start = time.perf_counter()
        while v.is_running():
            while toggles:
                toggles.pop()
                ses.toggle()
            if ses.step():
                ses.skin.colorize(ses.frame)
            if ses.i % 8 == 0:
                v.sync()
            if d.time >= next_ui:
                win.show()
                win.pump()   # always: if the sim runs slower than real time, the
                             # sleep branch below never runs and Tk would never draw
                next_ui = d.time + 1.0 / ui_hz
            lag = d.time - (time.perf_counter() - start)
            if lag > 0:
                win.pump()
                time.sleep(lag)
    win.close()
    ses.close()


def write_png(path, rgb):
    """Minimal PNG writer (avoids a Pillow dependency)."""
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


CHECK_SCRIPT = ["offer", "wave", "offer", "hesitate", "offer_withdraw", "offer"]


def run_check(seed=1, max_time=150.0):
    """Headless run of a fixed episode script: detection/trigger metrics per episode, then
    handshake contact and tactile metrics per robot state (as before)."""
    ses = Session(script=CHECK_SCRIPT, seed=seed)
    m, d, r, hu = ses.m, ses.d, ses.robot, ses.human
    pid = m.body("partner").id
    bid = m.body(MOUNT).id
    finger_geoms = {g for g in range(m.ngeom)
                    if m.body(m.geom_bodyid[g]).name.startswith("hand_right_")}
    snaps, stats, tac = {}, {}, {}
    snap_when = ["approach", "grip", "shake", "partner_release", "return"]
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = [0.65, -0.05, 0.55]
    cam.distance, cam.azimuth, cam.elevation = 1.3, 115, -12
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    pads = list(ses.skin.pads)
    os.makedirs(MEDIA, exist_ok=True)
    log = open(os.path.join(MEDIA, "tactile_log.csv"), "w", encoding="utf-8")
    log.write("time,phase,palm_total_N,palm_active,palm_max_N,"
              + ",".join(f"{p}_normal_N,{p}_shear1_N,{p}_shear2_N,{p}_slip" for p in pads) + "\n")
    commits = []          # (time, estimate, true partner position)
    det_err = []          # tracker error while a still, offered hand is visible
    head_snap = None
    n_events = 0
    print(f"detector calibrated on {ses.calib_pixels} hand pixels; script: {', '.join(CHECK_SCRIPT)}")
    while d.time < max_time:
        sensed = ses.step()
        name = r.state
        if len(r.events) > n_events:
            for ev in r.events[n_events:]:
                if ev[1] == "commit":
                    commits.append((ev[0], ev[2], d.xpos[pid].copy()))
                    if len(snaps) < 5:      # frames of the first attempt that completes
                        snaps, snap_when = {}, ["approach", "grip", "shake", "partner_release", "return"]
                    if head_snap is None:
                        head_snap = head_view(ses, ses.detector.camera.render_rgb(d))
            n_events = len(r.events)
        if sensed:
            f = ses.frame
            log.write(f"{f.time:.3f},{name},{f.palm_total:.3f},{f.palm_active},{np.nanmax(f.palm):.3f},"
                      + ",".join(f"{f.pads[p][0]:.3f},{f.pads[p][1]:.3f},{f.pads[p][2]:.3f},{int(f.slip[p])}"
                                 for p in pads) + "\n")
            t = tac.setdefault(name, {"palm": [], "active": [], "pads": {p: [] for p in pads},
                                      "slip": {p: 0 for p in pads}})
            t["palm"].append(f.palm_total)
            t["active"].append(f.palm_active)
            for p in pads:
                t["pads"][p].append(f.pads[p][0])
                t["slip"][p] += f.slip[p]
            est = ses.detector.estimate
            if est is not None and ses.detector.last.visible and not ses.detector.last.occluded                     and hu.activity in ("offer: hold", "offer_withdraw: hold"):
                det_err.append(np.linalg.norm(est - d.xpos[pid]))
        st = stats.setdefault(name, {"pos_err": 0.0, "touch": 0, "n": 0, "max_pen": 0.0})
        st["n"] += 1
        if r.drives_partner:
            st["pos_err"] = max(st["pos_err"], float(np.linalg.norm(d.xpos[bid] - d.xpos[pid])))
        touching = set()
        for c in d.contact[:d.ncon]:
            in1 = m.body_rootid[m.geom_bodyid[c.geom1]] == pid
            in2 = m.body_rootid[m.geom_bodyid[c.geom2]] == pid
            if in1 != in2:
                other = c.geom2 if in1 else c.geom1
                if other in finger_geoms:
                    touching.add(m.body(m.geom_bodyid[other]).name)
                st["max_pen"] = max(st["max_pen"], -c.dist)
        st["touch"] = max(st["touch"], len(touching))
        if snap_when and name == snap_when[0] and (
                r.t > 0.95 * {"approach": T_APPROACH, "return": T_RETURN}.get(name, 1e9) or
                (name in ENGAGE_DUR and engage_phase(r.et)[1] > (0.33 if name == "shake" else 0.95))):
            renderer.update_scene(d, cam, opt)
            snaps[snap_when.pop(0)] = renderer.render().copy()
        if hu.script == [] and hu.episodes and hu.episodes[-1].end is not None and \
                r.state == "home" and hu.activity == "rest":
            break
    log.close()
    print(f"ran {d.time:.1f} s simulated, finite={bool(np.isfinite(d.qpos).all())}, "
          f"handshakes completed: {r.handshakes}")

    # --- episodes: did the robot respond to the right ones, how fast, how accurately ---
    print("\nepisodes (offset = where the person put their hand vs the nominal meeting point)")
    print("  #  kind            offset mm   robot response              latency s   estimate error mm")
    false_triggers = 0
    for k, ep in enumerate(hu.episodes):
        end = ep.end if ep.end is not None else d.time
        cs = [c for c in commits if ep.start <= c[0] <= end]
        evs = [e for e in r.events if e[0] is not None and ep.start <= e[0] <= end]
        aborted = any(e[1] == "abort" for e in evs)
        engaged = any(e[1] == "engage" for e in evs)
        if cs and ep.kind in ("wave", "hesitate"):
            false_triggers += 1
        resp = ("handshake" if engaged else "committed, aborted" if (cs and aborted) else
                "committed" if cs else "none")
        lat = f"{cs[0][0] - ep.still_at:9.2f}" if cs and ep.still_at else "        -"
        err = f"{np.linalg.norm(cs[0][1] - cs[0][2]) * 1000:10.1f}" if cs else "         -"
        off = np.linalg.norm(ep.target - SHAKE_MOUNT_POS) * 1000
        print(f"  {k + 1}  {ep.kind:14s} {off:9.0f}   {resp:26s} {lat}   {err}")
    print(f"  false triggers on wave/hesitate: {false_triggers}")
    if det_err:
        e = np.array(det_err) * 1000
        print(f"  tracker error on a held hand: median {np.median(e):.1f} mm, 95th pct "
              f"{np.percentile(e, 95):.1f} mm, max {e.max():.1f} mm")

    # --- handshake contact + tactile, per robot state (all handshakes pooled) ---
    order = ["home", "reach", "approach"] + [n for n, _ in ENGAGE] + ["retreat", "return"]
    print("\nrobot state   max bracket-vs-partner err (mm)  max links touching partner  max penetration (mm)")
    for name in order:
        st = stats.get(name)
        if st:
            print(f"  {name:15s} {st['pos_err'] * 1000:8.1f} {st['touch']:24d} {st['max_pen'] * 1000:20.1f}")
    print("\ntactile (median / max during each state)")
    print("  state          palm total N      palm taxels   pads loaded (>0.2 N, median Fn)")
    for name in order:
        t = tac.get(name)
        if not t:
            continue
        loaded = {p: np.median([v for v in vals if v > 0.2])
                  for p, vals in t["pads"].items() if np.mean(np.array(vals) > 0.2) > 0.1}
        slips = sum(t["slip"].values())
        print(f"  {name:15s} {np.median(t['palm']):5.1f} / {np.max(t['palm']):5.1f}   "
              f"{np.median(t['active']):3.0f} / {np.max(t['active']):3d}      "
              + (", ".join(f"{p} {v:.1f}" for p, v in loaded.items()) or "-")
              + (f"   [slip samples: {slips}]" if slips else ""))
    print("saved", os.path.join(MEDIA, "tactile_log.csv"))
    if len(snaps) == 5:
        grid = np.vstack([np.hstack([snaps[k] for k in ("approach", "grip", "shake")]),
                          np.hstack([snaps[k] for k in ("partner_release", "return")]
                                    + [np.zeros_like(snaps["grip"])])])
        out = os.path.join(MEDIA, "handshake_frames.png")
        write_png(out, np.ascontiguousarray(grid))
        print("saved", out)
    if head_snap is not None:
        out = os.path.join(MEDIA, "headcam_detect.png")
        write_png(out, np.ascontiguousarray(head_snap))
        print("saved", out, "(head camera at the first commit)")
    ses.close()


def run_record(path, cycles=1, fps=30, width=1280, height=720, tactile=False, headcam=False,
               seed=0):
    """Render to an MP4 (needs: pip install imageio[ffmpeg]) until the robot has completed
    `cycles` handshakes (the person behaves randomly, so this takes a variable time).
    headcam: scene view (left) beside the robot's head camera, RGB over depth (right)."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("Recording needs imageio + ffmpeg:  pip install \"imageio[ffmpeg]\"")

    from readout import ReadoutRenderer

    ses = Session(script=["offer"] * cycles if not headcam else None, seed=seed)
    m, d, r = ses.m, ses.d, ses.robot
    w3d = width - 560 if tactile else width - 640 if headcam else width
    if headcam:
        renderer = mujoco.Renderer(m, height, w3d)
    elif tactile:
        renderer = ReadoutRenderer(m, height, w3d, 560)
    else:
        renderer = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    if tactile:   # closer on the hands so the tinted taxels/pads are visible
        cam.lookat[:] = SHAKE_PALM_WORLD + np.array([-0.05, -0.02, -0.02])
        cam.distance, cam.azimuth, cam.elevation = 0.48, 100, -25
    elif headcam:   # side view showing both the robot and the person
        cam.lookat[:] = [0.6, -0.05, 0.6]
        cam.distance, cam.azimuth, cam.elevation = 1.9, 65, -15
    else:
        cam.lookat[:] = [0.6, -0.05, 0.55]
        cam.distance, cam.azimuth, cam.elevation = 1.6, 120, -12
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0   # hide collision boxes, show only the visual meshes
    opt.sitegroup[4] = int(tactile)

    frame_dt = 1.0 / fps
    next_frame = 0.0
    max_time = 40.0 * cycles
    done_at = None
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=8) as writer:
        while d.time < max_time:
            if ses.step() and tactile:
                ses.skin.colorize(ses.frame)
            if d.time >= next_frame:
                if headcam:
                    renderer.update_scene(d, cam, opt)
                    writer.append_data(np.hstack([
                        renderer.render(),
                        head_view(ses, ses.detector.camera.render_rgb(d), 640, height // 2)]))
                elif tactile:
                    writer.append_data(renderer.render(d, cam, opt, ses.frame, ses.phase))
                else:
                    renderer.update_scene(d, cam, opt)
                    writer.append_data(renderer.render())
                next_frame += frame_dt
                n = int(round(next_frame * fps))
                if n % fps == 0:
                    print(f"\r  {next_frame:5.1f} s simulated, handshakes {r.handshakes}/{cycles}",
                          end="", flush=True)
            if r.handshakes >= cycles and done_at is None:
                done_at = d.time
            if done_at is not None and r.state == "home" and d.time > done_at + T_RETURN + 0.5:
                break
    ses.close()
    print(f"\nsaved {path}  ({d.time:.1f} s, {r.handshakes} handshakes, @ {fps} fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="headless run with metrics + frames")
    ap.add_argument("--record", nargs="?", const=os.path.join(MEDIA, "handshake.mp4"), metavar="FILE.mp4",
                    help="render an MP4 instead of opening the viewer (default: media/handshake.mp4)")
    ap.add_argument("--cycles", type=int, default=1, help="handshakes to record")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for the person's behaviour (viewer: random by default)")
    ap.add_argument("--no-panel", action="store_true",
                    help="viewer: control window without the tactile readout panel")
    ap.add_argument("--autostart", action="store_true",
                    help="viewer: robot responds to offered hands right away instead of waiting for Start")
    ap.add_argument("--tactile", action="store_true",
                    help="with --record: close-up + live tactile readout panel (1600x720)")
    ap.add_argument("--headcam", action="store_true",
                    help="with --record: scene view + the robot's head camera, RGB over depth")
    ap.add_argument("--no-headcam", action="store_true",
                    help="viewer: control window without the head-camera view")
    args = ap.parse_args()
    if args.record:
        seed = args.seed if args.seed is not None else 0
        if args.headcam:
            run_record(args.record, args.cycles, headcam=True, seed=seed)
        elif args.tactile:
            run_record(args.record, args.cycles, width=1600, tactile=True, seed=seed)
        else:
            run_record(args.record, args.cycles, seed=seed)
    elif args.check:
        run_check(seed=args.seed if args.seed is not None else 1)
    else:
        run_viewer(panel=not args.no_panel, headcam=not args.no_headcam, autostart=args.autostart,
                   seed=args.seed)
