"""Scripted, randomized person: rests, then offers a hand (or does something else), and
reacts to the robot.

The person moves the partner hand by its mocap target (partner_to_target weld). Between
episodes the hand rests palm-down on their right thigh. Episodes:
  offer           raise the hand to a random spot near the nominal meeting point and hold
                  it there until the robot arrives (gives up after OFFER_TIMEOUT)
  offer_withdraw  offer, but pull back after WITHDRAW_AFTER s -> the robot must abort
  wave            wave side to side above the meeting zone -> must not trigger
  hesitate        start to offer, stop still about halfway (outside the zone), go back
When the robot's hand arrives (engage), the robot controller drives the hand; when the
robot lets go (release), the person waits a moment and returns to rest.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

OFFER_SPREAD = 0.05       # m, offers land uniformly within +/- this box around nominal
OFFER_TIMEOUT = 5.0       # s, waiting for the robot before giving up
WITHDRAW_AFTER = 0.9      # s, offer_withdraw holds this long, then pulls back
REST_TIME = (1.5, 3.0)    # s, between episodes
MOVE_TIME = (1.3, 1.9)    # s, rest -> offer
SWAY = 0.002              # m, a held hand is never perfectly still
ACCEPT_DIST = 0.04        # m, the person takes the robot's hand only if it arrives this close
EPISODE_WEIGHTS = {"offer": 0.6, "offer_withdraw": 0.1, "wave": 0.15, "hesitate": 0.15}

# Where the hand rests (world): palm on the right thigh near the knee, forearm back toward
# the elbow below the right shoulder (see human_body.py).
REST_PALM = np.array([0.95, 0.105, 0.57])
REST_ELBOW = np.array([1.09, 0.13, 0.61])


def _frame(a, b):
    """Rotation whose columns are a, (b orthogonalised to a), a x b."""
    a = a / np.linalg.norm(a)
    b = b - a * np.dot(a, b)
    b = b / np.linalg.norm(b)
    return np.column_stack([a, b, np.cross(a, b)])


def _mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R).ravel())
    return q


def _quat_interp(qa, qb, w):
    qb = qb if np.dot(qa, qb) >= 0 else -qb
    dot = np.clip(np.dot(qa, qb), -1.0, 1.0)
    th = np.arccos(dot)
    if th < 1e-6:
        return qa.copy()
    return (np.sin((1 - w) * th) * qa + np.sin(w * th) * qb) / np.sin(th)


def _smooth(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


@dataclass
class Episode:
    kind: str
    start: float
    target: np.ndarray                 # hand-frame position the person aims for (truth)
    still_at: float = None             # when the hand stopped at the target / pause spot
    engaged_at: float = None
    end: float = None
    notes: list = field(default_factory=list)


class HumanAgent:
    def __init__(self, m: mujoco.MjModel, offer_pos, offer_quat, script=None, seed=0):
        self.rng = np.random.default_rng(seed)
        self.offer_pos = np.asarray(offer_pos, float)
        self.offer_quat = np.asarray(offer_quat, float)
        self.script = list(script) if script else None
        self.rest_pos, self.rest_quat = self._rest_pose(m)
        self.pos, self.quat = self.rest_pos.copy(), self.rest_quat.copy()
        self.plan = []          # [kind, duration, payload], consumed front to back
        self.t0 = 0.0           # start time of plan[0]
        self.from_pos, self.from_quat = self.pos.copy(), self.quat.copy()
        self.episodes = []
        self.engaged = False
        self._last_t = 0.0
        self._rest(self.rng.uniform(0.8, 1.5))

    def _rest_pose(self, m):
        """Partner-root pose that puts the palm on the thigh, palm down, forearm to the elbow."""
        d = mujoco.MjData(m)
        adr = m.jnt_qposadr[m.joint("partner_free").id]
        d.qpos[adr + 3] = 1.0                      # partner root at the origin, identity
        mujoco.mj_kinematics(m, d)
        palm = d.geom_xpos[m.geom("partner_palm_skin").id].copy()
        wrist = d.geom_xpos[m.geom("partner_wrist").id]
        fore = d.geom_xpos[m.geom("partner_forearm").id]
        normal = d.geom_xmat[m.geom("partner_palm_skin").id].reshape(3, 3)[:, 2] * -1.0
        local = _frame(fore - wrist, normal)       # forearm axis, palm normal (root frame)
        world = _frame(REST_ELBOW - REST_PALM, np.array([0.0, 0.0, -1.0]))
        R = world @ local.T
        return REST_PALM - R @ palm, _mat2quat(R)

    # --- plan building ---
    def _rest(self, dur):
        self.plan.append(["rest", dur, None])

    def _move(self, pos, quat, dur, lift=0.06):
        self.plan.append(["move", dur, (np.asarray(pos, float), np.asarray(quat, float), lift)])

    def _new_episode(self, t):
        if self.script is not None:
            if not self.script:
                self._rest(1e9)
                return
            kind = self.script.pop(0)
        else:
            kinds = list(EPISODE_WEIGHTS)
            kind = self.rng.choice(kinds, p=np.array([EPISODE_WEIGHTS[k] for k in kinds]))
        target = self.offer_pos + self.rng.uniform(-OFFER_SPREAD, OFFER_SPREAD, 3)
        ep = Episode(str(kind), t, target)
        self.episodes.append(ep)
        T = self.rng.uniform(*MOVE_TIME)
        q = self.offer_quat
        if kind in ("offer", "offer_withdraw"):
            self._move(target, q, T)
            self.plan.append(["hold", OFFER_TIMEOUT if kind == "offer" else WITHDRAW_AFTER, ep])
        elif kind == "wave":
            center = target + np.array([0.05, 0.03, 0.22])
            ep.target = center
            self._move(center, q, T, lift=0.03)
            self.plan.append(["wave", 2.0, center])
        else:   # hesitate: stop about halfway, outside the meeting zone, then go back
            mid = self.rest_pos + 0.5 * (target - self.rest_pos)
            ep.target = mid
            self._move(mid, _quat_interp(self.rest_quat, q, 0.5), 0.6 * T, lift=0.03)
            self.plan.append(["hold", 0.8, ep])
        self._move(self.rest_pos, self.rest_quat, self.rng.uniform(1.1, 1.5))
        self._rest(self.rng.uniform(*REST_TIME))

    # --- robot interaction ---
    def engage(self, t, robot_hand_pos):
        """The robot's hand has arrived. The person takes it only if they are still holding
        out their hand and it is where the robot's hand is (within ACCEPT_DIST); then the
        robot drives the hand from now on. Returns whether they took it."""
        holding = bool(self.plan) and self.plan[0][0] == "hold"
        if not holding or np.linalg.norm(self.pos - robot_hand_pos) > ACCEPT_DIST:
            return False
        self.engaged = True
        ep = self.episodes[-1] if self.episodes else None
        if ep and ep.engaged_at is None:
            ep.engaged_at = t
        # Drop the rest of the current episode; after release: back to rest, then pause.
        self.plan = []
        return True

    def release(self, t, pos, quat):
        """The robot has let go: wait a moment, then put the hand back on the thigh."""
        self.engaged = False
        self.pos, self.quat = np.asarray(pos, float).copy(), np.asarray(quat, float).copy()
        self.plan = [["rest", 0.4, None]]
        self._move(self.rest_pos, self.rest_quat, self.rng.uniform(1.2, 1.6))
        self._rest(self.rng.uniform(*REST_TIME))
        self.t0 = t
        self.from_pos, self.from_quat = self.pos.copy(), self.quat.copy()
        if self.episodes and self.episodes[-1].end is None:
            self.episodes[-1].end = t

    # --- per-step target ---
    def update(self, t, robot_coming=False):
        """Mocap target (pos, quat) for the partner hand when the robot isn't driving it.
        robot_coming: the robot is visibly reaching for the hand, so a person offering it
        keeps holding instead of giving up (offer_withdraw still pulls back)."""
        if self.engaged:
            return self.pos, self.quat
        if robot_coming and self.plan and self.plan[0][0] == "hold" and                 self.episodes and self.episodes[-1].kind == "offer":
            self.t0 = min(self.t0 + (t - self._last_t), t)   # pause the hold timer
        self._last_t = t
        while True:
            if not self.plan:
                if self.episodes and self.episodes[-1].end is None:
                    self.episodes[-1].end = t
                self._new_episode(t)
                self.t0 = t
                self.from_pos, self.from_quat = self.pos.copy(), self.quat.copy()
            kind, dur, payload = self.plan[0]
            s = (t - self.t0) / dur
            if s < 1.0:
                break
            # Step finished: land exactly on its end pose and start the next one.
            if kind == "move":
                self.pos, self.quat = payload[0].copy(), payload[1].copy()
                ep = self.episodes[-1] if self.episodes else None
                if ep is not None and ep.still_at is None and len(self.plan) > 1 and \
                        self.plan[1][0] == "hold":
                    ep.still_at = self.t0 + dur
            self.t0 += dur
            self.plan.pop(0)
            self.from_pos, self.from_quat = self.pos.copy(), self.quat.copy()
        if kind == "move":
            pos, quat, lift = payload
            w = _smooth(s)
            self.pos = self.from_pos + w * (pos - self.from_pos) + np.array([0, 0, lift * np.sin(np.pi * w)])
            self.quat = _quat_interp(self.from_quat, quat, w)
        elif kind == "hold":
            ph = 2 * np.pi * (t - self.t0)
            self.pos = self.from_pos + SWAY * np.array([np.sin(0.7 * ph), np.sin(0.5 * ph + 1), np.sin(0.9 * ph + 2)])
        elif kind == "wave":
            ph = 2 * np.pi * 1.2 * (t - self.t0)
            self.pos = payload + np.array([0.0, 0.07 * np.sin(ph), 0.0])
        return self.pos, self.quat

    @property
    def activity(self):
        if self.engaged:
            return "engaged"
        if not self.plan:
            return "rest"
        step = self.plan[0]
        if step[0] == "rest":
            return "rest"
        if step[0] == "move" and np.allclose(step[2][0], self.rest_pos):
            return "returning"
        ep = self.episodes[-1].kind if self.episodes else ""
        return f"{ep}: {step[0]}"
