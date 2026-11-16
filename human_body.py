"""Simple seated person around the partner hand, so the head camera sees a person and not
a floating arm.

The scene is smaller than a standing adult (the hands meet at 0.55 m), which matches a
person sitting on a stool. They face the robot (-x), so their right side is +y, and the
existing partner forearm + hand (partner_hand.py) is their right forearm.

Everything here is visual only (no contacts, no mass), so the tuned handshake physics is
unchanged. The body is static; the right upper arm is a mocap body re-posed every step to
span shoulder -> partner elbow (its length is updated to match, so it stretches while the
scripted hand approaches; a jointed arm comes with the scripted human behaviour).
"""

import mujoco
import numpy as np

SHIRT = [0.25, 0.35, 0.55, 1]      # same as the partner sleeve
PANTS = [0.22, 0.23, 0.27, 1]
SKIN = [0.87, 0.70, 0.58, 1]
HAIR = [0.20, 0.15, 0.12, 1]
STOOL = [0.30, 0.24, 0.20, 1]

# Layout (m), in the body frame; the whole body sits BODY_X further from the robot. Right
# shoulder sits ~0.29 m above/behind the partner's elbow at the handshake.
BODY_X = 0.06                       # follows the meeting point (handshake.SHAKE_PALM_WORLD)
CENTER_Y = -0.07                    # body centre line
R_SHOULDER_LOCAL = np.array([1.06, 0.12, 0.88])
L_SHOULDER = np.array([1.06, -0.26, 0.88])
R_SHOULDER = R_SHOULDER_LOCAL + [BODY_X, 0.0, 0.0]   # world
HIP_Z, SEAT_Z = 0.47, 0.42
TORSO_X = 1.09
UPPER_ARM_R = 0.042

VIS = dict(contype=0, conaffinity=0, group=1, density=0)


def _capsule(body, name, a, b, r, rgba):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=np.concatenate([a, b]), size=[r, 0, 0], rgba=rgba, **VIS)


def add_human_body(spec: mujoco.MjSpec) -> None:
    w = spec.worldbody
    body = w.add_body(name="human_body", pos=[BODY_X, 0, 0])   # static: fixed to the world
    cy = CENTER_Y
    # Stool and seat.
    body.add_geom(name="human_stool", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=[TORSO_X + 0.02, cy, SEAT_Z / 2], size=[0.17, SEAT_Z / 2, 0], rgba=STOOL, **VIS)
    # Pelvis and torso (slightly leaning forward), shoulders, neck, head.
    body.add_geom(name="human_pelvis", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  pos=[TORSO_X + 0.01, cy, HIP_Z + 0.03], size=[0.11, 0.17, 0.08], rgba=PANTS, **VIS)
    body.add_geom(name="human_torso", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  pos=[TORSO_X - 0.01, cy, 0.68], size=[0.10, 0.17, 0.22], rgba=SHIRT, **VIS)
    _capsule(body, "human_shoulders", L_SHOULDER, R_SHOULDER_LOCAL, 0.055, SHIRT)
    neck = np.array([TORSO_X - 0.025, cy, 0.89])
    _capsule(body, "human_neck", neck, neck + [-0.005, 0, 0.07], 0.045, SKIN)
    head = np.array([TORSO_X - 0.04, cy, 1.05])
    body.add_geom(name="human_head", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, pos=head,
                  size=[0.095, 0.08, 0.115], rgba=SKIN, **VIS)
    body.add_geom(name="human_hair", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, pos=head + [0.012, 0, 0.03],
                  size=[0.093, 0.083, 0.098], rgba=HAIR, **VIS)
    # Legs: thighs forward to slightly spread knees, shins down, feet.
    for side, s in (("r", 1), ("l", -1)):
        hip = np.array([TORSO_X, cy + s * 0.09, HIP_Z])
        knee = np.array([0.80, cy + s * 0.21, HIP_Z + 0.01])
        ankle = np.array([0.78, cy + s * 0.22, 0.07])
        _capsule(body, f"human_thigh_{side}", hip, knee, 0.068, PANTS)
        _capsule(body, f"human_shin_{side}", knee, ankle, 0.05, PANTS)
        body.add_geom(name=f"human_foot_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=ankle + [-0.06, 0, -0.04], size=[0.11, 0.045, 0.03], rgba=PANTS, **VIS)
    # Left arm resting on the lap.
    l_elbow = np.array([1.04, -0.31, 0.63])
    l_wrist = np.array([0.86, -0.27, 0.56])
    _capsule(body, "human_l_upper_arm", L_SHOULDER, l_elbow, UPPER_ARM_R, SHIRT)
    _capsule(body, "human_l_forearm", l_elbow, l_wrist, 0.032, SHIRT)
    body.add_geom(name="human_l_hand", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  pos=l_wrist + [-0.07, 0.01, -0.01], size=[0.075, 0.045, 0.016], rgba=SKIN, **VIS)
    # Right upper arm: mocap body, posed by HumanBody.update().
    arm = w.add_body(name="human_r_upper_arm", mocap=True)
    arm.add_geom(name="human_r_upper_arm", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                 size=[UPPER_ARM_R, 0.14, 0], rgba=SHIRT, **VIS)


class HumanBody:
    """Keeps the right upper arm attached between the shoulder and the partner's elbow."""

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.mocap = m.body_mocapid[m.body("human_r_upper_arm").id]
        self.geom = m.geom("human_r_upper_arm").id
        self.forearm = m.geom("partner_forearm").id
        self.hand = m.body("partner_hand").id

    def elbow(self, d):
        """Far end of the partner forearm capsule (the end away from the hand)."""
        g = self.forearm
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
        half = self.m.geom_size[g][1]
        a, b = d.geom_xpos[g] + axis * half, d.geom_xpos[g] - axis * half
        hand = d.xpos[self.hand]
        return a if np.linalg.norm(a - hand) > np.linalg.norm(b - hand) else b

    def update(self, d):
        e = self.elbow(d)
        v = e - R_SHOULDER
        length = float(np.linalg.norm(v))
        z = v / length
        x = np.cross([0.0, 0.0, 1.0], z)
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
        y = np.cross(z, x)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
        d.mocap_pos[self.mocap] = R_SHOULDER + v / 2
        d.mocap_quat[self.mocap] = q
        self.m.geom_size[self.geom][1] = length / 2
        self.m.geom_rbound[self.geom] = length / 2 + UPPER_ARM_R
