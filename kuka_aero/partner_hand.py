"""Anatomical partner hand (a person's right hand) for the handshake demo.

Built from capsules/ellipsoids with human proportions: palm, thenar pad, four fingers
(proximal / middle / distal phalanges), a 3-segment thumb, wrist and forearm.
Each finger joint has a soft position actuator so the partner can grip back.

Frame convention (the partner body frame equals the robot bracket frame at the shake
pose): +z points from the robot's palm towards the back of the partner's hand, i.e.
the partner's palm faces -z, toward the robot's palm. Flexing a finger curls it toward -z.
"""

import mujoco
import numpy as np

SKIN = [0.87, 0.70, 0.58, 1]
NAIL = [0.93, 0.82, 0.78, 1]
SLEEVE = [0.25, 0.35, 0.55, 1]

# Palm layout (partner frame).
PALM_CENTER = np.array([0.150, 0.000, -0.006])  # inside the robot fingers' closing arc (searched)
PALM_ANGLE = np.radians(90)        # palm crosses the robot palm; fingers wrap its pinky edge
PALM_HALF = (0.046, 0.041, 0.012)  # half length (wrist->knuckles), half width, half thickness

# name: (offset across palm from centre line [+ = thumb side], knuckle setback, splay deg,
#        phalanx lengths, radius)
FINGERS = {
    "index":  (+0.029, 0.004, +6, (0.040, 0.024, 0.019), 0.0090),
    "middle": (+0.009, 0.000, +1, (0.044, 0.028, 0.020), 0.0092),
    "ring":   (-0.011, 0.004, -4, (0.041, 0.026, 0.019), 0.0087),
    "pinky":  (-0.029, 0.012, -9, (0.032, 0.019, 0.017), 0.0078),
}
THUMB = ((0.040, 0.032, 0.026), 0.0105)   # metacarpal, proximal, distal; radius
THUMB_LIFT = 0.012                         # m, thumb base raised off the palm plane
THUMB_DIR = np.array([-0.4, -0.9, 0.0])   # over the top of the robot hand (None = along its thumb web)

# Grip pose (rad) the actuators drive toward; contact with the robot hand stops them.
GRIP_FINGER = (1.25, 1.2, 0.9)   # MCP, PIP, DIP
GRIP_THUMB = (0.45, 0.9, 0.7)    # curls down over the top of the robot hand

KP = 0.35         # N*m/rad per joint: a firm-but-polite human grip (~5-8 N at the fingertips)
JOINT_RANGE = (0.0, 1.65)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _frame_quat(x_dir, z_hint=(0, 0, 1)):
    """Quaternion of a frame whose x axis is x_dir and z axis is as close to z_hint as possible."""
    x = _unit(x_dir)
    y = _unit(np.cross(z_hint, x))
    z = np.cross(x, y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return q


def _rotz(v, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def _chain(parent, name, pos, direction, lengths, radius, joint_names, spec):
    """A serial finger: one hinge per phalanx, flexing toward the palm (-z)."""
    body = parent.add_body(name=f"{name}_1", pos=pos, quat=_frame_quat(direction))
    for i, length in enumerate(lengths):
        jname = f"{name}_{joint_names[i]}"
        body.add_joint(name=jname, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
                       range=JOINT_RANGE, damping=0.01, armature=0.0008)
        r = radius * (1.0 - 0.08 * i)
        body.add_geom(name=f"{name}_{i + 1}_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0, 0, 0, length, 0, 0], size=[r, 0, 0], rgba=SKIN,
                      friction=[1.0, 0.02, 0.001], density=1100)
        a = spec.add_actuator(name=f"act_{jname}", target=jname,
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=KP, kv=0.02)
        a.ctrlrange = JOINT_RANGE
        if i == len(lengths) - 1:   # fingernail
            body.add_geom(name=f"{name}_nail", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                          pos=[length - 0.004, 0, r * 0.75], size=[0.006, r * 0.75, 0.0015],
                          rgba=NAIL, contype=0, conaffinity=0, density=0)
        else:
            body = body.add_body(name=f"{name}_{i + 2}", pos=[length, 0, 0])


def add_partner_hand(spec: mujoco.MjSpec, body: mujoco.MjsBody, prefix="partner"):
    """Populate `body` with the anatomical hand. Returns actuator names in grip order."""
    u = np.array([-np.cos(PALM_ANGLE), np.sin(PALM_ANGLE), 0.0])   # wrist -> knuckles
    # Right hand, palm facing -z: thumb side = u x n with n = (0,0,-1).
    t = np.cross(u, [0, 0, -1.0])
    C = PALM_CENTER
    L, W, T = PALM_HALF
    palm_q = _frame_quat(u)

    body.add_geom(name=f"{prefix}_palm", type=mujoco.mjtGeom.mjGEOM_BOX, pos=C, quat=palm_q,
                  size=[L, W, T], rgba=SKIN, friction=[1.0, 0.02, 0.001], density=1100)
    # Rounded knuckle ridge and heel so the box reads as a palm.
    k = C + u * L
    body.add_geom(name=f"{prefix}_knuckles", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=np.concatenate([k + t * (W - 0.012), k - t * (W - 0.010)]),
                  size=[T, 0, 0], rgba=SKIN, density=1100)
    h = C - u * L
    body.add_geom(name=f"{prefix}_heel", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=np.concatenate([h + t * (W - 0.014), h - t * (W - 0.012)]),
                  size=[T, 0, 0], rgba=SKIN, density=1100)
    # Thenar pad (thumb muscle) bulging on the palm side.
    body.add_geom(name=f"{prefix}_thenar", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  pos=C + t * 0.022 - u * 0.016, quat=palm_q,
                  size=[0.030, 0.017, 0.012], rgba=SKIN, density=1100)

    names = []
    for fname, (off, setback, splay, lengths, radius) in FINGERS.items():
        knuckle = C + u * (L + 0.004 - setback) + t * off
        # Rotating u about +z turns it toward t, so +splay fans a finger toward the thumb.
        _chain(body, f"{prefix}_{fname}", knuckle, _rotz(u, splay),
               lengths, radius, ("mcp", "pip", "dip"), spec)
        names += [f"act_{prefix}_{fname}_{j}" for j in ("mcp", "pip", "dip")]

    # Thumb: from the thenar side, lying along the robot's index/thumb web toward its wrist.
    tb = C + t * (W - 0.004) - u * 0.020 + np.array([0, 0, THUMB_LIFT])
    tdir = _unit(THUMB_DIR if THUMB_DIR is not None else 0.6 * t + 0.4 * u)
    _chain(body, f"{prefix}_thumb", tb, tdir, THUMB[0], THUMB[1], ("cmc", "mcp", "ip"), spec)
    names += [f"act_{prefix}_thumb_{j}" for j in ("cmc", "mcp", "ip")]

    # Wrist and forearm (sleeve), running back toward the person.
    wrist = h - u * 0.012
    elbow_dir = _unit([1.0, -0.15, 0.0])
    body.add_geom(name=f"{prefix}_wrist", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=np.concatenate([wrist, wrist + elbow_dir * 0.05]),
                  size=[0.022, 0, 0], rgba=SKIN, density=1000)
    body.add_geom(name=f"{prefix}_forearm", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=np.concatenate([wrist + elbow_dir * 0.07, wrist + elbow_dir * 0.32]),
                  size=[0.032, 0, 0], rgba=SLEEVE, density=250)   # light: less inertia in the shake
    return names


def grip_targets(level):
    """Joint targets for all partner actuators at grip level 0 (open) .. 1 (closed)."""
    per_finger = list(GRIP_FINGER) * len(FINGERS)
    return level * np.array(per_finger + list(GRIP_THUMB))
