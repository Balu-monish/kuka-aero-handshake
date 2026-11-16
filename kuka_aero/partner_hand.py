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
# Placement from a scored search (palm seating, finger wrap, thumbs on each other's backs,
# then checked through the shake). The palm lies on the robot's distal palm with its pinky
# edge in the hook of the robot fingers; the hands cross at 74 deg.
PALM_CENTER = np.array([0.116, 0.024, -0.0005])
PALM_ANGLE = np.radians(74)        # palm crosses the robot palm; fingers wrap its pinky edge
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
THUMB_LIFT = 0.011                         # m, thumb base raised off the palm plane
# Across the robot's thumb web toward its index knuckle, angled down over its top edge.
# Stable window: about -54..-58 deg in the bracket xy plane (-56 here). Below -50 the thumb
# jams into the robot hand; past -59 it jams against the robot thumb.
THUMB_DIR = np.array([np.cos(np.radians(-56)), np.sin(np.radians(-56)), -0.29])

# Grip pose (rad) the actuators drive toward; contact with the robot hand stops them.
GRIP_FINGER = (1.25, 1.2, 0.9)   # MCP, PIP, DIP
GRIP_THUMB = (0.08, 1.15, 1.0)   # CMC barely moves; MCP/IP curl the thumb over the robot hand

KP = 0.35         # N*m/rad per joint: a firm-but-polite human grip (~5-8 N at the fingertips)
JOINT_RANGE = (0.0, 1.65)

# Forearm direction from the wrist toward the person's elbow, in the partner root frame
# (bracket frame at the shake pose): unaffected by PALM_TILT, which the wrist absorbs.
FOREARM_DIR = np.array([0.94, -0.34, 0.30])   # back toward the person, clear of the robot fingers

# Whole hand tilted about the bracket y axis (deg, about PALM_CENTER) so the palm lies
# along the robot palm, which slopes ~23 deg down from heel to knuckles.
PALM_TILT = 12.0

# Soft palm tissue: the palm face is a grid of spring-loaded pads (hidden, collision
# group 3) over a recessed rigid core, so the palm conforms to the robot's curved palm and
# spreads the load like flesh instead of touching on one rigid edge.
PAD_GRID = (6, 5)        # rows along the palm (heel -> knuckles), columns across it
PAD_RADIUS = 0.009       # m
PAD_PROTRUDE = 0.004     # m, pad surface beyond the visual palm face at rest
PAD_TRAVEL = 0.006       # m, how far a pad can be pushed in
PAD_K = 400.0            # N/m per pad (~ soft palm tissue: 1 N per 2.5 mm)
PAD_DAMP = 3.0           # N*s/m per pad
PAD_ARMATURE = 0.02      # kg, keeps the pad spring well-resolved at the 2 ms step

# Palm press: the hand rides on a slide along its palm normal. A force-limited servo
# (the person's arm) presses the palm in; contact and the two grips set the rest.
PRESS_KP = 2000.0        # N/m
PRESS_FORCE = 15.0       # N, maximum press
PRESS_TARGET = -0.02     # m along the palm normal when fully pressing (saturates the press)
PRESS_RANGE = (-0.02, 0.02)


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


def add_partner_hand(spec: mujoco.MjSpec, root: mujoco.MjsBody, prefix="partner"):
    """Build the anatomical hand under `root` (the free partner body).

    Structure: root -> {prefix}_hand (palm-normal slide, press actuator) -> palm, pads,
    fingers, thumb, forearm. Returns (grip actuator names in grip_targets order,
    press actuator name).
    """
    # Hand body: tilted about PALM_CENTER, sliding along its own palm normal.
    tilt = np.radians(PALM_TILT)
    Ry = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]])
    q_tilt = np.zeros(4)
    mujoco.mju_mat2Quat(q_tilt, Ry.ravel())
    body = root.add_body(name=f"{prefix}_hand", pos=PALM_CENTER - Ry @ PALM_CENTER,
                         quat=q_tilt, gravcomp=1.0)
    body.add_joint(name=f"{prefix}_press", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                   range=PRESS_RANGE, damping=40.0, armature=0.05)
    press = spec.add_actuator(name=f"act_{prefix}_press", target=f"{prefix}_press",
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
    press.set_to_position(kp=PRESS_KP, kv=0.0)
    press.ctrlrange = PRESS_RANGE
    press.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    press.forcerange = [-PRESS_FORCE, PRESS_FORCE]

    u = np.array([-np.cos(PALM_ANGLE), np.sin(PALM_ANGLE), 0.0])   # wrist -> knuckles
    # Right hand, palm facing -z: thumb side = u x n with n = (0,0,-1).
    t = np.cross(u, [0, 0, -1.0])
    C = PALM_CENTER
    L, W, T = PALM_HALF
    palm_q = _frame_quat(u)
    visual = dict(contype=0, conaffinity=0, group=1)

    # Visual palm: box with a rounded knuckle ridge and heel, plus the thenar bulge.
    body.add_geom(name=f"{prefix}_palm_skin", type=mujoco.mjtGeom.mjGEOM_BOX, pos=C,
                  quat=palm_q, size=[L, W, T], rgba=SKIN, density=1100, **visual)
    k = C + u * L
    body.add_geom(name=f"{prefix}_knuckles", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=np.concatenate([k + t * (W - 0.012), k - t * (W - 0.010)]),
                  size=[T, 0, 0], rgba=SKIN, density=1100, **visual)
    h = C - u * L
    body.add_geom(name=f"{prefix}_heel", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=np.concatenate([h + t * (W - 0.014), h - t * (W - 0.012)]),
                  size=[T, 0, 0], rgba=SKIN, density=1100, **visual)
    body.add_geom(name=f"{prefix}_thenar", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  pos=C + t * 0.022 - u * 0.016, quat=palm_q,
                  size=[0.030, 0.017, 0.012], rgba=SKIN, density=1100, **visual)
    # Rigid core (bones): covers the palm outline incl. heel and knuckles; its palm face
    # is recessed so the pads take the palm contact. The back and edges stay firm.
    recess = PAD_PROTRUDE + PAD_TRAVEL
    body.add_geom(name=f"{prefix}_palm", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=C + np.array([0, 0, recess / 2]), quat=palm_q,
                  size=[L + T * 0.8, W, T + 0.002 - recess / 2], group=3, rgba=[0.9, 0.6, 0.5, 0.4],
                  friction=[1.0, 0.02, 0.001], density=0)
    # Soft tissue pads on the palm face.
    pads = []
    rows, cols = PAD_GRID
    face_z = C[2] - T
    for i in range(rows):
        for j in range(cols):
            a = -L - 0.006 + (2 * L + 0.010) * i / (rows - 1)          # along the palm
            b = -(W - 0.007) + (2 * W - 0.014) * j / (cols - 1)        # across it
            p = C + u * a + t * b
            p[2] = face_z - PAD_PROTRUDE + PAD_RADIUS
            pad = body.add_body(name=f"{prefix}_pad_{i}{j}", pos=p)
            pad.add_joint(name=f"{prefix}_pad_{i}{j}_z", type=mujoco.mjtJoint.mjJNT_SLIDE,
                          axis=[0, 0, 1], range=[0.0, PAD_TRAVEL], stiffness=PAD_K,
                          damping=PAD_DAMP, armature=PAD_ARMATURE)
            pad.add_geom(name=f"{prefix}_pad_{i}{j}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                         size=[PAD_RADIUS, 0, 0], group=3, rgba=[0.9, 0.5, 0.4, 0.5],
                         contype=2, conaffinity=1,   # collide with the robot, not each other
                         friction=[1.0, 0.02, 0.001], density=300)
            pads.append(pad.name)
    # Pads only touch the robot: no contact with the partner's own fingers or thumb.
    own = [f"{prefix}_{f}_{s}" for f in list(FINGERS) + ["thumb"] for s in (1, 2, 3)]
    for p in pads:
        for o in own:
            spec.add_exclude(bodyname1=p, bodyname2=o)

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
    elbow_dir = Ry.T @ _unit(FOREARM_DIR)
    body.add_geom(name=f"{prefix}_wrist", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=np.concatenate([wrist, wrist + elbow_dir * 0.05]),
                  size=[0.022, 0, 0], rgba=SKIN, density=1000)
    body.add_geom(name=f"{prefix}_forearm", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=np.concatenate([wrist + elbow_dir * 0.07, wrist + elbow_dir * 0.32]),
                  size=[0.032, 0, 0], rgba=SLEEVE, density=250)   # light: less inertia in the shake
    return names, press.name


def grip_targets(level):
    """Joint targets for all partner actuators at grip level 0 (open) .. 1 (closed)."""
    per_finger = list(GRIP_FINGER) * len(FINGERS)
    return level * np.array(per_finger + list(GRIP_THUMB))
