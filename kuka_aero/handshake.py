"""Handshake demo: the iiwa reaches out, the Aero Hand grips a partner hand, shakes, lets go.

  python handshake.py             # interactive viewer, loops the handshake
  python handshake.py --check     # headless run: prints tracking/grip metrics, saves handshake_frames.png

The partner is an anatomical right hand (see partner_hand.py), softly welded to a mocap
target so it gives a little, with finger actuators so it grips back.
"""

import argparse
import os

import mujoco
import numpy as np

from partner_hand import add_partner_hand
from partner_hand import grip_targets as partner_grip_targets

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(HERE, "scene.xml")

ARM_HOME = np.array([0.0, 0.65, 0.0, -1.3, 0.0, 1.2, 0.0])
MOUNT = "hand_tetheria_mount"

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
SHAKE_PALM_WORLD = np.array([0.62, 0.0, 0.55])       # where the palms meet
SHAKE_MOUNT_POS = SHAKE_PALM_WORLD - SHAKE_ROT @ PALM_CENTER_LOCAL

# Partner hand geometry lives in partner_hand.py (its frame = robot bracket frame at the
# shake pose). It slides in from this far along +x world.
PARTNER_APPROACH = 0.25                              # m

# Hand actuators (tendon lengths): high = open, low = closed.
# Order: index, middle, ring, pinky, thumb abduction, thumb flex 1, thumb flex 2.
HAND_OPEN = np.array([0.11, 0.11, 0.11, 0.11, -0.1, 0.038, 0.112])     # thumb out of the way
HAND_GRIP = np.array([0.059, 0.059, 0.059, 0.059, 0.5, 0.032, 0.095])  # wrap; contact stops it

SHAKE_AMPLITUDE = 0.035   # m, vertical
SHAKE_FREQ = 1.6          # Hz
SHAKE_CYCLES = 3

# Timeline (s).
PHASES = [("reach", 2.5), ("partner_in", 1.5), ("grip", 1.0),
          ("shake", SHAKE_CYCLES / SHAKE_FREQ), ("hold", 0.5), ("release", 0.8),
          ("partner_out", 1.2), ("return", 2.5), ("pause", 1.0)]
CYCLE = sum(t for _, t in PHASES)


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(SCENE_XML)
    # Gravity compensation so the position-controlled arm holds its targets.
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    # Partner hand: a free body (frame = robot bracket frame at the shake pose) held by
    # two soft welds that the script switches between:
    #   partner_to_target: to a mocap target -> scripted approach / retreat
    #   partner_to_robot:  to the robot bracket -> while gripped, the person's relaxed arm
    #                      simply goes along with the shake (no fighting through contacts)
    spec.worldbody.add_body(name="partner_target", mocap=True)
    partner = spec.worldbody.add_body(name="partner", gravcomp=1.0)
    partner.add_freejoint(name="partner_free")
    add_partner_hand(spec, partner)
    # The Aero Hand's contacts are soft (tuned for 10 ms steps and light touches); a
    # firm handshake would sink through them. Give the partner's geoms priority and
    # stiffer contact parameters so every robot<->partner contact uses these.
    for geom in spec.geoms:
        if geom.name.startswith("partner"):
            geom.priority = 1
            geom.solref = [0.005, 1.0]
            geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    data = np.zeros(12)
    data[6], data[10] = 1.0, 1.0   # identity relative pose, torquescale 1
    for name, other in (("partner_to_target", "partner_target"), ("partner_to_robot", MOUNT)):
        weld = spec.add_equality(name=name, type=mujoco.mjtEq.mjEQ_WELD,
                                 objtype=mujoco.mjtObj.mjOBJ_BODY, name1="partner",
                                 name2=other, solref=[0.01, 1.0])
        weld.data = data
        weld.active = name == "partner_to_target"
    # Extend the home keyframe with the partner's free joint (parked out of the way).
    key = spec.key("home")
    start = SHAKE_MOUNT_POS + np.array([PARTNER_APPROACH, 0, 0])
    n_partner = len(partner_grip_targets(0.0))
    key.qpos = list(key.qpos) + list(start) + list(mat2quat(SHAKE_ROT)) + [0.0] * n_partner
    key.ctrl = list(key.ctrl) + [0.0] * n_partner
    key.mpos = list(start)
    key.mquat = list(mat2quat(SHAKE_ROT))
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


def shake_offset(s):
    """Vertical shake displacement at shake-phase progress s in [0, 1]."""
    envelope = np.sin(np.pi * s) ** 0.5   # ease in/out of the pumping
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


class Handshake:
    def __init__(self, m):
        self.m = m
        self.d_ik = mujoco.MjData(m)
        self.q_shake = solve_ik_global(m, self.d_ik, SHAKE_MOUNT_POS, SHAKE_QUAT, ARM_HOME)
        self.q_last = self.q_shake.copy()
        self.eq_target = m.equality("partner_to_target").id
        self.eq_robot = m.equality("partner_to_robot").id

    def phase(self, t):
        t = t % CYCLE
        for name, dur in PHASES:
            if t < dur:
                return name, t / dur
            t -= dur
        return PHASES[-1][0], 1.0

    def targets(self, t):
        """Return (arm joint targets, hand ctrl, partner pos, partner quat, shake offset)."""
        name, s = self.phase(t)
        q, hand, approach, dz = self.q_shake, HAND_OPEN, 0.0, 0.0
        if name == "reach":
            q = ARM_HOME + smooth(s) * (self.q_shake - ARM_HOME)
            approach = 1.0
        elif name == "partner_in":
            approach = 1.0 - smooth(s)
        elif name == "grip":
            hand = HAND_OPEN + smooth(s) * (HAND_GRIP - HAND_OPEN)
        elif name in ("shake", "hold"):
            hand = HAND_GRIP
            if name == "shake":
                dz = shake_offset(s)
        elif name == "release":
            hand = HAND_GRIP + smooth(s) * (HAND_OPEN - HAND_GRIP)
        elif name == "partner_out":
            approach = smooth(s)
        elif name in ("return", "pause"):
            q = self.q_shake + smooth(s if name == "return" else 1.0) * (ARM_HOME - self.q_shake)
            approach = 1.0
        if dz:
            pos = SHAKE_MOUNT_POS + np.array([0, 0, dz])
            q = solve_ik(self.m, self.d_ik, self.q_last, pos, SHAKE_QUAT, self.q_shake, 20)
        self.q_last = q
        partner_pos = SHAKE_MOUNT_POS + np.array([PARTNER_APPROACH * approach, 0, dz])
        return q, hand, partner_pos, SHAKE_QUAT, name

    def apply(self, d, t):
        """Update arm/hand targets (call at the control rate, ~50 Hz)."""
        q, hand, ppos, pquat, name = self.targets(t)
        d.ctrl[:7] = q
        d.ctrl[7:14] = hand
        # Partner grips back on the same schedule as the robot hand.
        _, s = self.phase(t)
        level = {"grip": smooth(s), "shake": 1.0, "hold": 1.0, "release": 1.0 - smooth(s)}.get(name, 0.0)
        d.ctrl[14:] = partner_grip_targets(level)
        return name

    def update_partner(self, d, t):
        """Move the partner (call every physics step).

        Approach/retreat: welded to the scripted mocap target. Grip..release: welded to the
        robot bracket, so the person's arm goes along with the shake. Both welds overlap
        during grip and release for a smooth handover. (Driving the partner on its own
        path while the fingers are interlocked makes them snag and jerk.)
        """
        name, s = self.phase(t)
        approach = {"partner_in": 1.0 - smooth(s), "partner_out": smooth(s)}.get(
            name, 0.0 if name in ("grip", "shake", "hold", "release") else 1.0)
        d.mocap_pos[0] = SHAKE_MOUNT_POS + np.array([PARTNER_APPROACH * approach, 0, 0])
        d.mocap_quat[0] = SHAKE_QUAT
        d.eq_active[self.eq_target] = name not in ("shake", "hold")
        d.eq_active[self.eq_robot] = name in ("grip", "shake", "hold", "release")


def reset(m, d):
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    d.ctrl[7:14] = HAND_OPEN
    mujoco.mj_forward(m, d)


def run_viewer():
    import time

    import mujoco.viewer

    m = build_model()
    d = mujoco.MjData(m)
    hs = Handshake(m)
    reset(m, d)
    ctrl_every = 10  # update targets every 20 ms
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.45, 0.0, 0.45]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.5, 135, -15
        step = 0
        start = time.perf_counter()
        while v.is_running():
            if step % ctrl_every == 0:
                hs.apply(d, d.time)
            hs.update_partner(d, d.time)
            mujoco.mj_step(m, d)
            step += 1
            if step % 8 == 0:
                v.sync()
            lag = d.time - (time.perf_counter() - start)
            if lag > 0:
                time.sleep(lag)


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


def run_check():
    m = build_model()
    d = mujoco.MjData(m)
    hs = Handshake(m)
    reset(m, d)
    pid = m.body("partner").id
    bid = m.body(MOUNT).id
    finger_geoms = {g for g in range(m.ngeom)
                    if m.body(m.geom_bodyid[g]).name.startswith("hand_right_")}
    snaps, stats = {}, {}
    snap_at = {"reach": 0.99, "grip": 0.99, "shake": 0.33, "release": 0.99, "return": 0.99}
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = [0.5, 0.0, 0.5]
    cam.distance, cam.azimuth, cam.elevation = 1.0, 120, -10
    step = 0
    while d.time < CYCLE:
        if step % 10 == 0:
            name = hs.apply(d, d.time)
        hs.update_partner(d, d.time)
        mujoco.mj_step(m, d)
        step += 1
        _, s = hs.phase(d.time)
        st = stats.setdefault(name, {"pos_err": 0.0, "touch": 0, "n": 0, "max_pen": 0.0})
        st["n"] += 1
        if name in ("partner_in", "grip", "shake", "hold"):
            st["pos_err"] = max(st["pos_err"], float(np.linalg.norm(d.xpos[bid] - d.xpos[pid])))
        touching = set()
        for c in d.contact[:d.ncon]:
            pair = {c.geom1, c.geom2}
            if m.geom_bodyid[c.geom1] == pid or m.geom_bodyid[c.geom2] == pid:
                other = (pair - {g for g in pair if m.geom_bodyid[g] == pid})
                touching |= {m.body(m.geom_bodyid[g]).name for g in other & finger_geoms}
                st["max_pen"] = max(st["max_pen"], -c.dist)
        st["touch"] = max(st["touch"], len(touching))
        if name in snap_at and name not in snaps and s >= snap_at[name]:
            renderer.update_scene(d, cam)
            snaps[name] = renderer.render().copy()
    print(f"cycle length {CYCLE:.1f} s, finite={bool(np.isfinite(d.qpos).all())}")
    print("phase         max bracket-vs-partner err (mm)  max links touching partner  max penetration (mm)")
    for name, _ in PHASES:
        st = stats.get(name)
        if st:
            print(f"  {name:12s} {st['pos_err'] * 1000:10.1f} {st['touch']:24d} {st['max_pen'] * 1000:20.1f}")
    grid = np.vstack([np.hstack([snaps[k] for k in ("reach", "grip", "shake")]),
                      np.hstack([snaps[k] for k in ("release", "return")] + [np.zeros_like(snaps["reach"])])])
    out = os.path.join(HERE, "handshake_frames.png")
    write_png(out, np.ascontiguousarray(grid))
    print("saved", out)


def run_record(path, cycles=1, fps=30, width=1280, height=720):
    """Render the handshake offscreen to an MP4 (needs: pip install imageio[ffmpeg])."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("Recording needs imageio + ffmpeg:  pip install \"imageio[ffmpeg]\"")

    m = build_model()
    d = mujoco.MjData(m)
    hs = Handshake(m)
    reset(m, d)
    renderer = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = [0.45, 0.0, 0.5]
    cam.distance, cam.azimuth, cam.elevation = 1.45, 125, -12
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0   # hide collision boxes, show only the visual meshes

    duration = cycles * CYCLE
    frame_dt = 1.0 / fps
    next_frame = 0.0
    step = 0
    n_frames = int(duration * fps)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=8) as writer:
        while d.time < duration:
            if step % 10 == 0:
                hs.apply(d, d.time)
            hs.update_partner(d, d.time)
            mujoco.mj_step(m, d)
            step += 1
            if d.time >= next_frame:
                renderer.update_scene(d, cam, opt)
                writer.append_data(renderer.render())
                next_frame += frame_dt
                done = int(round(next_frame * fps))
                if done % fps == 0:
                    print(f"\r  {done}/{n_frames} frames", end="", flush=True)
    print(f"\nsaved {path}  ({duration:.1f} s, {width}x{height} @ {fps} fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="headless run with metrics + frames")
    ap.add_argument("--record", nargs="?", const=os.path.join(HERE, "handshake.mp4"), metavar="FILE.mp4",
                    help="render an MP4 instead of opening the viewer (default: handshake.mp4)")
    ap.add_argument("--cycles", type=int, default=1, help="handshake cycles to record")
    args = ap.parse_args()
    if args.record:
        run_record(args.record, args.cycles)
    elif args.check:
        run_check()
    else:
        run_viewer()
