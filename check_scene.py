"""Sanity checks for scene.xml: loads, holds home pose, hand closes/opens, renders a snapshot."""

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
mujoco.mj_forward(m, d)

# Mount alignment: bracket face centre should sit on the flange site.
site = d.site("attachment_site")
mb = d.body("hand_tetheria_mount")
face = mb.xpos + mb.xmat.reshape(3, 3) @ np.array([-0.022, 0.010, 0.009])
print("face-to-flange offset (mm):", np.round((face - site.xpos) * 1000, 2))
print("hand +x vs flange +z (dot):",
      round(float(mb.xmat.reshape(3, 3)[:, 0] @ site.xmat.reshape(3, 3)[:, 2]), 4))

# Hold home for 3 s.
q0 = d.qpos[:7].copy()
for _ in range(int(3 / m.opt.timestep)):
    mujoco.mj_step(m, d)
print("arm drift after 3 s (deg):", np.round(np.degrees(np.abs(d.qpos[:7] - q0)).max(), 3))
print("palm height (m):", round(float(d.body("hand_palm").xpos[2]), 3))
print("finite state:", bool(np.isfinite(d.qpos).all()))
bad = [(m.geom(c.geom1).name, m.geom(c.geom2).name, round(c.dist * 1000, 1))
       for c in d.contact[:d.ncon] if c.dist < -0.001]
print("contacts at rest:", d.ncon, "penetrating >1mm:", bad)


def finger_angles():
    return np.round(np.degrees(d.qpos[7:]), 0)


print("hand joints, home (deg):", finger_angles())
# Hand actuators set tendon length: high ctrl = long tendon = open, low = closed.
hand_act = slice(7, 14)
lo, hi = m.actuator_ctrlrange[hand_act].T
d.ctrl[hand_act] = hi
for _ in range(int(2 / m.opt.timestep)):
    mujoco.mj_step(m, d)
print("hand joints, open (deg):", finger_angles())
d.ctrl[hand_act] = lo
for _ in range(int(2 / m.opt.timestep)):
    mujoco.mj_step(m, d)
print("hand joints, closed (deg):", finger_angles())

# Snapshot of the home pose.
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
mujoco.mj_forward(m, d)
with mujoco.Renderer(m, 720, 960) as r:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = [0.35, 0, 0.45]
    cam.distance, cam.azimuth, cam.elevation = 1.6, -140, -15
    r.update_scene(d, cam)
    img = r.render()
    cam.lookat[:] = d.body("hand_palm").xpos
    cam.distance = 0.4
    r.update_scene(d, cam)
    close = r.render()
import struct  # noqa: E402
import zlib  # noqa: E402


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


os.makedirs(os.path.join(HERE, "media"), exist_ok=True)
out = os.path.join(HERE, "media", "snapshot.png")
write_png(out, np.ascontiguousarray(np.hstack([img, close])))
print("saved", out)
