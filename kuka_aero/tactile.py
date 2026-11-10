"""Simulated tactile skin for the Aero Hand.

Two sensor types, following the hardware plan:

* Palm: Velostat piezoresistive matrix. A grid of taxels (8 mm pitch) tiling the palm's
  curved collision surface. Each taxel is a MuJoCo `touch` sensor (sum of normal contact
  force inside its patch). Velostat senses normal force only; the model adds creep drift
  under sustained load, noise, saturation and ADC quantisation, and supports re-zeroing
  between handshakes.

* Fingers and thumb: MLX90393 Hall-effect pads (ReSkin/AnySkin style). One pad per
  phalanx, 3-axis: normal force plus two shear components in the pad frame, computed
  from the contact forces on the palmar side of each phalanx. A slip flag is raised when
  any of its contacts nears the friction limit (|shear| > 0.8 mu normal). An optional constant offset per pad stands in for
  magnetic interference from the servos.

Usage:
    add_tactile(spec)                  # before compiling the model
    skin = TactileSensors(model)
    frame = skin.read(data, dt)        # each control step
    skin.rezero()                      # e.g. between handshakes
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

MOUNT = "hand_tetheria_mount"
PALM = "hand_palm"

# --- Velostat palm matrix ---
TAXEL_PITCH = 0.008                    # m (5-10 mm target range)
TAXEL_DEPTH = 0.004                    # m, half-thickness of each sensing patch
PALM_GRID_X = (0.034, 0.142)           # bracket-frame extent scanned for palm surface
PALM_GRID_Y = (-0.062, 0.058)
VELOSTAT = dict(
    noise=0.02,          # N, per-taxel RMS
    deadband=0.06,       # N, readings below this report 0
    full_scale=10.0,     # N, saturation per taxel
    adc_bits=10,
    creep=0.04,          # drift gained per second, as a fraction of the applied load
    recovery_tau=30.0,   # s, drift relaxation time once unloaded
)

# --- MLX90393 Hall pads ---
FINGERS = ("index", "middle", "ring", "pinky")
PAD_BODIES = [(f"{f}_{seg}", f"hand_right_{f}_{seg}_link")
              for f in FINGERS for seg in ("proximal", "middle", "distal")]
PAD_BODIES += [("thumb_proximal", "hand_right_thumb_proximal_link"),
               ("thumb_distal", "hand_right_thumb_distal_link")]
HALL = dict(
    noise_normal=0.01,   # N RMS
    noise_shear=0.01,    # N RMS
    full_scale_normal=20.0,
    full_scale_shear=10.0,
    interference=0.0,    # N, constant offset per pad (servo magnetic field stand-in)
)
SLIP_RATIO = 0.8         # slip flagged when a contact uses > 80% of its friction cone
SLIP_MIN_NORMAL = 0.05   # N

# Pose used to decide which face of each phalanx is the palmar (sensing) side.
_CURL_TARGET = np.array([0.09, 0.0, 0.02])   # point above the palm, bracket frame


def _bracket(m, d):
    b = m.body(MOUNT).id
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def add_tactile(spec: mujoco.MjSpec) -> None:
    """Add Velostat taxel sites + touch sensors and Hall-pad sites to the robot hand."""
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)          # qpos0: fingers straight, thumb neutral
    p0, R0 = _bracket(m, d)
    palm_id = m.body(PALM).id
    Rp, pp = d.xmat[palm_id].reshape(3, 3), d.xpos[palm_id]
    group = np.zeros(6, np.uint8)
    group[3] = 1                        # robot collision geoms only
    geomid = np.zeros(1, np.int32)
    q_rel = np.zeros(4)
    mujoco.mju_mat2Quat(q_rel, (Rp.T @ R0).ravel())

    palm = spec.body(PALM)
    nx = int(round((PALM_GRID_X[1] - PALM_GRID_X[0]) / TAXEL_PITCH)) + 1
    ny = int(round((PALM_GRID_Y[1] - PALM_GRID_Y[0]) / TAXEL_PITCH)) + 1
    for i in range(ny):
        for j in range(nx):
            local = np.array([PALM_GRID_X[0] + j * TAXEL_PITCH,
                              PALM_GRID_Y[0] + i * TAXEL_PITCH, 0.08])
            start = p0 + R0 @ local
            dist = mujoco.mj_ray(m, d, start, -R0[:, 2], group, 1, -1, geomid)
            if dist < 0 or m.geom_bodyid[geomid[0]] != palm_id:
                continue                # off the palm, or the ray hit a finger/bracket
            hit = start - R0[:, 2] * dist
            name = f"taxel_r{i:02d}_c{j:02d}"
            palm.add_site(name=name, pos=Rp.T @ (hit - pp), quat=q_rel,
                          type=mujoco.mjtGeom.mjGEOM_BOX, group=4,
                          size=[TAXEL_PITCH / 2, TAXEL_PITCH / 2, TAXEL_DEPTH],
                          rgba=[0.2, 0.2, 0.25, 0.35])
            spec.add_sensor(name=name, type=mujoco.mjtSensor.mjSENS_TOUCH,
                            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=name)

    # Hall pads: a site per phalanx, matching its collision box (visualisation + frame).
    for pad, body_name in PAD_BODIES:
        bid = m.body(body_name).id
        geoms = [g for g in range(m.ngeom)
                 if m.geom_bodyid[g] == bid and m.geom_contype[g] and m.geom_group[g] == 3]
        g = max(geoms, key=lambda k: np.prod(m.geom_size[k]))
        spec.body(body_name).add_site(
            name=f"pad_{pad}", pos=m.geom_pos[g], quat=m.geom_quat[g],
            type=mujoco.mjtGeom.mjGEOM_BOX, group=4, size=m.geom_size[g] + 0.0015,
            rgba=[0.2, 0.2, 0.25, 0.35])


@dataclass
class TactileFrame:
    time: float
    palm: np.ndarray                    # (rows, cols) N, NaN where no taxel exists
    palm_true: np.ndarray               # ideal normal force, same layout
    pads: dict = field(default_factory=dict)   # name -> np.array([normal, shear1, shear2]) N
    slip: dict = field(default_factory=dict)   # name -> bool

    @property
    def palm_total(self):
        return float(np.nansum(self.palm))

    @property
    def palm_active(self):
        return int(np.nansum(self.palm > 0.1))


class TactileSensors:
    def __init__(self, m: mujoco.MjModel, seed: int = 0):
        self.m = m
        self.rng = np.random.default_rng(seed)
        # Velostat taxels: (sensor adr, row, col)
        cells = []
        for s in range(m.nsensor):
            name = m.sensor(s).name
            if name.startswith("taxel_r"):
                cells.append((m.sensor_adr[s], int(name[7:9]), int(name[11:13]),
                              m.sensor_objid[s]))
        self.taxel_adr = np.array([c[0] for c in cells])
        self.taxel_rc = np.array([(c[1], c[2]) for c in cells])
        self.taxel_site = np.array([c[3] for c in cells])
        self.shape = (self.taxel_rc[:, 0].max() + 1, self.taxel_rc[:, 1].max() + 1)
        self.drift = np.zeros(len(cells))
        self.baseline = np.zeros(len(cells))

        # Hall pads: geoms per pad + local axes (normal, shear1, shear2) in pad-site frame.
        self.pads = {}
        d = mujoco.MjData(m)
        self._curl_pose(d)
        p0, R0 = _bracket(m, d)
        palm_target = p0 + R0 @ _CURL_TARGET
        # The thumb opposes the fingers: its sensing side faces the index finger.
        thumb_target = d.xpos[m.body("hand_right_index_middle_link").id].copy()
        for pad, body_name in PAD_BODIES:
            bid = m.body(body_name).id
            site = m.site(f"pad_{pad}").id
            geoms = {g for g in range(m.ngeom)
                     if m.geom_bodyid[g] == bid and m.geom_contype[g] and m.geom_group[g] == 3}
            Rs = d.site_xmat[site].reshape(3, 3)
            target = thumb_target if pad.startswith("thumb") else palm_target
            v = Rs.T @ (target - d.site_xpos[site])
            size = m.site_size[site]
            k_long = int(np.argmax(size))                  # phalanx length axis
            cand = [k for k in range(3) if k != k_long]
            k = max(cand, key=lambda a: abs(v[a]))         # palmar axis: across the phalanx
            n_local = np.zeros(3)
            n_local[k] = np.sign(v[k])
            t1 = np.zeros(3)
            t1[k_long] = 1.0                               # shear 1: along the phalanx
            t2 = np.cross(n_local, t1)                     # shear 2: across it
            self.pads[pad] = dict(site=site, geoms=geoms, axes=np.stack([n_local, t1, t2]),
                                  half_thick=float(size[k]))
        self.interference = {p: HALL["interference"] * self.rng.standard_normal(3)
                             for p in self.pads}
        self._force = np.zeros(6)

    def _curl_pose(self, d):
        """Fingers and thumb flexed, so 'facing the palm' picks the palmar side."""
        m = self.m
        mujoco.mj_resetData(m, d)
        for j in range(m.njnt):
            name = m.joint(j).name
            if name.startswith("hand_right_"):
                lo, hi = m.jnt_range[j]
                d.qpos[m.jnt_qposadr[j]] = lo + 0.6 * (hi - lo)
        mujoco.mj_kinematics(m, d)

    def rezero(self):
        """Velostat re-zero: current unloaded drift becomes the new baseline."""
        self.baseline = self.drift.copy()

    def read(self, d: mujoco.MjData, dt: float) -> TactileFrame:
        m = self.m
        # --- Velostat palm ---
        true = d.sensordata[self.taxel_adr].copy()
        v = VELOSTAT
        self.drift += dt * (v["creep"] * true - self.drift / v["recovery_tau"])
        raw = true + self.drift + v["noise"] * self.rng.standard_normal(len(true))
        lsb = v["full_scale"] / (2 ** v["adc_bits"])
        raw = np.round(np.clip(raw, 0.0, v["full_scale"]) / lsb) * lsb
        meas = raw - self.baseline
        meas[meas < v["deadband"]] = 0.0     # readout deadband (~3x noise)
        palm = np.full(self.shape, np.nan)
        palm_true = np.full(self.shape, np.nan)
        palm[self.taxel_rc[:, 0], self.taxel_rc[:, 1]] = meas
        palm_true[self.taxel_rc[:, 0], self.taxel_rc[:, 1]] = true

        # --- Hall pads ---
        forces = {p: np.zeros(3) for p in self.pads}
        cone = {p: 0.0 for p in self.pads}
        for i in range(d.ncon):
            c = d.contact[i]
            for pad, info in self.pads.items():
                if c.geom1 in info["geoms"]:
                    sgn = -1.0
                elif c.geom2 in info["geoms"]:
                    sgn = 1.0
                else:
                    continue
                site = info["site"]
                Rs = d.site_xmat[site].reshape(3, 3)
                n_world = Rs @ info["axes"][0]
                if (c.pos - d.site_xpos[site]) @ n_world < -0.4 * info["half_thick"]:
                    continue                      # back of the finger: no sensor there
                mujoco.mj_contactForce(m, d, i, self._force)
                f_world = sgn * (c.frame.reshape(3, 3).T @ self._force[:3])  # force on pad
                local = info["axes"] @ (Rs.T @ f_world)
                forces[pad] += np.array([-local[0], local[1], local[2]])     # normal = pressing in
                fn, ft = self._force[0], np.hypot(self._force[1], self._force[2])
                if fn > SLIP_MIN_NORMAL:          # friction-cone usage of this contact
                    cone[pad] = max(cone[pad], ft / (c.friction[0] * fn))
        pads, slip = {}, {}
        h = HALL
        for pad, f in forces.items():
            noisy = f + self.interference[pad] + np.array(
                [h["noise_normal"], h["noise_shear"], h["noise_shear"]]) * self.rng.standard_normal(3)
            noisy[0] = np.clip(noisy[0], 0.0, h["full_scale_normal"])
            noisy[1:] = np.clip(noisy[1:], -h["full_scale_shear"], h["full_scale_shear"])
            pads[pad] = noisy
            slip[pad] = bool(cone[pad] > SLIP_RATIO)
        return TactileFrame(time=d.time, palm=palm, palm_true=palm_true, pads=pads, slip=slip)

    def colorize(self, frame: TactileFrame, scale=1.5):
        """Tint taxel and pad sites by force (for the viewer / video)."""
        m = self.m
        vals = frame.palm[self.taxel_rc[:, 0], self.taxel_rc[:, 1]]
        m.site_rgba[self.taxel_site] = heat_rgba(vals / scale)
        for pad, info in self.pads.items():
            m.site_rgba[info["site"]] = heat_rgba(np.array([frame.pads[pad][0] / (2 * scale)]))[0]


def heat_rgba(x):
    """0..1 -> transparent grey -> yellow -> red, as RGBA rows."""
    x = np.clip(np.nan_to_num(np.asarray(x, float)), 0.0, 1.0)
    r = np.clip(0.25 + 1.5 * x, 0, 1)
    g = np.clip(0.25 + 1.2 * x - 1.4 * np.maximum(x - 0.5, 0) * 2, 0, 1)
    b = np.clip(0.3 - 0.6 * x, 0, 1)
    a = np.where(x > 0.02, 0.9, 0.25)
    return np.stack([r, g, b, a], axis=-1)


def heat_rgb_u8(x):
    return (heat_rgba(x)[..., :3] * 255).astype(np.uint8)
