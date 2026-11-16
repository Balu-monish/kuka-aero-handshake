"""Robot 'head' camera: a fixed RGB-D camera modelled on an Intel RealSense D435.

The scene is smaller than a standing person (the hands meet at 0.55 m with the iiwa on
the floor), so the head sits where eyes would be relative to the handshake: ~0.6 m above
it, just behind the robot's base, looking down at the person.

Two co-located MuJoCo cameras, like the D435's depth module and RGB sensor:
  head_depth: 848 x 480, 87 x 58 deg FOV, 0.3-3 m, 1 mm units, stereo noise model
  head_rgb:   1280 x 720, 69 x 42 deg FOV, 15 mm to the right of the depth origin
Segmentation (geom ids per pixel) comes from the depth camera; it is the ground truth
for detection, not something a real camera provides.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

HEAD_POS = np.array([-0.10, 0.0, 1.15])       # m, world: behind and above the robot base
HEAD_LOOK_AT = np.array([1.00, 0.0, 0.66])    # m, world: the person / approach zone

D435 = dict(
    depth_size=(480, 848), depth_fovy=58.0,   # (rows, cols), vertical FOV deg (87 deg horizontal)
    rgb_size=(720, 1280), rgb_fovy=42.0,      # 69 deg horizontal
    rgb_offset=0.015,                         # m, RGB sensor to the right of the depth origin
    z_min=0.3, z_max=3.0,                     # m, valid depth range (0 = no data, as on the D435)
    baseline=0.050,                           # m, stereo baseline
    subpixel=0.08,                            # px, disparity noise (RMS)
    depth_unit=0.001,                         # m, z16 depth units
)


def _look_xyaxes(pos, target):
    """MuJoCo camera xyaxes (image right, image up) for a camera at pos looking at target."""
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.concatenate([right, up])


def add_head_camera(spec: mujoco.MjSpec) -> None:
    xy = _look_xyaxes(HEAD_POS, HEAD_LOOK_AT)
    for name, fovy, offset in (("head_depth", D435["depth_fovy"], 0.0),
                               ("head_rgb", D435["rgb_fovy"], D435["rgb_offset"])):
        cam = spec.worldbody.add_camera(name=name, pos=HEAD_POS + xy[:3] * offset,
                                        fovy=fovy)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = xy


@dataclass
class HeadFrame:
    time: float
    rgb: np.ndarray        # (720, 1280, 3) uint8
    depth: np.ndarray      # (480, 848) float32 m, 0 = no data
    depth_true: np.ndarray  # (480, 848) float32 m, noiseless
    seg: np.ndarray        # (480, 848) int32 geom id, -1 = background / not a geom


class HeadCamera:
    """Renders the head camera's RGB, depth (with D435-like noise) and segmentation."""

    def __init__(self, m: mujoco.MjModel, seed: int = 0):
        self.m = m
        self.rng = np.random.default_rng(seed)
        dh, dw = D435["depth_size"]
        rh, rw = D435["rgb_size"]
        self.r_rgb = mujoco.Renderer(m, rh, rw)
        self.r_depth = mujoco.Renderer(m, dh, dw)
        self.r_depth.enable_depth_rendering()
        self.r_seg = mujoco.Renderer(m, dh, dw)
        self.r_seg.enable_segmentation_rendering()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0          # what a camera sees: visual geoms, no collision boxes
        self.opt.sitegroup[4] = 0          # no tactile sites
        # Depth noise from stereo geometry: sigma_z = z^2 * subpixel / (f_px * baseline).
        self.f_px = (dh / 2) / np.tan(np.radians(D435["depth_fovy"] / 2))

    def close(self):
        for r in (self.r_rgb, self.r_depth, self.r_seg):
            r.close()

    def _seg_fallback(self):
        """Same decoding as mujoco.Renderer's segmentation, but pixels whose id falls outside
        the scene (seen occasionally at object edges) count as background instead of
        raising IndexError. Reads the frame the failed render() just drew."""
        r = self.r_seg
        rgb = np.empty((r._height, r._width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, r._rect, r._mjr_context)
        rgb = rgb.astype(np.int64)
        segimage = rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 65536
        n = r._scene.ngeom
        table = np.full((n + 2, 2), -1, dtype=np.int32)
        for g in r._scene.geoms[:n]:
            if g.segid != -1:
                table[g.segid + 1] = (g.objid, g.objtype)
        out = table[np.clip(segimage, 0, n + 1)]
        out[segimage > n + 1] = -1
        return np.flipud(out)

    def render_rgb(self, d: mujoco.MjData) -> np.ndarray:
        self.r_rgb.update_scene(d, "head_rgb", self.opt)
        return self.r_rgb.render().copy()

    def render(self, d: mujoco.MjData, rgb=True, seg=True) -> HeadFrame:
        self.r_depth.update_scene(d, "head_depth", self.opt)
        z = self.r_depth.render().astype(np.float32)
        noisy = z + self.rng.standard_normal(z.shape, dtype=np.float32) * (
            z * z * np.float32(D435["subpixel"] / (self.f_px * D435["baseline"])))
        noisy = np.round(noisy / D435["depth_unit"]) * D435["depth_unit"]
        noisy[(z < D435["z_min"]) | (z > D435["z_max"])] = 0.0
        img = None
        if rgb:
            self.r_rgb.update_scene(d, "head_rgb", self.opt)
            img = self.r_rgb.render().copy()
        ids = None
        if seg:
            self.r_seg.update_scene(d, "head_depth", self.opt)
            try:
                s = self.r_seg.render()
            except IndexError:      # rare out-of-range segment id: decode it ourselves
                s = self._seg_fallback()
            ids = np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1).astype(np.int32)
        return HeadFrame(d.time, img, noisy.astype(np.float32), z, ids)


def depth_rgb_u8(depth, z_min=D435["z_min"], z_max=2.0):
    """Colour-map depth for display (near = warm, far = cool, no data = black)."""
    x = np.clip((depth - z_min) / (z_max - z_min), 0.0, 1.0)
    rgb = np.stack([1.0 - x, 1.0 - np.abs(2 * x - 1.0), x], axis=-1)
    rgb = (rgb * 220 + 25).astype(np.uint8)
    rgb[depth <= 0] = 0
    return rgb
