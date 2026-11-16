"""Hand detection and tracking from the robot's head camera, plus the "hand offered" trigger.

Stage 1 detector (simulation): the hand mask comes from MuJoCo's segmentation (a perfect
2D detector), and its 3D position from the camera's noisy D435-like depth. So the 3D part
is realistic (noise, quantisation, only the visible surface), the 2D part is idealised; a
learned detector (e.g. MediaPipe Hands on the RGB image) can replace `_mask()` later.

Pipeline, at DETECT_HZ:
  mask -> back-project the hand pixels with depth -> median 3D point of the visible surface
  -> + calibrated offset -> estimate of the hand frame (partner root) position
  -> alpha-beta tracker (position, velocity)
  -> trigger: hand inside the meeting zone and nearly still for OFFER_DWELL seconds.
The offset maps "middle of what the camera sees" to the hand frame; it is measured once
with the hand at the nominal offer pose, from the same viewpoint.

Self-occlusion: as the robot's hand closes in, it hides part of the person's hand and the
median of what is left drifts by centimetres. A real robot knows where its own arm is in
the image (forward kinematics + its model); here the segmentation gives that self-mask.
A frame is flagged occluded when the robot covers more than OCCLUDED_FRAC of the area
around the visible hand (its box, dilated) AND the visible hand has shrunk below
SHRINK_FRAC of its recent unoccluded size. (Robot pixels nearby alone are not enough: a
hand pulling away from a reaching arm is near it but fully visible.) Occluded frames keep
the track alive but don't move the estimate.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from head_camera import D435, HeadCamera

DETECT_HZ = 10.0
MIN_PIXELS = 150         # fewer visible hand pixels than this = not detected
ZONE_RADIUS = 0.10       # m, around the nominal meeting point: where a hand counts as offered
STILL_SPEED = 0.04       # m/s, "holding still"
OFFER_DWELL = 0.4        # s, still inside the zone this long -> offered
LOST_AFTER = 0.5         # s without detections -> track dropped
OCCLUDED_FRAC = 0.10     # robot pixels around the visible hand above this -> occluded
BOX_MARGIN = 20          # px, dilation of the hand's box for the occlusion test
SHRINK_FRAC = 0.85       # visible hand below this fraction of its recent size -> hidden
TRACK_ALPHA, TRACK_BETA = 0.5, 0.2


@dataclass
class Detection:
    time: float
    visible: bool
    pixels: int
    raw: np.ndarray = None           # this frame's estimate (hand frame position, world)
    occluded: bool = False           # the robot covers part of the hand: estimate unreliable
    bbox: tuple = None               # (v0, v1, u0, u1) in depth-image pixels
    depth: np.ndarray = None         # the depth frame it came from (for display)


@dataclass
class Track:
    pos: np.ndarray = None
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_seen: float = -1e9
    dwell: float = 0.0

    @property
    def speed(self):
        return float(np.linalg.norm(self.vel))


class HandDetector:
    def __init__(self, m: mujoco.MjModel, nominal_pos, seed=0):
        self.m = m
        self.camera = HeadCamera(m, seed=seed)
        self.nominal = np.asarray(nominal_pos, float)
        self.cam_id = m.camera("head_depth").id
        h, w = D435["depth_size"]
        self.f = (h / 2) / np.tan(np.radians(D435["depth_fovy"] / 2))
        self.cx, self.cy = (w - 1) / 2, (h - 1) / 2
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        self.ray_x = (uu - self.cx) / self.f        # camera frame: x right, y up, looking -z
        self.ray_y = -(vv - self.cy) / self.f
        names = [m.geom(g).name for g in range(m.ngeom)]
        self.hand_ids = np.array([g for g, n in enumerate(names) if n.startswith("partner")
                                  and n not in ("partner_forearm", "partner_wrist")])
        partner = m.body("partner").id
        self.robot_ids = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] != 0
                                   and m.body_rootid[m.geom_bodyid[g]] != partner
                                   and not names[g].startswith("human")])
        self.offset = np.zeros(3)
        self.ref_pixels = None
        self.track = Track()
        self.last = Detection(0.0, False, 0)
        self.state = "none"          # none / tracking / offered

    def close(self):
        self.camera.close()

    # --- measurement ---
    def _mask(self, seg):
        return np.isin(seg, self.hand_ids)

    def _measure(self, d, depth_key="depth"):
        frame = self.camera.render(d, rgb=False, seg=True)
        z = getattr(frame, depth_key)
        mask = self._mask(frame.seg) & (z > 0)
        n = int(mask.sum())
        if n < MIN_PIXELS:
            return Detection(d.time, False, n, depth=frame.depth)
        zs = z[mask]
        pts_cam = np.stack([self.ray_x[mask] * zs, self.ray_y[mask] * zs, -zs], axis=1)
        R = d.cam_xmat[self.cam_id].reshape(3, 3)
        pts = pts_cam @ R.T + d.cam_xpos[self.cam_id]
        vs, us = np.nonzero(mask)
        box = (vs.min(), vs.max(), us.min(), us.max())
        occluded = False
        v0, v1 = max(box[0] - BOX_MARGIN, 0), box[1] + BOX_MARGIN + 1
        u0, u1 = max(box[2] - BOX_MARGIN, 0), box[3] + BOX_MARGIN + 1
        near = np.isin(frame.seg[v0:v1, u0:u1], self.robot_ids).mean() > OCCLUDED_FRAC
        shrunk = self.ref_pixels is not None and n < SHRINK_FRAC * self.ref_pixels
        occluded = near and shrunk
        if not occluded:   # running size of the unoccluded hand
            self.ref_pixels = n if self.ref_pixels is None else 0.7 * self.ref_pixels + 0.3 * n
        return Detection(d.time, True, n, raw=np.median(pts, axis=0) + self.offset, bbox=box,
                         depth=frame.depth, occluded=occluded)

    def calibrate(self, d_nominal, true_pos):
        """Offset from the visible-surface median to the hand frame, measured with the hand
        at the nominal offer pose (noise-free depth)."""
        self.offset = np.zeros(3)
        det = self._measure(d_nominal, depth_key="depth_true")
        self.ref_pixels = None
        if not det.visible:
            raise RuntimeError("calibration: hand not visible from the head camera")
        self.offset = np.asarray(true_pos) - det.raw
        return det.pixels

    # --- tracking + trigger ---
    def update(self, d, dt):
        det = self._measure(d)
        self.last = det
        tr = self.track
        if det.visible and det.occluded and tr.pos is not None:
            tr.last_seen = d.time          # still there, just partly hidden: hold the estimate
        elif det.visible:
            if tr.pos is None or d.time - tr.last_seen > LOST_AFTER:
                tr.pos, tr.vel, tr.dwell = det.raw.copy(), np.zeros(3), 0.0
            else:
                pred = tr.pos + tr.vel * dt
                r = det.raw - pred
                tr.pos = pred + TRACK_ALPHA * r
                tr.vel = tr.vel + TRACK_BETA * r / dt
            tr.last_seen = d.time
        elif tr.pos is not None and d.time - tr.last_seen > LOST_AFTER:
            tr.pos, tr.dwell = None, 0.0
        if tr.pos is None:
            self.state = "none"
            return det
        in_zone = np.linalg.norm(tr.pos - self.nominal) < ZONE_RADIUS
        still = tr.speed < STILL_SPEED
        tr.dwell = tr.dwell + dt if (det.visible and not det.occluded and in_zone and still) else 0.0
        self.state = "offered" if tr.dwell >= OFFER_DWELL else "tracking"
        return det

    @property
    def offered(self):
        return self.state == "offered"

    @property
    def estimate(self):
        return None if self.track.pos is None else self.track.pos.copy()

    def project_rgb(self, d, p):
        """Pixel (u, v) of world point p in the head RGB image, or None if behind it."""
        cid = self.m.camera("head_rgb").id
        q = d.cam_xmat[cid].reshape(3, 3).T @ (p - d.cam_xpos[cid])
        if q[2] >= 0:
            return None
        h, w = D435["rgb_size"]
        f = (h / 2) / np.tan(np.radians(D435["rgb_fovy"] / 2))
        return (w - 1) / 2 + f * q[0] / -q[2], (h - 1) / 2 - f * q[1] / -q[2]
