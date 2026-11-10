"""Video frames with a live tactile readout panel next to the 3D view.

Drawn with MuJoCo's own OpenGL helpers (mjr_rectangle / mjr_drawPixels / mjr_label),
so no plotting library is needed. Uses the Renderer's GL context directly.
"""

import mujoco
import numpy as np

from tactile import FINGERS, HALL, VELOSTAT, heat_rgb_u8

BG = (0.09, 0.10, 0.13)
TEXT = (0.92, 0.93, 0.95)
DIM = (0.60, 0.63, 0.70)
NORMAL_COL = (0.98, 0.70, 0.20)
SHEAR_COL = (0.35, 0.75, 0.98)
SLIP_COL = (0.95, 0.30, 0.30)

PALM_SCALE = 2.0     # N per taxel mapped to full heat colour
PAD_SCALE = 10.0     # N mapped to a full bar


class ReadoutRenderer:
    def __init__(self, m, height=720, width_3d=1040, width_panel=560):
        self.m = m
        self.h, self.w3, self.wp = height, width_3d, width_panel
        self.r = mujoco.Renderer(m, height, width_3d + width_panel,
                                 font_scale=mujoco.mjtFontScale.mjFONTSCALE_150)
        self.font = mujoco.mjtFont.mjFONT_NORMAL

    def close(self):
        self.r.close()

    # --- small drawing helpers (y measured from the top of the panel) ---
    def _rect(self, x, y, w, h):
        return mujoco.MjrRect(int(self.w3 + x), int(self.h - y - h), int(w), int(h))

    def _fill(self, x, y, w, h, rgb, a=1.0):
        mujoco.mjr_rectangle(self._rect(x, y, w, h), *rgb, a)

    def _text(self, x, y, w, h, txt, rgb=TEXT, bg=BG, font=None):
        # Left-aligned text at the rect's top-left. (mjr_label silently drops text wider
        # than its box, so it isn't used here; overlay text colour is fixed by MuJoCo.)
        mujoco.mjr_overlay(font if font is not None else self.font,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, self._rect(x, y - 6, w, h + 12),
                           txt, "", self.con)

    def render(self, d, cam, opt, frame, phase):
        r = self.r
        r.update_scene(d, cam, opt)
        if r._gl_context:
            r._gl_context.make_current()
        self.con = con = r._mjr_context
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, self.w3, self.h), r.scene, con)
        self._fill(0, 0, self.wp, self.h, BG)

        self._text(20, 14, 520, 40, f"TACTILE   t={frame.time:5.2f}s   {phase}")

        # Palm Velostat heatmap.
        self._text(20, 62, 520, 26, f"PALM  Velostat {frame.palm.shape[0]}x{frame.palm.shape[1]} @ 8 mm"
                   f"   total {frame.palm_total:5.1f} N   active {frame.palm_active}", rgb=DIM)
        grid = frame.palm
        rows, cols = grid.shape
        cell = min(240 // rows, 300 // cols)
        img = heat_rgb_u8(grid / PALM_SCALE)
        img[np.isnan(grid)] = (30, 32, 40)
        img = np.repeat(np.repeat(img, cell, axis=0), cell, axis=1)
        img[::cell, :, :] = img[::cell, :, :] // 2          # faint grid lines
        img[:, ::cell, :] = img[:, ::cell, :] // 2
        ph, pw = img.shape[:2]
        px, py = 20, 94
        images = [(img, self._rect(px, py, pw, ph))]
        # Colour scale.
        sx = px + pw + 24
        ramp = heat_rgb_u8(np.linspace(1, 0, ph)[:, None].repeat(14, 1))
        images.append((ramp, self._rect(sx, py, 14, ph)))
        self._text(sx + 18, py - 6, 120, 22, f"{PALM_SCALE:.0f} N", rgb=DIM)
        self._text(sx + 18, py + ph - 16, 120, 22, "0 N", rgb=DIM)
        self._text(sx + 18, py + ph // 2 - 22, 160, 22, "thumb side ^", rgb=DIM)
        self._text(sx + 18, py + ph // 2, 160, 22, "fingers ->", rgb=DIM)

        # Hall pads: one row per finger, three segments + thumb.
        y0 = py + ph + 22
        self._text(20, y0, 520, 26, "PADS  MLX90393 Hall    orange=normal  blue=|shear|  red=slip",
                   rgb=DIM)
        y = y0 + 32
        names = [(f, (f"{f}_proximal", f"{f}_middle", f"{f}_distal")) for f in FINGERS]
        names.append(("thumb", ("thumb_proximal", None, "thumb_distal")))
        seg_w = 150
        for label, segs in names:
            self._text(20, y, 70, 22, label, rgb=TEXT)
            for k, pad in enumerate(segs):
                x = 96 + k * seg_w
                self._fill(x, y + 3, seg_w - 12, 18, (0.16, 0.17, 0.21))
                if pad is None:
                    continue
                fn, s1, s2 = frame.pads[pad]
                ft = float(np.hypot(s1, s2))
                wn = (seg_w - 12) * min(fn / PAD_SCALE, 1.0)
                wt = (seg_w - 12) * min(ft / PAD_SCALE, 1.0)
                col = SLIP_COL if frame.slip[pad] else NORMAL_COL
                if wn > 0.5:
                    self._fill(x, y + 3, wn, 11, col)
                if wt > 0.5:
                    self._fill(x, y + 15, wt, 6, SHEAR_COL)
                if fn > 0.05:
                    self._text(x + 2, y + 22, seg_w - 14, 16, f"{fn:4.1f}N /{ft:4.1f}", rgb=DIM)
            y += 44
        self._text(96, y, 3 * seg_w, 20, "proximal          middle          distal", rgb=DIM)
        self._text(20, self.h - 34, 520, 24,
                   f"Velostat FS {VELOSTAT['full_scale']:.0f} N/taxel, {VELOSTAT['adc_bits']}-bit;"
                   f"  Hall FS {HALL['full_scale_normal']:.0f} N", rgb=DIM)

        # Images last: mjr_drawPixels leaves GL state that breaks later text drawing.
        for pix, rect in images:
            mujoco.mjr_drawPixels(np.ascontiguousarray(np.flipud(pix)).ravel(), None, rect, con)

        buf = np.empty((self.h, self.w3 + self.wp, 3), np.uint8)
        mujoco.mjr_readPixels(buf, None, mujoco.MjrRect(0, 0, self.w3 + self.wp, self.h), con)
        return np.flipud(buf)
