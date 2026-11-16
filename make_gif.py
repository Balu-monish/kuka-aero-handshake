"""Render one handshake episode to an animated GIF for the README.

  python make_gif.py                  # -> media/handshake.gif
  python make_gif.py --seed 2 --speed 1.5 --fps 12 --out media/other.gif

One scripted "offer" episode, from the person resting to the robot back at home: the
person offers their hand, the robot sees it (head camera, right), reaches out, they shake,
let go and both return (scene view, left). Encoded with ffmpeg's palette filters (the
ffmpeg binary that ships with imageio-ffmpeg), so no extra tools are needed.
"""

import argparse
import os
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np

import handshake as H


def render_episode(seed=1, speed=1.5, fps=12, scene_size=(360, 480), cam_w=320):
    """Frames of one offer episode: scene view (left) beside the head camera (right)."""
    ses = H.Session(script=["offer"], seed=seed)
    m, d, r = ses.m, ses.d, ses.robot
    h, w = scene_size
    renderer = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = [0.62, -0.03, 0.58]
    cam.distance, cam.azimuth, cam.elevation = 1.75, 62, -14
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    frame_dt = speed / fps                 # simulated seconds per GIF frame
    next_frame, frames, done_at = 0.0, [], None
    while d.time < 40.0:
        ses.step()
        if d.time >= next_frame:
            renderer.update_scene(d, cam, opt)
            head = H.head_view(ses, ses.detector.camera.render_rgb(d), cam_w, h // 2)
            frames.append(np.hstack([renderer.render(), head]))
            next_frame += frame_dt
        if r.handshakes >= 1 and done_at is None:
            done_at = d.time
        if done_at is not None and r.state == "home" and ses.human.activity == "rest" \
                and d.time > done_at + H.T_RETURN + 1.0:
            break
    ses.close()
    renderer.close()
    return frames


def write_gif(frames, path, fps=12, colors=128):
    import imageio.v2 as imageio
    import imageio_ffmpeg

    tmp = tempfile.mkdtemp()
    try:
        for i, f in enumerate(frames):
            imageio.imwrite(os.path.join(tmp, f"{i:04d}.png"), f)
        graph = (f"split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
                 "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-framerate", str(fps), "-i", os.path.join(tmp, "%04d.png"),
                        "-filter_complex", graph, "-loop", "0", path], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.5, help="playback speed vs real time")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(H.MEDIA, "handshake.gif"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames = render_episode(args.seed, args.speed, args.fps)
    write_gif(frames, args.out, args.fps)
    print(f"saved {args.out}: {len(frames)} frames, {len(frames) / args.fps:.1f} s, "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB")
