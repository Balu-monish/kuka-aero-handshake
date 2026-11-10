# KUKA LBR iiwa 14 + TetherIA Aero Hand Open (MuJoCo)

## Sources (clone next to this folder)
- `aero-hand-open/`: https://github.com/TetherIA/aero-hand-open (`git submodule update --init sim_rl/simulation`)
- `mujoco_menagerie/`: https://github.com/google-deepmind/mujoco_menagerie (sparse: `kuka_iiwa_14`)

## Run
```
..\.venv\Scripts\python -m mujoco.viewer --mjcf=scene.xml
```
Then Simulation -> Load key (`home`).

- `build_scene.py`: regenerates `scene.xml` (hand mounted on the iiwa flange).
- `check_scene.py`: sanity checks + `snapshot.png`.

## Handshake demo
```
..\.venv\Scripts\python handshake.py           # viewer, loops the handshake (~13 s per cycle)
..\.venv\Scripts\python handshake.py --check   # headless: metrics + handshake_frames.png
```
Sequence: reach (fingers forward, thumb up) -> partner hand comes in -> grip -> 3 shakes
-> release -> partner leaves -> back to home. Tunables are constants at the top of
`handshake.py` (`SHAKE_PALM_WORLD`, `SHAKE_AMPLITUDE`, `SHAKE_FREQ`, `HAND_GRIP`, ...).
The partner (`partner_hand.py`) is an anatomical right hand: palm with thenar pad, four
3-phalanx fingers, 3-segment thumb, wrist and sleeve. Its 15 finger joints have soft
position actuators, so it grips back. It is softly welded to a scripted target while
approaching/leaving and to the robot's hand while gripped (a relaxed arm going along with
the shake). Its contacts are stiffer than the Aero Hand's soft defaults so a firm grip
doesn't sink in.

Record a video (needs `pip install "imageio[ffmpeg]"`):
```
..\.venv\Scripts\python handshake.py --record             # -> handshake.mp4
..\.venv\Scripts\python handshake.py --record --cycles 3
```
