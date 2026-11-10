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
The partner is a free body softly welded to a mocap target, so it gives a little when gripped.
