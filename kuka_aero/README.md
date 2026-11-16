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
..\.venv\Scripts\python handshake.py               # viewer + control window, starts idle
..\.venv\Scripts\python handshake.py --autostart   # viewer, start handshaking right away
..\.venv\Scripts\python handshake.py --check       # headless: metrics + handshake_frames.png
```
Start/Stop: buttons in the control window, Space/Enter there, or Enter in the 3D viewer.
Start loops handshakes (~16 s per cycle) until Stop. Stop winds down gracefully from any
phase: before contact it reverses out; mid-grip it releases; mid-shake it finishes the
current half-stroke, then releases. The partner leaves and the arm returns home, then idles.

Sequence: reach (fingers forward, thumb up) -> partner hand comes in -> the person closes
their hand fully (partner_grip) -> the robot wraps (grip) -> 3 gentle shakes (22 mm at
1.25 Hz, eased in and out) -> robot opens (release) -> person opens (partner_release) ->
partner leaves -> back to home. Tunables are constants at the top of `handshake.py`
(`SHAKE_PALM_WORLD`, `SHAKE_AMPLITUDE`, `SHAKE_FREQ`, `HAND_GRIP`, `PHASES`, ...).

The partner (`partner_hand.py`) is an anatomical right hand: palm with thenar pad, four
3-phalanx fingers, 3-segment thumb, wrist and sleeve. Its 15 finger joints have soft
position actuators, so it grips back. It is softly welded to a scripted target while
approaching/leaving and to the robot's hand while gripped (a relaxed arm going along with
the shake). Its contacts are stiffer than the Aero Hand's soft defaults so a firm grip
doesn't sink in.

Palm contact: the partner's palm face is a 6 x 5 grid of spring-loaded tissue pads
(400 N/m each, 6 mm travel; hidden collision group 3) over a recessed rigid core, so it
conforms to the robot's stepped palm and spreads the load (~20 N over ~14 taxels while
holding). The hand rides on a palm-normal slide with a force-limited press (15 N, the
person's arm pushing in), so palm pressure comes from the press and the grips rather than
from a weld forcing the hands into each other.

Thumbs: the robot thumb is abducted (1.4 rad) so it rises off its palm and rests on the
back of the partner's hand; the partner thumb crosses the robot's thumb web and hooks
over the top of the robot hand toward its index knuckle.

Record a video (needs `pip install "imageio[ffmpeg]"`):
```
..\.venv\Scripts\python handshake.py --record             # -> handshake.mp4
..\.venv\Scripts\python handshake.py --record --cycles 3
```

## Tactile skin (simulated)
`tactile.py` adds the planned sensors to the Aero Hand:
- **Palm: Velostat matrix.** 138 taxels at 8 mm pitch, ray-cast onto the palm's curved
  collision surface, each a MuJoCo `touch` sensor (normal force only). The model adds
  creep drift under load, 0.02 N noise, a 0.06 N deadband, 10 N/taxel full scale and a
  10-bit ADC. It is re-zeroed at the start of every handshake.
- **Fingers + thumb: MLX90393 Hall pads.** 14 pads, one per phalanx (thumb: proximal +
  distal). Each reads 3 axes in the pad frame: normal, shear along the phalanx, shear
  across it. Only palmar-side contacts count. Slip is flagged when any contact on a pad
  uses more than 80% of its friction cone. `HALL["interference"]` adds a constant offset
  per pad as a stand-in for servo magnetic fields.

Readouts:
```
..\.venv\Scripts\python handshake.py --check                                  # per-phase summary + tactile_log.csv (50 Hz)
..\.venv\Scripts\python handshake.py --record handshake_tactile.mp4 --tactile # close-up + live panel
..\.venv\Scripts\python handshake.py                                          # viewer (taxels/pads tinted by force)
                                                                              #   + tactile readout in the control window
..\.venv\Scripts\python handshake.py --no-panel                               # control window without the readout
```
For realistic forces, the Aero Hand tendon actuators are capped at
`HAND_FORCE_LIMIT = 20 N` (an assumption; the source model has no limit).
