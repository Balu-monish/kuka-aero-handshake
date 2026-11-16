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
..\.venv\Scripts\python handshake.py               # viewer + control window; robot starts off
..\.venv\Scripts\python handshake.py --autostart   # viewer, robot responds right away
..\.venv\Scripts\python handshake.py --seed 3      # repeatable person behaviour (default: random)
..\.venv\Scripts\python handshake.py --check       # headless episode script: detection, trigger
                                                   #   and handshake metrics + handshake_frames.png
```
A seated person (`human_agent.py`) behaves on their own: rests with the hand on their
thigh, then offers a hand (to a random spot within ~5 cm of the meeting point), waves,
hesitates halfway, or offers and pulls back. The robot watches with its head camera and,
once Start is pressed, reaches out only when it sees a hand **offered**: inside the
meeting zone (10 cm) and held still for 0.4 s.

Start/Stop (buttons in the control window, Space/Enter there, or Enter in the 3D viewer)
switch the robot's responses on and off; the person keeps going either way. Stop winds
down gracefully: while reaching it backs off; mid-grip it releases; mid-shake it finishes
the current half-stroke, then releases.

Robot sequence: home (arm tucked low on its right, out of the camera's view) -> reach to
15 cm short of the detected hand -> straight-line approach into it -> the person closes
their hand fully (partner_grip) -> the robot wraps (grip) -> 3 gentle shakes (22 mm at
1.25 Hz, eased in and out) -> robot opens (release) -> person opens (partner_release) ->
retreat -> home. It aborts while reaching if the visible hand moves > 6 cm away, and backs
off if its palm feels no contact (< 1.5 N) 0.8 s after arriving. Tunables are constants
at the top of `handshake.py` (`SHAKE_PALM_WORLD`, `SHAKE_AMPLITUDE`, `HAND_GRIP`, `ENGAGE`,
`T_REACH`, `ABORT_DIST`, ...) and `perception.py` (`ZONE_RADIUS`, `OFFER_DWELL`, ...).

Hand detection (`perception.py`), at 10 Hz: the hand mask comes from MuJoCo segmentation
(a perfect 2D detector, to be replaced by a real one), its 3D position from the noisy
depth (median of the visible surface + an offset calibrated at the nominal pose), then an
alpha-beta tracker. Frames where the robot's own arm hides part of the hand (self-mask
from segmentation, as a real robot would from its kinematics) hold the estimate.

`--check` runs 6 scripted episodes (offer, wave, offer, hesitate, offer_withdraw, offer);
over seeds 1-3: 9/9 offers -> handshake, 0/6 false triggers, 3/3 withdrawals aborted,
commit 0.42-0.60 s after the hand stops, hand estimate 1-5 mm from the truth, shake
overlap <= 4.5 mm, ~28 N over 14 palm taxels while holding.

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

Head camera and person: the robot has a fixed "head" camera (`head_camera.py`) modelled
on a RealSense D435 (depth 848x480, 87x58 deg, 0.3-3 m with stereo noise; RGB 1280x720,
69x42 deg), about 0.6 m above the handshake and just behind the robot's base, looking at
the person. The person (`human_body.py`) sits on a stool around the partner hand; their
body is visual only, so it doesn't change the handshake physics. The control window shows
the camera view (RGB over depth, with the detection box: grey none, yellow tracking,
green offered, blue committed) beside the tactile panel; `--no-headcam` hides it.
```
..\.venv\Scripts\python handshake.py --record headcam_view.mp4 --headcam --cycles 2   # scene + head camera
```

Thumbs: the robot thumb is abducted (1.4 rad) so it rises off its palm and rests on the
back of the partner's hand; the partner thumb crosses the robot's thumb web and hooks
over the top of the robot hand toward its index knuckle.

Record a video (needs `pip install "imageio[ffmpeg]"`):
```
..\.venv\Scripts\python handshake.py --record             # -> handshake.mp4 (one handshake)
..\.venv\Scripts\python handshake.py --record --cycles 3  # until 3 handshakes are done
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
