"""Build a MuJoCo scene with the TetherIA Aero Hand Open (right) mounted on a KUKA LBR iiwa 14.

Sources (unmodified):
  - KUKA iiwa 14:  ../mujoco_menagerie/kuka_iiwa_14/iiwa14.xml
  - Aero Hand:     ../aero-hand-open/sim_rl/simulation/mujoco/right_hand.xml

Writes scene.xml next to this script. Re-run after updating either source model.
"""

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IIWA_XML = os.path.join(ROOT, "mujoco_menagerie", "kuka_iiwa_14", "iiwa14.xml")
HAND_XML = os.path.join(ROOT, "aero-hand-open", "sim_rl", "simulation", "mujoco", "right_hand.xml")
OUT_XML = os.path.join(HERE, "scene.xml")

HAND_PREFIX = "hand_"

# Hand bracket pose relative to the iiwa flange (link7 "attachment_site").
# The bracket's flat mounting face is on its -x side (centre at -0.022, 0.010, 0.009 m
# in the bracket frame) and the fingers point along +x. Rotating -90 deg about y maps
# bracket +x onto flange +z, and the offset puts the face centre on the flange.
MOUNT_QUAT = [np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0]
MOUNT_POS = [0.009, -0.010, 0.022]

# Arm home pose: hand ~0.6 m in front of the base, ~0.36 m up, fingers pointing down.
ARM_HOME = [0.0, 0.65, 0.0, -1.3, 0.0, 1.2, 0.0]
# Hand home pose, copied from the Aero Hand scene_right.xml keyframe.
HAND_HOME_QPOS = [1.57335, 0.347305, 0.352258, 1.57303, 0.400789, 0.406,
                  1.57335, 0.365219, 0.370336, 1.57336, 0.515229, 0.520112,
                  0.746438, 0.260182, -0.00155382, 1.0748]
HAND_HOME_CTRL = [0.09, 0.09, 0.09, 0.09, 0.75, 0.035, 0.1]


def absolutize_meshes(spec: mujoco.MjSpec, xml_path: str) -> None:
    """Make mesh paths absolute so they survive attaching into another spec."""
    meshdir = os.path.join(os.path.dirname(xml_path), spec.meshdir or "")
    for mesh in spec.meshes:
        if not os.path.isabs(mesh.file):
            mesh.file = os.path.normpath(os.path.join(meshdir, mesh.file))
    spec.meshdir = ""


def build() -> mujoco.MjSpec:
    iiwa = mujoco.MjSpec.from_file(IIWA_XML)
    hand = mujoco.MjSpec.from_file(HAND_XML)
    absolutize_meshes(iiwa, IIWA_XML)
    absolutize_meshes(hand, HAND_XML)

    # Drop the source keyframes; a combined one is added below.
    for spec in (iiwa, hand):
        for key in list(spec.keys):
            spec.delete(key)

    # The scene uses the arm's solver settings (2 ms implicitfast); the hand's 10 ms
    # Euler settings are dropped on attach. Match them up front to avoid conflict warnings.
    iiwa.option.timestep = 0.002
    for field in ("timestep", "integrator", "iterations", "ls_iterations"):
        setattr(hand.option, field, getattr(iiwa.option, field))

    # Mount the hand on the flange. The bracket body is offset 3 cm in the source file;
    # zero that so the frame below fully determines the mount pose.
    hand.body("tetheria_mount").pos = [0, 0, 0]
    site = iiwa.site("attachment_site")
    frame = iiwa.body("link7").add_frame(
        pos=np.asarray(site.pos) + np.asarray(MOUNT_POS), quat=MOUNT_QUAT)
    frame.attach_body(hand.body("tetheria_mount"), HAND_PREFIX, "")

    # Bracket and palm sit against the flange; don't let them collide with it.
    for b in ("tetheria_mount", "palm"):
        iiwa.add_exclude(bodyname1="link7", bodyname2=HAND_PREFIX + b)
        iiwa.add_exclude(bodyname1="link6", bodyname2=HAND_PREFIX + b)

    # Scene dressing (same look as the Menagerie iiwa scene).
    iiwa.modelname = "iiwa14_aero_hand_right"
    iiwa.option.timestep = 0.002
    iiwa.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    iiwa.stat.center = [0.4, 0.0, 0.5]
    iiwa.stat.extent = 1.2
    iiwa.visual.global_.azimuth = -140
    iiwa.visual.global_.elevation = -20
    iiwa.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    iiwa.visual.headlight.ambient = [0.3, 0.3, 0.3]
    iiwa.visual.quality.shadowsize = 8192
    iiwa.visual.global_.offwidth = 1920  # allow larger offscreen renders
    iiwa.visual.global_.offheight = 1080
    iiwa.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    iiwa.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4],
                     rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    mat = iiwa.add_material(name="groundplane", texuniform=True, texrepeat=[5, 5],
                            reflectance=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    wb = iiwa.worldbody
    wb.add_light(pos=[0, 0, 2])
    wb.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                material="groundplane")

    # Combined home keyframe: arm joints first, then hand joints; same for actuators.
    iiwa.add_key(name="home", qpos=ARM_HOME + HAND_HOME_QPOS, ctrl=ARM_HOME + HAND_HOME_CTRL)
    return iiwa


def main() -> None:
    spec = build()
    model = spec.compile()
    xml = spec.to_xml()
    with open(OUT_XML, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {OUT_XML}")
    print(f"nq={model.nq} nu={model.nu} ntendon={model.ntendon} nbody={model.nbody}")
    print("actuators:", [model.actuator(i).name for i in range(model.nu)])


if __name__ == "__main__":
    main()
