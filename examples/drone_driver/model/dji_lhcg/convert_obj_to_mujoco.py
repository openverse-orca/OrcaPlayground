"""
Convert DJI drone OBJ model to MuJoCo XML format.

Strategy: Load OBJ with group_material=False (preserves 372 object positions),
then manually group by material and export each group as a separate STL.
Only simplify groups exceeding 190000 faces. This preserves far more detail
than merging everything into one mesh.

Rotor strategy: Export blade-only meshes (9268-face objects) with radial scaling
to fill the disc area, so rotation is visually obvious (like x2's tri_blade_propeller).
"""

import math
import pathlib
import logging
import sys
import shutil

import numpy as np
import trimesh
from fast_simplification import simplify as quadric_simplify

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SRC_DIR = pathlib.Path("/home/guojiatao/Assets/3d ")
DST_DIR = pathlib.Path(__file__).parent
MESHES_DIR = DST_DIR / "meshes"

SCALE = 0.001
MESH_SCALE = 10
BODY_CENTER = np.array([-4.5, 2.5, 40.0])
BODY_MAX_SPAN = 300
CENTER_XY_THRESH = 80
BODY_MAX_DIST = 150.0
MAX_FACES = 190000

BLADE_FACE_COUNT = 9268
DISC_FACE_COUNT = 2412
MOTOR_SHELL_SMALL = 6026
MOTOR_SHELL_LARGE = 17304

ROTOR_ARM_MAP = {"FR": "FR", "RR": "BR", "RL": "BL", "FL": "FL"}


def classify_body(scene: trimesh.Scene) -> list[tuple[str, trimesh.Trimesh, str, str]]:
    results = []
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        center = geom.centroid
        dist = float(np.linalg.norm(center - BODY_CENTER))
        span = geom.bounds[1] - geom.bounds[0]
        max_span = float(max(span))
        is_near = abs(center[0]) < CENTER_XY_THRESH and abs(center[1]) < CENTER_XY_THRESH
        if (max_span > BODY_MAX_SPAN and not is_near) or dist > BODY_MAX_DIST:
            continue

        mat_name = "unknown"
        rgba = "0.8 0.8 0.8 1"
        visual = geom.visual
        if visual is not None and hasattr(visual, "material"):
            mat = visual.material
            if hasattr(mat, "name") and mat.name:
                mat_name = mat.name
            if hasattr(mat, "diffuse"):
                try:
                    d = mat.diffuse
                    rgba = f"{int(d[0])/255:.4f} {int(d[1])/255:.4f} {int(d[2])/255:.4f} 1"
                except Exception:
                    pass

        n_faces = len(geom.faces)
        if "mat_012" in mat_name and n_faces in (BLADE_FACE_COUNT, DISC_FACE_COUNT, MOTOR_SHELL_SMALL, MOTOR_SHELL_LARGE):
            continue

        short = mat_name.replace("wurenji_lhcg_mat_", "mat_")
        if short in ("mat_010", "mat_011", "mat_013", "mat_014", "mat_015", "mat_016", "mat_032"):
            continue

        results.append((name, geom, mat_name, rgba))

    logger.info("Body objects: %d", len(results))
    return results


def group_by_material(objects: list[tuple[str, trimesh.Trimesh, str, str]]) -> dict[str, tuple[list[trimesh.Trimesh], str]]:
    groups: dict[str, tuple[list[trimesh.Trimesh], str]] = {}
    for _, geom, mat_name, rgba in objects:
        short = mat_name.replace("wurenji_lhcg_mat_", "mat_")
        if short not in groups:
            groups[short] = ([], rgba)
        groups[short][0].append(geom)
    return groups


def simplify_mesh(verts: np.ndarray, faces: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    reduction = 1.0 - (target / len(faces))
    reduction = max(0.1, min(0.95, reduction))
    v1, f1 = quadric_simplify(verts, faces, target_reduction=reduction)
    if len(f1) > MAX_FACES:
        r2 = 1.0 - (target / len(f1))
        v2, f2 = quadric_simplify(v1, f1, target_reduction=max(0.1, min(0.9, r2)))
        return v2, f2
    return v1, f1


def export_rotors(scene: trimesh.Scene) -> list[tuple[str, np.ndarray]]:
    rotor_objects = []
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        visual = geom.visual
        mat_name = "unknown"
        if visual is not None and hasattr(visual, 'material'):
            mat = visual.material
            if hasattr(mat, 'name') and mat.name:
                mat_name = mat.name
        short = mat_name.replace("wurenji_lhcg_mat_", "mat_")
        if short not in ("mat_010", "mat_011", "mat_012", "mat_013", "mat_014", "mat_015", "mat_016", "mat_032"):
            continue
        center = geom.centroid
        dist = float(np.linalg.norm(center - BODY_CENTER))
        span = geom.bounds[1] - geom.bounds[0]
        max_span = float(max(span))
        is_near = abs(center[0]) < 80 and abs(center[1]) < 80
        if (max_span > 300 and not is_near) or dist > 150:
            continue
        n_faces = len(geom.faces)
        if short == "mat_012" and n_faces == BLADE_FACE_COUNT:
            continue
        rotor_objects.append((name, geom, center, n_faces))

    arm_groups: dict[str, list] = {"FR": [], "RR": [], "RL": [], "FL": []}
    for name, geom, center, n_faces in rotor_objects:
        x, y = center[0] - BODY_CENTER[0], center[1] - BODY_CENTER[1]
        if x > 0 and y < 0:
            arm_groups["FR"].append((name, geom, center, n_faces))
        elif x > 0 and y > 0:
            arm_groups["RR"].append((name, geom, center, n_faces))
        elif x < 0 and y > 0:
            arm_groups["RL"].append((name, geom, center, n_faces))
        else:
            arm_groups["FL"].append((name, geom, center, n_faces))

    rotor_entries = []
    for arm_name, objects in arm_groups.items():
        if not objects:
            continue

        disc_z = [c[2] for _, _, c, nf in objects if nf == DISC_FACE_COUNT]
        if len(disc_z) < 2:
            continue
        z_mid = (min(disc_z) + max(disc_z)) / 2.0

        upper_parts = []
        lower_parts = []
        seen = set()
        for name, geom, center, n_faces in objects:
            key = (round(center[0], 1), round(center[1], 1), round(center[2], 1), n_faces)
            if key in seen:
                continue
            seen.add(key)
            if center[2] >= z_mid:
                upper_parts.append(geom)
            else:
                lower_parts.append(geom)

        for suffix, parts in [("upper", upper_parts), ("lower", lower_parts)]:
            if not parts:
                continue
            merged = trimesh.util.concatenate(parts)
            disc_centroid = merged.centroid.copy()

            verts = merged.vertices.copy()
            verts -= disc_centroid
            verts *= SCALE

            body_pos = (disc_centroid - BODY_CENTER) * SCALE

            mesh = trimesh.Trimesh(vertices=verts, faces=merged.faces, process=True)
            src_key = f"rotor_{arm_name}_{suffix}"
            stl_path = MESHES_DIR / f"{src_key}.stl"
            mesh.export(str(stl_path), file_type="stl")

            dst_name = ROTOR_ARM_MAP.get(arm_name, arm_name)
            dst_key = f"rotor_{dst_name}_{suffix}"
            if dst_key != src_key:
                shutil.copy2(str(stl_path), str(MESHES_DIR / f"{dst_key}.stl"))

            rotor_entries.append((dst_key, body_pos))
            logger.info("  %s: pos=(%.4f, %.4f, %.4f)",
                        dst_key, body_pos[0], body_pos[1], body_pos[2])

    return rotor_entries


def generate_xml(
    mesh_entries: list[tuple[str, str]],
    rotor_entries: list[tuple[str, np.ndarray]],
    body_mass: float,
    body_inertia: np.ndarray,
) -> str:
    ix, iy, iz = body_inertia[0, 0], body_inertia[1, 1], body_inertia[2, 2]
    lines = [
        '<mujoco model="dji_lhcg">',
        '  <compiler angle="radian" meshdir="meshes/" texturedir="meshes/" balanceinertia="true"/>',
        '',
        '  <asset>',
        '    <texture name="texplane" type="2d" builtin="flat" rgb1="0.95 0.95 0.95" width="5120" height="5120"/>',
        '    <material name="MatPlane" texture="texplane" texrepeat="1 1" texuniform="true" reflectance="0"/>',
    ]
    for mesh_name, _ in mesh_entries:
        lines.append(f'    <mesh name="{mesh_name}" file="{mesh_name}.stl" scale="10 10 10"/>')
    for rotor_key, _ in rotor_entries:
        lines.append(f'    <mesh name="{rotor_key}" file="{rotor_key}.stl" scale="10 10 10"/>')
    lines.extend([
        '  </asset>',
        '',
        '  <default>',
        '    <light castshadow="false"/>',
        '    <geom rgba="0.8 0.8 0.8 1"/>',
        '    <default class="dji_lhcg">',
        '      <joint limited="true" range="-1.5 1.5" damping="0.1" armature="0.001"/>',
        '      <geom contype="1" conaffinity="1" condim="4" group="1" margin="0.001"/>',
        '      <default class="main_body">',
        '        <geom type="mesh" contype="0" conaffinity="0" group="1" mass="0"/>',
        '      </default>',
        '      <default class="rotor_joint">',
        '        <joint type="hinge" limited="false" damping="0.02" armature="0.0005"/>',
        '      </default>',
        '      <default class="rotor_geom">',
        '        <geom type="mesh" contype="0" conaffinity="0" group="1" mass="0" rgba="0.0667 0.0667 0.0667 1"/>',
        '      </default>',
        '      <default class="collision">',
        '        <geom contype="1" conaffinity="1" group="4" rgba="1 0.3 1 0.5" friction="2 0.0005 0.00001"/>',
        '      </default>',
        '    </default>',
        '  </default>',
        '',
        '  <visual>',
        '    <headlight ambient="0.5 0.5 0.5"/>',
        '  </visual>',
        '',
        '  <worldbody>',
        '    <light pos="0 0 1000" directional="false" diffuse="0.3 0.3 0.3" specular="0.3 0.3 0.3"/>',
        '',
        '    <body name="drone_frame" pos="0 0 0">',
        '      <joint name="drone_free" type="free"/>',
        '      <body name="camera" pos="-5 0 0" euler="0 0 -1.570796">',
        '        <camera name="indi_eye_camera" pos="0 0 0" fovy="90"/>',
        '      </body>',
        '',
        '      <body name="Drone" childclass="dji_lhcg" pos="0 0 0">',
        '        <site name="imu" pos="0 0 0" size="0.01" rgba="0.5 0.5 0.5 1" group="0"/>',
        '        <site name="drone_body_center_site" pos="0 0 0" group="3" size="0.05" rgba="1 1 0 0.8"/>',
        '        <site name="drone_forward_site" pos="0 1.2 0" size="0.03" group="3" rgba="0 1 0 0.8"/>',
        '        <site name="drone_up_site" pos="0 0 1.0" size="0.04" group="3" rgba="0 0.7 1 0.8"/>',
        f'        <inertial pos="0 0 0" mass="{body_mass:.4f}" diaginertia="{ix:.6f} {iy:.6f} {iz:.6f}"/>',
        '',
    ])
    for mesh_name, rgba in mesh_entries:
        lines.append(f'        <geom name="body_{mesh_name}" class="main_body" mesh="{mesh_name}" rgba="{rgba}"/>')
    lines.append('        <geom name="body_col" class="collision" type="cylinder" size="1.5 0.4"/>')

    rotor_specs = [
        ("FL_upper_blade", "FL_joint", "0 0 1", "rotor_fl_site", "rotor_FL_upper"),
        ("FL_lower_blade", "FL2_joint", "0 0 -1", "rotor_fl2_site", "rotor_FL_lower"),
        ("FR_upper_blade", "FR_joint", "0 0 1", "rotor_fr_site", "rotor_FR_upper"),
        ("FR_lower_blade", "FR2_joint", "0 0 -1", "rotor_fr2_site", "rotor_FR_lower"),
        ("BL_upper_blade", "BL_joint", "0 0 1", "rotor_bl_site", "rotor_BL_upper"),
        ("BL_lower_blade", "BL2_joint", "0 0 -1", "rotor_bl2_site", "rotor_BL_lower"),
        ("BR_upper_blade", "BR_joint", "0 0 1", "rotor_br_site", "rotor_BR_upper"),
        ("BR_lower_blade", "BR2_joint", "0 0 -1", "rotor_br2_site", "rotor_BR_lower"),
    ]

    rotor_pos_map = {key: pos for key, pos in rotor_entries}
    for body_name, joint_name, axis, site_name, mesh_key in rotor_specs:
        pos = rotor_pos_map.get(mesh_key)
        if pos is None:
            logger.warning("Missing rotor position for %s", mesh_key)
            continue
        px, py, pz = f"{pos[0]*MESH_SCALE:.3f}", f"{pos[1]*MESH_SCALE:.3f}", f"{pos[2]*MESH_SCALE:.3f}"
        site_rgba = "1 0.5 0 0.8" if "2_" not in joint_name else "1 0.3 0 0.8"
        lines.extend([
            '',
            f'        <body name="{body_name}" pos="{px} {py} {pz}">',
            f'          <inertial pos="0 0 0" mass="0.03" diaginertia="3e-4 1.5e-4 1.5e-4"/>',
            f'          <joint class="rotor_joint" name="{joint_name}" axis="{axis}"/>',
            f'          <site name="{site_name}" pos="0 0 0" size="0.05" rgba="{site_rgba}"/>',
            f'          <geom class="rotor_geom" mesh="{mesh_key}"/>',
            '        </body>',
        ])

    lines.extend([
        '      </body>',
        '    </body>',
        '  </worldbody>',
        '',
        '  <sensor>',
        '    <gyro name="body_gyro" site="imu"/>',
        '    <accelerometer name="body_linacc" site="imu"/>',
        '    <framequat name="body_quat" objtype="site" objname="imu"/>',
        '  </sensor>',
        '</mujoco>',
        '',
    ])
    return "\n".join(lines)


def main() -> None:
    MESHES_DIR.mkdir(parents=True, exist_ok=True)

    for f in MESHES_DIR.glob("mat_*.stl"):
        f.unlink()
    for f in MESHES_DIR.glob("dji_body*.stl"):
        f.unlink()
    for f in MESHES_DIR.glob("rotor_*.stl"):
        f.unlink()

    obj_path = SRC_DIR / "obj.obj"
    if not obj_path.exists():
        logger.error("OBJ not found: %s", obj_path)
        sys.exit(1)

    logger.info("Loading OBJ (group_material=False)...")
    scene = trimesh.load(str(obj_path), split_object=True, group_material=False)
    logger.info("Loaded: %d geometries", len(scene.geometry))

    body_objects = classify_body(scene)
    if not body_objects:
        logger.error("No body parts found!")
        sys.exit(1)

    groups = group_by_material(body_objects)
    logger.info("Material groups: %d", len(groups))

    mesh_entries = []
    for mat_name, (parts, rgba) in sorted(groups.items()):
        merged = trimesh.util.concatenate(parts)
        n_faces = len(merged.faces)
        merged.vertices -= BODY_CENTER
        merged.vertices *= SCALE

        if n_faces > MAX_FACES:
            target = MAX_FACES - 10000
            v, f = simplify_mesh(merged.vertices, merged.faces, target)
            simplified = trimesh.Trimesh(vertices=v, faces=f, process=True)
            logger.info("  %s: %d -> %df", mat_name, n_faces, len(simplified.faces))
        else:
            simplified = merged
            logger.info("  %s: %df (kept)", mat_name, n_faces)

        stl_path = MESHES_DIR / f"{mat_name}.stl"
        simplified.export(str(stl_path), file_type="stl")
        mesh_entries.append((mat_name, rgba))

    total = 0
    for name, _ in mesh_entries:
        p = MESHES_DIR / f"{name}.stl"
        m = trimesh.load(str(p))
        total += len(m.faces)
    logger.info("Total faces: %d across %d meshes", total, len(mesh_entries))

    logger.info("Exporting rotors (blade-only with radial scaling)...")
    rotor_entries = export_rotors(scene)

    body_mass = 15.0
    body_inertia = np.diag([6.2508, 3.7158, 2.8518])

    xml_content = generate_xml(mesh_entries, rotor_entries, body_mass, body_inertia)
    xml_path = DST_DIR / "dji_lhcg.xml"
    xml_path.write_text(xml_content, encoding="utf-8")
    logger.info("XML: %s", xml_path)

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    logger.info("MuJoCo OK: nbody=%d, ngeom=%d, nmesh=%d", model.nbody, model.ngeom, model.nmesh)
    logger.info("Done!")


if __name__ == "__main__":
    main()
