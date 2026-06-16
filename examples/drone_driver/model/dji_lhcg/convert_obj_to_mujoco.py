"""
Convert DJI drone OBJ model to MuJoCo XML format with per-material meshes and textures.

Steps:
1. Load OBJ and split by material
2. Classify body parts vs propeller-sweep (motion-blur) geometries
3. Export UV-mapped parts as OBJ (preserving UVs), non-UV parts as STL
4. Assign texture atlas (wurenji-lhcg_001.jpg) to UV-mapped parts
5. Detect 4 rotor positions via angle-based clustering of far vertices
6. Create simplified 2-blade propeller mesh
7. Generate MuJoCo XML with per-material geoms and textures
"""

import pathlib
import logging
import sys
import shutil
from dataclasses import dataclass

import numpy as np
import trimesh
import pyfqmr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SRC_DIR = pathlib.Path("/home/guojiatao/Assets/3d")
DST_DIR = pathlib.Path(__file__).parent
MESHES_DIR = DST_DIR / "meshes"

SCALE = 0.001

MUJOCO_MAX_FACES = 524288
MAX_FACES_PER_PART = 190000

BODY_MAX_SPAN_THRESHOLD = 300
CENTER_XY_THRESHOLD = 80
BODY_CENTER_ESTIMATE = np.array([-4.5, 2.5, 40.0])
BODY_MAX_DIST_FROM_CENTER = 150.0
ISOLATION_GAP_MM = 10.0

ROTOR_Z_LO = 35.0
ROTOR_Z_HI = 55.0
FAR_PERCENTILE = 98

BODY_TEXTURE = "wurenji-lhcg_001.png"


@dataclass
class MaterialInfo:
    name: str
    rgba: str
    has_uv: bool = False
    texture_file: str | None = None


def _build_spatial_grid(verts: np.ndarray, grid_size: float) -> dict:
    from collections import defaultdict
    grid = defaultdict(list)
    for i, v in enumerate(verts):
        key = (int(v[0] / grid_size), int(v[1] / grid_size), int(v[2] / grid_size))
        grid[key].append(i)
    return grid


def _min_distance_to_grid(verts: np.ndarray, ref_verts: np.ndarray, ref_grid: dict, grid_size: float, sample: int = 200) -> float:
    sample_idx = np.random.choice(len(verts), min(sample, len(verts)), replace=False)
    min_dist = float("inf")
    for i in sample_idx:
        v = verts[i]
        gx, gy, gz = int(v[0] / grid_size), int(v[1] / grid_size), int(v[2] / grid_size)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    key = (gx + dx, gy + dy, gz + dz)
                    if key in ref_grid:
                        indices = ref_grid[key]
                        dists = np.linalg.norm(ref_verts[indices] - v, axis=1)
                        d = float(np.min(dists))
                        if d < min_dist:
                            min_dist = d
    return min_dist


def load_and_classify(scene: trimesh.Scene) -> tuple[list[tuple[str, trimesh.Trimesh]], list[tuple[str, trimesh.Trimesh]]]:
    body_parts = []
    sweep_parts = []

    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        bounds = geom.bounds
        span = bounds[1] - bounds[0]
        max_span = float(max(span))
        center = geom.centroid
        is_near_center = abs(center[0]) < CENTER_XY_THRESHOLD and abs(center[1]) < CENTER_XY_THRESHOLD
        dist_from_body = float(np.linalg.norm(center - BODY_CENTER_ESTIMATE))

        if max_span > BODY_MAX_SPAN_THRESHOLD and not is_near_center:
            sweep_parts.append((name, geom))
            logger.info("  SWEEP (large+far): %s, dist=%.1fmm", name, dist_from_body)
        elif dist_from_body > BODY_MAX_DIST_FROM_CENTER:
            sweep_parts.append((name, geom))
            logger.info("  SWEEP (far): %s, dist=%.1fmm, max_span=%.1f", name, dist_from_body, max_span)
        else:
            body_parts.append((name, geom))

    core_verts_list = []
    for name, geom in body_parts:
        if len(geom.faces) > 10000:
            core_verts_list.append(geom.vertices)

    if core_verts_list:
        core_verts = np.vstack(core_verts_list)
        grid_size = ISOLATION_GAP_MM
        core_grid = _build_spatial_grid(core_verts, grid_size)

        changed = True
        while changed:
            changed = False
            all_body_verts_list = [geom.vertices for _, geom in body_parts]
            all_body_verts = np.vstack(all_body_verts_list)
            all_body_grid = _build_spatial_grid(all_body_verts, grid_size)

            isolated = []
            for name, geom in body_parts:
                if len(geom.faces) > 10000:
                    continue
                min_gap = _min_distance_to_grid(geom.vertices, all_body_verts, all_body_grid, grid_size)
                if min_gap > ISOLATION_GAP_MM:
                    isolated.append((name, geom))
                    logger.info("  ISOLATED: %s, min_gap=%.1fmm > %.1fmm threshold", name, min_gap, ISOLATION_GAP_MM)

            if isolated:
                for name, geom in isolated:
                    body_parts.remove((name, geom))
                    sweep_parts.append((name, geom))
                changed = True

    logger.info("Classified: %d body parts, %d sweep parts", len(body_parts), len(sweep_parts))
    return body_parts, sweep_parts


def extract_material_info(name: str, geom: trimesh.Trimesh) -> MaterialInfo:
    mat_name = name.replace("wurenji_lhcg_mat_", "mat_")
    diffuse = None
    has_uv = False
    texture_file = None

    visual = geom.visual
    if visual is not None and hasattr(visual, "material"):
        mat = visual.material
        if hasattr(mat, "diffuse"):
            try:
                d = mat.diffuse
                r, g, b = int(d[0]) / 255.0, int(d[1]) / 255.0, int(d[2]) / 255.0
                diffuse = f"{r:.4f} {g:.4f} {b:.4f} 1"
            except Exception:
                pass

    if visual is not None:
        try:
            uv = visual.uv
            if uv is not None and len(uv) > 0 and uv.shape[0] == len(geom.vertices):
                has_uv = True
                texture_file = BODY_TEXTURE
        except Exception:
            pass

    if diffuse is None:
        diffuse = "0.8 0.8 0.8 1"

    return MaterialInfo(name=mat_name, rgba=diffuse, has_uv=has_uv, texture_file=texture_file)


def find_rotor_positions(merged: trimesh.Trimesh, body_center: np.ndarray) -> list[np.ndarray]:
    verts = merged.vertices

    z_mask = (verts[:, 2] > ROTOR_Z_LO) & (verts[:, 2] < ROTOR_Z_HI)
    rotor_z_verts = verts[z_mask]

    xy = rotor_z_verts[:, :2]
    center_xy = body_center[:2]
    distances = np.linalg.norm(xy - center_xy, axis=1)
    threshold = np.percentile(distances, FAR_PERCENTILE)
    far_mask = distances > threshold
    far_verts = rotor_z_verts[far_mask]

    logger.info("Far vertices at rotor height: %d (threshold=%.1f)", len(far_verts), threshold)

    target_angles = [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]
    labels = ["FR", "FL", "BL", "BR"]
    rotor_positions = []

    angles = np.arctan2(far_verts[:, 1] - body_center[1], far_verts[:, 0] - body_center[0])

    for label, ta in zip(labels, target_angles):
        angle_diff = np.abs(np.arctan2(np.sin(angles - ta), np.cos(angles - ta)))
        near_mask = angle_diff < np.pi / 4
        if not np.any(near_mask):
            logger.warning("No points found for rotor %s", label)
            continue
        cluster = far_verts[near_mask]
        cluster_center = cluster.mean(axis=0)
        rotor_positions.append(cluster_center)
        logger.info("Rotor %s: center=[%.1f, %.1f, %.1f], n=%d", label, cluster_center[0], cluster_center[1], cluster_center[2], len(cluster))

    return rotor_positions


def create_blade_mesh(radius_m: float, n_blades: int = 2) -> trimesh.Trimesh:
    blade_length = radius_m * 0.9
    blade_width = radius_m * 0.12
    blade_thickness = radius_m * 0.015

    blade = trimesh.creation.box(extents=[blade_length, blade_width, blade_thickness])
    blade.apply_translation([blade_length / 2, 0, 0])

    blades = []
    for i in range(n_blades):
        angle = i * np.pi / n_blades
        rotated = blade.copy()
        rotation = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        rotated.apply_transform(rotation)
        blades.append(rotated)

    hub = trimesh.creation.cylinder(radius=blade_width * 0.5, height=blade_thickness * 2)
    blades.append(hub)

    return trimesh.util.concatenate(blades)


def center_and_scale_mesh(mesh: trimesh.Trimesh, center: np.ndarray, scale: float) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    return mesh


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh

    logger.info("Simplifying mesh: %d -> %d faces", len(mesh.faces), target_faces)
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(target_count=target_faces, aggressiveness=7, preserve_border=False, verbose=False)
    vertices, faces, _ = mesh_simplifier.getMesh()

    simplified = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    logger.info("Simplified: %d vertices, %d faces", len(simplified.vertices), len(simplified.faces))
    return simplified


def simplify_part(geom: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(geom.faces) <= target_faces:
        return geom
    try:
        s = pyfqmr.Simplify()
        s.setMesh(geom.vertices, geom.faces)
        s.simplify_mesh(target_count=target_faces, aggressiveness=7, preserve_border=True, verbose=False)
        verts, faces, _ = s.getMesh()
        return trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    except Exception:
        return geom


def export_obj_with_texture(mesh: trimesh.Trimesh, obj_path: pathlib.Path, texture_file: str) -> None:
    mtl_filename = obj_path.stem + ".mtl"
    mtl_path = obj_path.parent / mtl_filename

    mtl_content = (
        f"newmtl {obj_path.stem}_mat\n"
        f"Ka 0.2 0.2 0.2\n"
        f"Kd 0.8 0.8 0.8\n"
        f"Ks 0.0 0.0 0.0\n"
        f"map_Kd {texture_file}\n"
    )
    mtl_path.write_text(mtl_content, encoding="utf-8")

    mesh_obj = mesh.copy()
    if mesh_obj.visual is None or not hasattr(mesh_obj.visual, "uv"):
        mesh_obj.visual = trimesh.visual.ColorVisuals(mesh_obj)

    export_data = trimesh.exchange.obj.export_obj(
        mesh_obj,
        include_texture=True,
        mtl_name=mtl_filename,
    )
    obj_path.write_text(export_data, encoding="utf-8")


def generate_xml(
    body_mass: float,
    body_inertia: np.ndarray,
    material_meshes: list[tuple[str, MaterialInfo]],
    texture_entries: list[tuple[str, str]],
) -> str:
    ix, iy, iz = body_inertia[0, 0], body_inertia[1, 1], body_inertia[2, 2]

    xml_lines = [
        '<mujoco model="dji_lhcg">',
        '  <compiler angle="radian" meshdir="meshes/" texturedir="meshes/" balanceinertia="true"/>',
        '',
        '  <asset>',
        '    <texture name="texplane" type="2d" builtin="flat" rgb1="0.95 0.95 0.95" width="5120" height="5120"/>',
        '    <material name="MatPlane" texture="texplane" texrepeat="1 1" texuniform="true" reflectance="0"/>',
    ]

    for tex_name, tex_file in texture_entries:
        xml_lines.append(f'    <texture name="{tex_name}" type="2d" file="{tex_file}"/>')

    for mesh_name, mat_info in material_meshes:
        ext = "obj" if mat_info.has_uv else "stl"
        xml_lines.append(f'    <mesh name="{mesh_name}" file="{mesh_name}.{ext}"/>')

    for mesh_name, mat_info in material_meshes:
        if mat_info.has_uv and mat_info.texture_file:
            tex_name = mat_info.name + "_tex"
            xml_lines.append(f'    <material name="{mat_info.name}_mat" texture="{tex_name}" rgba="{mat_info.rgba}" texuniform="false"/>')
        else:
            xml_lines.append(f'    <material name="{mat_info.name}_mat" rgba="{mat_info.rgba}"/>')

    xml_lines.extend([
        '  </asset>',
        '',
        '  <default>',
        '    <light castshadow="false"/>',
        '    <geom rgba="0.8 0.8 0.8 1"/>',
        '',
        '    <default class="dji_lhcg">',
        '      <joint limited="true" range="-1.5 1.5" damping="0.1" armature="0.001"/>',
        '      <geom contype="1" conaffinity="1" condim="4" group="1" margin="0.001"/>',
        '',
        '      <default class="main_body">',
        '        <geom type="mesh" contype="0" conaffinity="0" group="1" mass="0"/>',
        '      </default>',
        '',
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
        '      <body name="camera" pos="-0.5 0 0" euler="0 0 -1.570796">',
        '        <camera name="indi_eye_camera" pos="0 0 0" fovy="90"/>',
        '      </body>',
        '',
        '      <body name="Drone" childclass="dji_lhcg" pos="0 0 0">',
        '        <site name="drone_body_center_site" pos="0 0 0" group="3" size="0.01" rgba="1 1 0 0.8"/>',
        '        <site name="drone_up_site" pos="0 0 0.1" size="0.008" group="3" rgba="0 0.7 1 0.8"/>',
        '',
        f'        <inertial pos="0 0 0" mass="{body_mass:.4f}" diaginertia="{ix:.6f} {iy:.6f} {iz:.6f}"/>',
        '',
    ])

    for i, (mesh_name, mat_info) in enumerate(material_meshes):
        xml_lines.append(
            f'        <geom name="body_{mesh_name}" class="main_body" mesh="{mesh_name}" material="{mat_info.name}_mat" rgba="{mat_info.rgba}"/>'
        )

    xml_lines.append(
        '        <geom name="body_col" class="collision" type="cylinder" size="0.15 0.04"/>'
    )

    xml_lines.extend([
        '      </body>',
        '    </body>',
        '  </worldbody>',
        '',
        '  <sensor>',
        '    <gyro name="body_gyro" site="drone_body_center_site"/>',
        '    <accelerometer name="body_linacc" site="drone_body_center_site"/>',
        '    <framequat name="body_quat" objtype="site" objname="drone_body_center_site"/>',
        '  </sensor>',
        '</mujoco>',
        '',
    ])

    return "\n".join(xml_lines)


def main() -> None:
    MESHES_DIR.mkdir(parents=True, exist_ok=True)

    for f in MESHES_DIR.glob("*.stl"):
        f.unlink()
        logger.info("Removed old mesh: %s", f.name)
    for f in MESHES_DIR.glob("*.obj"):
        if f.name != "obj.obj":
            f.unlink()
            logger.info("Removed old mesh: %s", f.name)
    for f in MESHES_DIR.glob("mat_*.mtl"):
        f.unlink()
        logger.info("Removed old mtl: %s", f.name)

    obj_path = SRC_DIR / "obj.obj"
    if not obj_path.exists():
        logger.error("OBJ file not found: %s", obj_path)
        sys.exit(1)

    logger.info("Loading OBJ file: %s", obj_path)
    scene = trimesh.load(str(obj_path), split_object=True, group_material=True)
    logger.info("Loaded scene with %d geometries", len(scene.geometry))

    body_parts, sweep_parts = load_and_classify(scene)

    if not body_parts:
        logger.error("No body parts found!")
        sys.exit(1)

    body_center = BODY_CENTER_ESTIMATE.copy()
    logger.info("Using body center: [%.1f, %.1f, %.1f]", body_center[0], body_center[1], body_center[2])

    material_meshes: list[tuple[str, MaterialInfo]] = []
    texture_entries: list[tuple[str, str]] = []
    total_faces = 0

    for name, geom in body_parts:
        mat_info = extract_material_info(name, geom)
        mesh_name = mat_info.name

        scaled = center_and_scale_mesh(geom, body_center, SCALE)

        if len(scaled.faces) > MAX_FACES_PER_PART:
            target = max(int(len(scaled.faces) * 0.5), 1000)
            logger.info("Simplifying %s: %d -> %d faces (UV will be lost)", mesh_name, len(scaled.faces), target)
            scaled = simplify_part(scaled, target)
            mat_info.has_uv = False
            mat_info.texture_file = None

        if len(scaled.faces) == 0:
            logger.warning("Skipping empty mesh: %s", mesh_name)
            continue

        if mat_info.has_uv and mat_info.texture_file:
            obj_file = MESHES_DIR / f"{mesh_name}.obj"
            export_obj_with_texture(scaled, obj_file, mat_info.texture_file)
            logger.info("Exported %s: %d faces -> OBJ with texture %s", mesh_name, len(scaled.faces), mat_info.texture_file)

            tex_name = f"{mat_info.name}_tex"
            if tex_name not in [t[0] for t in texture_entries]:
                texture_entries.append((tex_name, mat_info.texture_file))
        else:
            mesh_path = MESHES_DIR / f"{mesh_name}.stl"
            scaled.export(str(mesh_path), file_type="stl")
            logger.info("Exported %s: %d faces -> STL (rgba=%s)", mesh_name, len(scaled.faces), mat_info.rgba)

        material_meshes.append((mesh_name, mat_info))
        total_faces += len(scaled.faces)

    logger.info("Total faces across all material meshes: %d", total_faces)

    body_mass = 1.5
    body_inertia = np.diag([0.0347563, 0.0458929, 0.0806492])

    logger.info("Body mass: %.4f kg", body_mass)
    logger.info("Body inertia diag: [%.6f, %.6f, %.6f]", body_inertia[0, 0], body_inertia[1, 1], body_inertia[2, 2])

    xml_content = generate_xml(
        body_mass=body_mass,
        body_inertia=body_inertia,
        material_meshes=material_meshes,
        texture_entries=texture_entries,
    )

    xml_path = DST_DIR / "dji_lhcg.xml"
    logger.info("Writing MuJoCo XML to %s", xml_path)
    xml_path.write_text(xml_content, encoding="utf-8")

    texture_files = [
        ("wurenji-lhcg_001.jpg", "wurenji-lhcg_001.png"),
        ("wurenji-lhcg_002.jpg", "wurenji-lhcg_002.png"),
        ("wurenji-lhcg_003.tif", "wurenji-lhcg_003.png"),
        ("wurenji-lhcg_004.jpg", "wurenji-lhcg_004.png"),
        ("wurenji-lhcg_005.jpg", "wurenji-lhcg_005.png"),
        ("wurenji-lhcg_006.jpg", "wurenji-lhcg_006.png"),
    ]
    from PIL import Image as PILImage
    for src_name, dst_name in texture_files:
        src = SRC_DIR / src_name
        dst = MESHES_DIR / dst_name
        if src.exists() and not dst.exists():
            img = PILImage.open(str(src))
            img.save(str(dst), "PNG")
            logger.info("Converted texture: %s -> %s", src_name, dst_name)

    mtl_src = SRC_DIR / "obj.mtl"
    if mtl_src.exists():
        shutil.copy2(str(mtl_src), str(MESHES_DIR / "obj.mtl"))
        logger.info("Copied MTL file")

    logger.info("Conversion complete!")
    logger.info("Output directory: %s", DST_DIR)
    mesh_files = sorted(set(p.name for p in list(MESHES_DIR.glob("*.stl")) + list(MESHES_DIR.glob("*.obj")) if p.name != "obj.obj"))
    logger.info("Mesh files: %s", mesh_files)
    tex_files = sorted(set(p.name for p in list(MESHES_DIR.glob("*.jpg")) + list(MESHES_DIR.glob("*.tif"))))
    logger.info("Texture files: %s", tex_files)


if __name__ == "__main__":
    main()
