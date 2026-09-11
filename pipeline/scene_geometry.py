"""
Step 1a: POSTECH scene geometry helpers.

Extracts, from the loaded Sionna RT scene:
  - per-building 2D footprint (convex hull in the x,y plane) + height,
    used to EXCLUDE positions that fall inside a building.
  - a ground-height interpolator z_ground(x, y) built from the terrain
    mesh vertices, for whatever shape the terrain 'Plane' happens to
    be. It has changed across map re-uploads: originally a single
    TILTED quad (z varying -8.4..+8.4 m, NOT aligned with the
    buildings' z=0 base) and ROTATED relative to world x/y axes; as of
    the latest upload it is a flat (z=0) AXIS-ALIGNED rectangle
    x=[-215,245], y=[-245,215], consistent with the buildings' own
    z=0 base. build_ground_height_interpolator() fits a plane via
    least squares so it keeps working either way, but don't assume a
    specific shape -- re-verify with this module's __main__ block
    after any future map change.

Verified manually (see verify_scene_geometry() at the bottom):
  - 35 total shapes in POSTECH.xml = 1 terrain plane + 17 buildings x 2
    parts each (an "itu_concrete" body from z=0 to the building height,
    and an "itu_brick" roof cap of zero z-extent at that height).
  - Building name <-> scene object id mapping is recovered by zipping
    the mesh filename order in the XML with scene.objects insertion
    order (both follow the same shape order); LG연구동 maps to
    elm__30 (concrete body) / elm__31 (roof cap), footprint
    x in [-144.8, -67.3], y in [-59.0, 17.8], height 13.5 m -- which is
    consistent with the user's TX at (-70, -25, 15): inside that
    footprint, 1.5 m above the roof.
"""
import os
# This is a shared lab server; GPU 0 is routinely full from other users'
# real jobs (observed 47.7/49GB in use by another lab member's training
# run -> Dr.Jit/OptiX crashed with an out-of-memory JIT compile error).
# GPU 1 and 2 were essentially idle (~867MB, just other users' notebook
# kernels touching them at CUDA init). Pin this project to GPU 2 so ray
# tracing runs on hardware, not CPU, without fighting GPU 0's owner.
# If GPU 2 ever becomes busy too, check `nvidia-smi` and change this.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import re
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

POSTECH_SCENE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "POSTECH_scene", "POSTECH.xml")


def _mesh_filename_order(scene_xml_path):
    """Return the ordered list of mesh filenames as they appear in the
    Mitsuba XML file (this is the same order Sionna assigns object ids in)."""
    with open(scene_xml_path, "r", encoding="utf-8") as f:
        xml = f.read()
    return re.findall(r'<string name="filename" value="meshes/([^"]+)\.ply"/>', xml)


def load_scene_with_geometry(scene_xml_path=POSTECH_SCENE, merge_shapes=False):
    """Load the scene and return (scene, name_to_meshfile) where
    name_to_meshfile maps Sionna object names (e.g. 'elm__30') to the
    original mesh filename (e.g. 'LG연구동-itu_concrete')."""
    from sionna.rt import load_scene

    scene = load_scene(scene_xml_path, merge_shapes=merge_shapes)
    mesh_order = _mesh_filename_order(scene_xml_path)
    obj_names = list(scene.objects.keys())
    if len(mesh_order) != len(obj_names):
        raise RuntimeError(
            f"Shape count mismatch: XML has {len(mesh_order)} meshes, "
            f"scene has {len(obj_names)} objects -- name<->mesh mapping "
            f"would be wrong, needs re-checking."
        )
    name_to_meshfile = dict(zip(obj_names, mesh_order))
    return scene, name_to_meshfile


def extract_building_footprints(scene, name_to_meshfile):
    """Group the two shapes per building (itu_concrete body + itu_brick
    roof) by building name, and compute each building's 2D convex-hull
    footprint (in x,y) and height (max z of the body).

    Returns: list of dicts: {"name": str, "hull_xy": (H,2) ndarray,
                              "height": float, "obj_names": [str, ...]}
    """
    buildings = {}
    for obj_name, meshfile in name_to_meshfile.items():
        m = re.match(r"^(.*)-itu_(concrete|brick)$", meshfile)
        if not m:
            continue  # terrain (Plane) has no such suffix
        bldg_name, part = m.group(1), m.group(2)
        buildings.setdefault(bldg_name, {"obj_names": [], "verts": []})
        obj = scene.objects[obj_name]
        buildings[bldg_name]["obj_names"].append(obj_name)
        buf = np.array(obj.mi_mesh.vertex_positions_buffer(), dtype=np.float64)
        verts = buf.reshape(-1, 3)
        buildings[bldg_name]["verts"].append(verts)

    footprints = []
    for name, data in buildings.items():
        verts = np.concatenate(data["verts"], axis=0)
        xy = verts[:, :2]
        height = float(verts[:, 2].max())
        hull = ConvexHull(xy)
        hull_xy = xy[hull.vertices]
        footprints.append({
            "name": name,
            "hull_xy": hull_xy,
            "height": height,
            "obj_names": data["obj_names"],
        })
    return footprints


def points_inside_any_building(xy, footprints, margin=0.0):
    """xy: (N,2) array of candidate (x,y) points.
    margin: buffer distance (m) to additionally exclude near building walls
            (approximated by scaling the hull outward from its centroid;
            fine for a first pass, not exact for very concave buildings).
    Returns boolean mask, True = inside (i.e. should be EXCLUDED)."""
    inside = np.zeros(len(xy), dtype=bool)
    for fp in footprints:
        hull_xy = fp["hull_xy"]
        if margin > 0:
            centroid = hull_xy.mean(axis=0, keepdims=True)
            hull_xy = centroid + (hull_xy - centroid) * (1.0 + margin / np.maximum(
                np.linalg.norm(hull_xy - centroid, axis=1, keepdims=True).mean(), 1e-6))
        path = Path(hull_xy)
        inside |= path.contains_points(xy)
    return inside


def build_ground_height_interpolator(scene, name_to_meshfile, terrain_meshfile="Plane"):
    """Return a callable z_ground(xy) -> z for the terrain.

    NOTE: the terrain mesh ('Plane.ply') turned out to have only 4
    vertices -- it is a single flat (but tilted) quad, not a detailed
    DEM. So we fit an exact least-squares plane z = a*x + b*y + c
    through its vertices (with 4 coplanar points this reproduces them
    exactly) rather than doing nearest-vertex / griddata interpolation.
    """
    terrain_name = next(
        n for n, mf in name_to_meshfile.items() if mf == terrain_meshfile
    )
    obj = scene.objects[terrain_name]
    buf = np.array(obj.mi_mesh.vertex_positions_buffer(), dtype=np.float64)
    verts = buf.reshape(-1, 3)

    A = np.column_stack([verts[:, 0], verts[:, 1], np.ones(len(verts))])
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, verts[:, 2], rcond=None)
    a, b, c = coeffs
    fit_err = np.abs(A @ coeffs - verts[:, 2]).max()
    if fit_err > 1e-3:
        raise RuntimeError(
            f"Terrain is not planar (max fit residual {fit_err:.4f} m); "
            f"the plane-fit ground model is invalid, need real interpolation."
        )

    def z_ground(xy):
        xy = np.asarray(xy)
        return a * xy[:, 0] + b * xy[:, 1] + c

    return z_ground, verts


if __name__ == "__main__":
    scene, name_to_meshfile = load_scene_with_geometry()
    footprints = extract_building_footprints(scene, name_to_meshfile)
    print(f"Found {len(footprints)} buildings:")
    for fp in footprints:
        xy = fp["hull_xy"]
        print(f"  {fp['name']:35s} height={fp['height']:6.2f}  "
              f"x=[{xy[:,0].min():8.1f},{xy[:,0].max():8.1f}]  "
              f"y=[{xy[:,1].min():8.1f},{xy[:,1].max():8.1f}]  "
              f"obj={fp['obj_names']}")

    z_ground, terrain_verts = build_ground_height_interpolator(scene, name_to_meshfile)
    print(f"\nTerrain: {len(terrain_verts)} vertices, "
          f"x=[{terrain_verts[:,0].min():.1f},{terrain_verts[:,0].max():.1f}] "
          f"y=[{terrain_verts[:,1].min():.1f},{terrain_verts[:,1].max():.1f}] "
          f"z=[{terrain_verts[:,2].min():.1f},{terrain_verts[:,2].max():.1f}]")

    tx_xy = np.array([[-70.0, -25.0]])
    inside = points_inside_any_building(tx_xy, footprints)
    z_tx_ground = z_ground(tx_xy)[0]
    print(f"\nTX (-70,-25,15): inside-a-building={inside[0]}  "
          f"ground_z_here={z_tx_ground:.2f}  "
          f"tx_height_above_ground={15.0 - z_tx_ground:.2f} m")
    for fp in footprints:
        if fp["name"] == "LG연구동":
            print(f"LG연구동 height={fp['height']:.2f} -> "
                  f"TX height above roof = {15.0 - fp['height']:.2f} m")
