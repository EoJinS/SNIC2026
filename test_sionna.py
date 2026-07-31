import sionna

scene = sionna.rt.load_scene(sionna.rt.scene.etoile)
try:
    paths = scene.compute_paths(max_depth=1)
except Exception as e:
    print(f"scene.compute_paths error: {e}")

try:
    paths = sionna.rt.compute_paths(scene=scene, max_depth=1)
except Exception as e:
    print(f"sionna.rt.compute_paths error: {e}")
