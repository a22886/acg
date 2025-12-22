import os

frames = 90

# e.g. In frame 24, the ply data of fluid will go to fluid_ply_prefix + "24.ply"
# also used in render.sh
fluid_ply_prefix = "output5/ply/fluid"
fluid_obj_prefix = "output5/obj/fluid"
rigids_obj_prefix = ["output5/obj/rigid0_", "output5/obj/rigid1_", "output5/obj/rigid2_"]
output_png_prefix = "output5/png/"
output_video = "output-full.mp4"

os.makedirs(os.path.dirname(fluid_ply_prefix), exist_ok=True)
os.makedirs(os.path.dirname(fluid_obj_prefix), exist_ok=True)
os.makedirs(os.path.dirname(output_png_prefix), exist_ok=True)
for p in rigids_obj_prefix:
    os.makedirs(os.path.dirname(p), exist_ok=True)

# Indicators: must match the objects in init_scene(), used in render.py
# whether you want to render the fluid data
has_fluid = True
# whether you want to render the corresponding rigid data
has_rigids = [True, True, True]

def visc_from_shear_func():
    import taichi as ti
    @ti.func
    def visc_from_shear(shear):
        return 0.008
        # return min(0.05 * max(1.0, shear ** 4), 0.5)
    return visc_from_shear

def traj1(time):
    return ti.Vector([0, -max(min(1.0, 4 * time - 2.0), 0), 0])

if __name__ == '__main__':
    from scene import Fluid, Rigid, Scene, ti, trimesh
    ti.init(ti.gpu, device_memory_fraction=0.9)
    fluid = Fluid(fluid_ply_prefix, trimesh.creation.box([0.6, 0.8, 0.6]), [0.0, -0.55, 0.0])
    rigid1 = Rigid(rigids_obj_prefix[0], trimesh.creation.torus(0.2, 0.1), [-0.4, 1.0, 0.4], density=500.0, fixed=False)
    rigid2 = Rigid(rigids_obj_prefix[1], trimesh.creation.torus(0.2, 0.1), [-0.5, 0.4, 0.3], density=500.0, trajectory=traj1)
    rigid3 = Rigid(rigids_obj_prefix[2], trimesh.creation.torus(0.2, 0.1), [-0.5, 0.4, 0.6], [0.0, 5.0, 0.0], density=500.0, fixed=False)
    scene = Scene([1.0, 2.0, 1.0], [0.0, 0.0, 0.0], fluid, [], 1.0 / 60)
    scene.start_for_frames(frames)