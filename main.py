import os

os.environ["TI_WARNINGS"] = "0"
frames = 90

# e.g. In frame 24, the ply data of fluid will go to fluid_ply_prefix + "24.ply"
# also used in render.sh
fluid_ply_prefix = "output5/ply/fluid"
fluid_obj_prefix = "output5/obj/fluid"
rigids_obj_prefix = [
    "output5/obj/rigid0_",
    "output5/obj/rigid1_",
    "output5/obj/rigid2_",
    "output5/obj/rigid3_",
    "output5/obj/rigid4_",
    "output5/obj/rigid5_",
    "output5/obj/rigid6_",
    "output5/obj/rigid7_",
    "output5/obj/rigid8_",
    "output5/obj/rigid9_",
]
output_png_prefix = "output5/png/"
output_video = "output-box.mp4"

os.makedirs(os.path.dirname(fluid_ply_prefix), exist_ok=True)
os.makedirs(os.path.dirname(fluid_obj_prefix), exist_ok=True)
os.makedirs(os.path.dirname(output_png_prefix), exist_ok=True)
for p in rigids_obj_prefix:
    os.makedirs(os.path.dirname(p), exist_ok=True)

# Indicators: must match the parameters of Scene() in the init_*() you use
# whether you want to render the fluid data
has_fluid = False
# whether you want to render the corresponding rigid data
has_rigids = [True] * 10 + [False] * 6


def visc_from_shear_func():
    import taichi as ti

    @ti.func
    def visc_from_shear(shear):
        # return 0.008
        return min(0.05 * max(1.0, shear**4), 0.5)

    return visc_from_shear


def traj1(time):
    return ti.Vector([0, -max(min(1.0, 4 * time - 2.0), 0), 0])


def init_full():
    fluid = Fluid(fluid_ply_prefix, box([0.6, 0.8, 0.6]), [0.0, -0.55, 0.0])
    rigid1 = Rigid(rigids_obj_prefix[0], torus(0.2, 0.1), [-0.4, 1.0, 0.4], density=500.0, fixed=False)
    rigid2 = Rigid(rigids_obj_prefix[1], torus(0.2, 0.1), [-0.5, 0.4, 0.3], density=500.0, trajectory=traj1)
    rigid3 = Rigid(rigids_obj_prefix[2], torus(0.2, 0.1), [-0.5, 0.4, 0.6], [0.0, 5.0, 0.0], density=500.0, fixed=False)
    scene = Scene(fluid=fluid, rigids=[rigid1, rigid2, rigid3])
    return scene


def init_box():
    def random_orientation():
        X = np.random.randn(3, 3)
        Q, R = np.linalg.qr(X)
        return Q * np.sign(np.diag(R))

    rigid1 = Rigid(
        rigids_obj_prefix[0],
        box([0.2, 0.2, 0.2]),
        [0.4, 0.8, -0.4],
        [0.0, -5.0, 0.0],
        gravity=[0.0, 0.0, 0.0],
        orientation=random_orientation(),
        fixed=False,
    )
    rigid2 = Rigid(
        rigids_obj_prefix[1],
        box([0.2, 0.2, 0.2]),
        [-0.5, 0.4, 0.3],
        [0.0, -5.0, 0.0],
        gravity=[0.0, 0.0, 0.0],
        orientation=random_orientation(),
        fixed=False,
    )
    rigid3 = Rigid(
        rigids_obj_prefix[2],
        box([0.2, 0.2, 0.2]),
        [-0.5, 0.4, 0.6],
        [0.0, -5.0, 0.0],
        gravity=[0.0, 0.0, 0.0],
        orientation=random_orientation(),
        fixed=False,
    )
    rigid4 = Rigid(
        rigids_obj_prefix[3],
        box([0.2, 0.2, 0.2]),
        [0.3, 0.5, -0.2],
        [0.0, -5.0, 0.0],
        gravity=[0.0, 0.0, 0.0],
        orientation=random_orientation(),
        fixed=False,
    )
    rigid5 = Rigid(
        rigids_obj_prefix[4],
        box([0.2, 0.2, 0.2]),
        [0.5, 0.2, -0.1],
        [0.0, -5.0, 0.0],
        gravity=[0.0, 0.0, 0.0],
        orientation=random_orientation(),
        fixed=False,
    )
    rigid6 = Rigid(
        rigids_obj_prefix[5],
        box([0.2, 0.2, 0.2]),
        [0.0, 0.4, 0.0],
        [0.0, -5.0, 0.0],
        gravity=[0.0, 0.0, 0.0],
        orientation=random_orientation(),
        fixed=False,
    )
    rigid7 = Rigid(
        rigids_obj_prefix[6],
        box([0.2, 0.2, 0.2]),
        [-0.2, -0.1, 0.3],
        [0.0, -5.0, 0.0],
        gravity=[0.0, 0.0, 0.0],
        orientation=random_orientation(),
        fixed=False,
    )
    rigid8 = Rigid(
        rigids_obj_prefix[7],
        box([0.2, 0.2, 0.2]),
        [-0.3, -0.3, 0.5],
        [0.0, -5.0, 0.0],
        gravity=[0.0, 0.0, 0.0],
        orientation=random_orientation(),
        fixed=False,
    )
    rigid9 = Rigid(
        rigids_obj_prefix[8],
        box([0.2, 0.2, 0.2]),
        [0.4, 0.3, 0.2],
        [0.0, -5.0, 0.0],
        gravity=[0.0, 0.0, 0.0],
        orientation=random_orientation(),
        fixed=False,
    )
    rigid10 = Rigid(
        rigids_obj_prefix[9],
        box([0.2, 0.2, 0.2]),
        [0.2, -0.3, -0.1],
        [0.0, -5.0, 0.0],
        gravity=[0.0, 0.0, 0.0],
        orientation=random_orientation(),
        fixed=False,
    )

    rb1 = Rigid(mesh=box([2.0, 2.0, 0.2]), offset=[0.0, 0.0, 1.1], gravity=[0.0, 0.0, 0.0])
    rb2 = Rigid(mesh=box([2.0, 2.0, 0.2]), offset=[0.0, 0.0, -1.1], gravity=[0.0, 0.0, 0.0])
    rb3 = Rigid(mesh=box([2.0, 0.2, 2.0]), offset=[0.0, 1.1, 0.0], gravity=[0.0, 0.0, 0.0])
    rb4 = Rigid(mesh=box([2.0, 0.2, 2.0]), offset=[0.0, -1.1, 0.0], gravity=[0.0, 0.0, 0.0])
    rb5 = Rigid(mesh=box([0.2, 2.0, 2.0]), offset=[1.1, 0.0, 0.0], gravity=[0.0, 0.0, 0.0])
    rb6 = Rigid(mesh=box([0.2, 2.0, 2.0]), offset=[-1.1, 0.0, 0.0], gravity=[0.0, 0.0, 0.0])
    scene = Scene(rigids=[rigid1, rigid2, rigid3, rigid4, rigid5, rigid6, rigid7, rigid8, rigid9, rigid10, rb1, rb2, rb3, rb4, rb5, rb6])
    return scene


def init_nn():
    fluid = Fluid(fluid_ply_prefix, box([0.6, 0.8, 0.6]), [0.0, -0.55, 0.0])
    rigid1 = Rigid(rigids_obj_prefix[0], box([0.2, 0.2, 0.2]), [-0.15, 0.4, -0.15], density=500.0, fixed=False)
    rigid2 = Rigid(rigids_obj_prefix[1], box([0.2, 0.2, 0.2]), [0.15, 0.4, 0.15], density=500.0, fixed=False)
    scene = Scene(fluid=fluid, rigids=[rigid1, rigid2])
    return scene


if __name__ == "__main__":
    from scene import Fluid, Rigid, Scene
    import taichi as ti
    import numpy as np
    from trimesh.creation import box, torus

    ti.init(ti.gpu, device_memory_fraction=0.9)
    scene = init_box()
    scene.start_for_frames(frames)
