import taichi as ti
from scene import *
import sys

def init_fluid():    
    fluid = Fluid([0.6, 0.8, 0.4], [1.0, -0.4, -6])
    scene = Scene([2.4, 1.0, 0.8], [1.0, -0.4, -6], fluid, [None], 1.0 / 60)
    return scene

ti.init(ti.gpu, device_memory_GB = 30)
fluid = Fluid([0.6, 0.8, 0.6], [1.0, -1.0, -6])
rigid1 = Rigid(trimesh.creation.box([0.2, 0.2, 0.2]), [0.85, 0.45, -6.15], density=500.0, fixed=False)
rigid2 = Rigid(trimesh.creation.box([0.2, 0.2, 0.2]), [1.15, 0.45, -5.85], density=500.0, fixed=False)
scene = Scene([1.0, 2.0, 1.0], [1.0, -0.45, -6], fluid, [rigid1, rigid2], 1.0 / 60)
scene.start_for_frames(int(sys.argv[1]), sys.argv[2])