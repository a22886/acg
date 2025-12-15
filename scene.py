# import taichi as ti
from DFSPH_solver import *
from Collision_solver import *
import time
import tqdm

# @ti.data_oriented
class Scene:
    def __init__(self, dimensions, center, fluid: Fluid, rigids: list[Rigid], time_step, ):
        self.DFSPH_solver = None
        self.Collision_solver = Collision_solver(rigids, 1e-4)
        self.subframes = 0
        self.time_step = time_step
        if fluid is not None:
            self.DFSPH_solver = DFSPH_solver(dimensions, center, fluid, rigids)
            self.subframes = int(time_step / fluid.time_step)
            self.time_step = self.subframes * fluid.time_step

    def start_for_frames(self, frames, dir):
        if self.DFSPH_solver is not None:
            for i in range(frames):
                max_v = 0.0
                if i == 0:
                    self.DFSPH_solver.get_rigid_pos()
                    self.DFSPH_solver.compute_N_rho_alpha()
                else:
                    for _ in tqdm.tqdm(range(self.subframes), desc=f"Frame {i - 1 :>3} to {i :>3}, start: {time.strftime('%H:%M:%S')}"):
                        self.DFSPH_solver.update()
                        max_v = max(max_v, self.DFSPH_solver.max_v[None])
                    print(f"max_v: {max_v}")
                self.Collision_solver.update_velocity()
                self.DFSPH_solver.write(dir, i)
        else:
            for i in range(frames):
                self.Collision_solver.update(self.time_step)
                self.Collision_solver.write(dir, i)

