from DFSPH_solver import *
from Collision_solver import *
import time, tqdm
class PlaceHolder:
    def __init__(self): pass
    update = compute_N_rho_alpha = get_rigid_pos = __init__
class Scene:
    def __init__(self, dimensions, center, fluid: Fluid, rigids: list[Rigid], time_step, ):
        self.DFSPH_solver = DFSPH_solver(dimensions, center, fluid, rigids) if fluid is not None else PlaceHolder()
        self.Collision_solver = Collision_solver(rigids, 1e-3)
        self.sub_time_step = fluid.time_step if fluid is not None else 1e-4
        self.subframes = int(time_step / self.sub_time_step)
        self.fluid = fluid
        self.rigids = rigids

    def start_for_frames(self, frames):
        for i in range(frames):
            if i == 0:
                self.DFSPH_solver.get_rigid_pos()
                self.DFSPH_solver.compute_N_rho_alpha()
            else:
                for _ in tqdm.tqdm(range(self.subframes), desc=f"Frame {i - 1 :>3} to {i :>3}, start: {time.strftime('%H:%M:%S')}"):
                    self.DFSPH_solver.update()
                    self.Collision_solver.update(self.sub_time_step)
            self.write(i)

    def write(self, i):
        if self.fluid is not None:
            self.fluid.write(i)
        for rigid in self.rigids:
            if rigid is not None:
                rigid.write(i)

