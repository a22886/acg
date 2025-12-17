from rigid import *

@ti.data_oriented
class Collision_solver:
    def __init__(self, rigids: list[Rigid], threshold):
        self.rigids = rigids
        self.nr = len(rigids)
        self.collide_position = ti.Vector([0.0, 0.0, 0.0])
        self.collide_normal = ti.Vector([0.0, 0.0, 0.0])

    def update(self, time_step):
        for i in range(self.nr):
            self.rigids[i].update(time_step)
        self.update_velocity()
    
    @ti.kernel
    def is_colliding(self, i: int, j: int) -> bool:
        return 0
        pass

    def compute_collision(self, i, j):
        if self.is_colliding(i, j):
            # compute the velocity after collision
            pass

    def update_velocity(self):
        # compute collisions in a specific order
        for i in range(self.nr):
            for j in range(i + 1, self.nr):
                self.compute_collision(i, j)