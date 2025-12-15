import taichi as ti
import numpy as np
# from ..utils import *
import time

@ti.data_oriented
class Fluid:
    def __init__(self, dimensions,
                 center=[0.0, 0.0, 0.0],
                 gravity=[0.0, -9.8, 0.0], rest_density=1000.0,
                 time_step=1e-4):
        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity[None] = gravity
        self.rest_density = rest_density
        self.time_step = time_step

        self.volume = ti.field(dtype=ti.f32, shape=())
        self.h = 0.04
        self.surface_tension = 0.01
        self.particle_diameter = 0.02
        # self.fps = fps
        
        self.volume[None] = dimensions[0] * dimensions[1] * dimensions[2]
        self.center = ti.Vector(center)
        self.dimensions = ti.Vector(dimensions)
        
        self.init_pos()
        # self.init_mass()
        self.init_velocity_and_density()
        self.p_mass = self.volume[None] / self.num_particles * self.rest_density
        print("Fluid initialized successfully")
    
    def init_pos(self):
        self.grid_size = 0.01
        num_particles = 0
        
        grid_num = np.array(self.dimensions / self.grid_size).astype(int)
        
        time1 = time.time()
        
        x = self.center[0] - self.dimensions[0] / 2 + self.grid_size * np.arange(grid_num[0])
        y = self.center[1] - self.dimensions[1] / 2 + self.grid_size * np.arange(grid_num[1])
        z = self.center[2] - self.dimensions[2] / 2 + self.grid_size * np.arange(grid_num[2])
        grid = np.array(np.meshgrid(x, y, z, indexing='ij')).astype(np.float32).reshape(3, -1).T
        
        # useful_grid = self.mesh.contains(grid)
        # useful_grid = np.where(useful_grid)[0]
        num_particles = len(grid) # = grid_num[0] * grid_num[1] * grid_num[2]
        # position = grid
        time2 = time.time()
        print(f"Time taken to initialize particles: {time2 - time1}")
        print(f"Number of particles: {num_particles}")

        self.num_particles = num_particles
        # self.mass = ti.field(dtype=ti.f32, shape=num_particles)
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.densities = ti.field(dtype=ti.f32, shape=num_particles)
        # self.pressures = ti.field(dtype=ti.f32, shape=num_particles)
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        # self.particle_volume = ti.field(dtype=ti.f32, shape=num_particles)
        self.viscosity = ti.field(dtype=ti.f32, shape=num_particles)
        
        self.positions.from_numpy(grid)
        
    @ti.kernel
    def init_mass(self):
        for i in range(self.num_particles):
            self.particle_volume[i] = self.volume[None] / self.num_particles
            # self.particle_volume[i] = 0.8 * self.particle_diameter ** 3
            self.mass[i] = self.rest_density * self.particle_volume[i]       

    @ti.func
    def kernel_func(self, r):
        ret = 0.0
        k = 8 / np.pi / self.h**3
        q = r / self.h
        if q <= 0.5:
            ret = k * (6 * q**3 - 6 * q**2 + 1)
        elif q <= 1.0:
            ret = 2 * k * (1 - q)**3
        return ret

    @ti.func
    def kernel_grad(self, r):
        ret = ti.Vector([0.0, 0.0, 0.0])
        k = 48 / np.pi / self.h**3
        l = r.norm()
        q = l / self.h
        if l > 1e-6:
            vec_k = k * r / l
            if q <= 0.5:
                ret = vec_k * q * (3 * q - 2) / self.h
            elif q <= 1.0:
                ret = -vec_k * (1 - q)**2 / self.h
        return ret
        
    @ti.kernel
    def init_velocity_and_density(self):
        for i in range(self.num_particles):
            self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
            self.densities[i] = self.rest_density
            # self.viscosity[i] = viscosity.visc_from_shear(0.0)   

    def positions_to_ply(self, output_path):
        with open(output_path, 'w') as f:
            f.write(f'ply\nformat ascii 1.0\nelement vertex {self.num_particles}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
            for pos in self.positions.to_numpy():
                f.write(f'{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n')
    
    @ti.kernel
    def update_velocity(self):
        for i in range(self.num_particles):
            self.velocities[i] += self.time_step * (self.forces[i] / self.p_mass + self.gravity[None])
        # avg_velocity = ti.Vector([0.0, 0.0, 0.0])
        # for i in range(self.num_particles):
        #     avg_velocity += self.velocities[i]
        # avg_velocity /= self.num_particles
        # print(avg_velocity)
    
    @ti.kernel
    def update_position(self):
        for i in range(self.num_particles):
            self.positions[i] += self.time_step * self.velocities[i]