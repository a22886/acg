import taichi as ti
import trimesh
import numpy as np
# from ..utils import *
import time

@ti.data_oriented
class Fluid:
    def __init__(self, ply_prefix, mesh: trimesh.Trimesh,
                 center=[0.0, 0.0, 0.0],
                 gravity=[0.0, -9.8, 0.0], rest_density=1000.0,
                 time_step=1e-4):
        self.ply_prefix = ply_prefix

        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity[None] = gravity
        self.rest_density = rest_density
        self.time_step = time_step

        self.volume = ti.field(dtype=ti.f32, shape=())
        self.h = 0.04
        self.surface_tension = 0.01
        self.particle_diameter = 0.02
        
        self.volume[None] = mesh.volume
        
        mesh.apply_translation(center)
        voxel = np.ascontiguousarray(mesh.voxelized(0.01).fill().points, np.float32)
        print(np.max(voxel, axis=0), np.min(voxel, axis=0))
        num_particles = voxel.shape[0]
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=(num_particles,))
        self.positions.from_numpy(voxel)
        print(f"Number of particles: {num_particles}")

        self.num_particles = num_particles
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.densities = ti.field(dtype=ti.f32, shape=num_particles)
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.viscosity = ti.field(dtype=ti.f32, shape=num_particles)

        self.init_velocity_and_density()
        self.p_mass = self.volume[None] / self.num_particles * self.rest_density
        print("Fluid initialized successfully")    

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

    def write(self, i):
        with open(self.ply_prefix + str(i) + ".ply", 'w') as f:
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