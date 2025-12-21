import taichi as ti
import numpy as np
import trimesh

@ti.data_oriented
class Rigid:
    def __init__(self, obj_prefix, mesh: trimesh.Trimesh, offset = np.array([0.0, 0.0, 0.0]), velocity = np.array([0.0, 0.0, 0.0]), angular_velocity = np.array([0.0, 0.0, 0.0]), gravity = np.array([0.0, -9.8, 0.0]), orientation = np.eye(3), density = 1000.0, fixed = True):
        self.obj_prefix = obj_prefix
        cm = mesh.center_mass

        self.mesh = mesh.apply_translation(-cm)
        trimesh.repair.fix_normals(self.mesh)
        # mesh.vertices = mesh.vertices.astype(np.float32)
        # mesh.faces = mesh.faces.astype(np.int32)
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.vertices))
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=len(mesh.faces))
        self.normals = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.faces))
        self.vertices.from_numpy(mesh.vertices)
        self.faces.from_numpy(mesh.faces)
        self.normals.from_numpy(mesh.face_normals)
        # for i in range(self.faces.shape[0]):
        #     print(self.normals[i][0], self.normals[i][1], self.normals[i][2])
        # print(f"Total {self.faces.shape[0]} normals")

        self.gravity = ti.Vector(gravity)
        self.density = density
        self.volume = mesh.volume
        self.mass = self.density * self.volume
        self.inertia_tensor = ti.Matrix(mesh.moment_inertia) * self.density

        self.voxel = np.ascontiguousarray(mesh.voxelized(0.01).fill().points, np.float32)
        self.num_particles = self.voxel.shape[0]
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_particles,))
        self.positions.from_numpy(self.voxel)
        # for i in range(self.positions.shape[0]):
        #     if(self.positions[i][1] < -0.09):
        #         print(i)

        self.position = ti.Vector.field(3, dtype=ti.f32, shape=()) # position of the center of mass
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=()) # velocity of the center of mass
        self.orientation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=()) # orientation matrix of the body
        self.angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=()) # angular velocity of the body
        self.position[None] = cm + offset
        self.velocity[None] = velocity
        self.orientation[None] = orientation
        self.angular_velocity[None] = angular_velocity
        self.collision_threshold = 1e-4
        
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.torque = ti.Vector.field(3, dtype=ti.f32, shape=()) # torque relative to the center of mass
        self.angular_momentum = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.fixed = fixed

    def write(self, i):
        with open(self.obj_prefix + str(i) + ".obj", 'w') as file:
            pos = self.position.to_numpy()
            orient = self.orientation.to_numpy()
            for v in self.mesh.vertices:
                v2 = orient @ v + pos
                file.write(f"v {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            for f in self.mesh.faces:
                file.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

    @ti.func
    def apply_force(self, force, pos):
        self.force[None] += force
        self.torque[None] += ti.math.cross(pos - self.position[None], force)

    @ti.kernel
    def update(self, time_step: ti.f32):
        if not self.fixed:
            acceleration = self.force[None] / self.mass
            self.velocity[None] += (acceleration + self.gravity) * time_step
            self.position[None] += self.velocity[None] * time_step

            # Angular motion
            # angular_acceleration = self.torque[None] / self.mass  # Simplified, should use inertia tensor
            inertia_tensor_now = self.orientation[None] @ self.inertia_tensor @ self.orientation[None].transpose() # inertia tensor relative to the center of mass with respect to the current frame
            self.angular_momentum[None] = inertia_tensor_now @ self.angular_velocity[None]
            torque = self.torque[None] - ti.math.cross(self.angular_velocity[None], self.angular_momentum[None])
            angular_acceleration = inertia_tensor_now.inverse() @ torque
            self.angular_velocity[None] += angular_acceleration * time_step
            angular_velocity_norm = self.angular_velocity[None].norm()
            exp_A = ti.Matrix.identity(ti.f32, 3)
            # print(angular_velocity_norm)
            if angular_velocity_norm > 1e-8:
                angular_velocity_matrix = ti.Matrix([
                    [0, -self.angular_velocity[None][2], self.angular_velocity[None][1]],
                    [self.angular_velocity[None][2], 0, -self.angular_velocity[None][0]],
                    [-self.angular_velocity[None][1], self.angular_velocity[None][0], 0]
                ]) / angular_velocity_norm

                exp_A = ti.Matrix.identity(ti.f32, 3) + angular_velocity_matrix * ti.sin(angular_velocity_norm * time_step) + angular_velocity_matrix @ angular_velocity_matrix * (1 - ti.cos(angular_velocity_norm * time_step))
            # print(exp_A)
            self.orientation[None] = exp_A @ self.orientation[None]

            # Reset forces and torques
            self.force[None] = ti.Vector([0.0, 0.0, 0.0])
            self.torque[None] = ti.Vector([0.0, 0.0, 0.0])