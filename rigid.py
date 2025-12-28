import taichi as ti
import numpy as np
import trimesh


@ti.data_oriented
class Rigid:
    def __init__(
        self,
        obj_prefix="",
        mesh: trimesh.Trimesh = None,
        offset=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        gravity=np.array([0.0, -9.8, 0.0]),
        orientation=np.eye(3),
        density=1000.0,
        fixed=True,
        trajectory=None,
    ):
        self.obj_prefix = obj_prefix
        cm = mesh.center_mass
        self.traj = trajectory
        self.t = ti.field(dtype=ti.f32, shape=())

        mesh.apply_translation(-cm)
        trimesh.repair.fix_normals(mesh)
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.vertices))
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=len(mesh.faces))
        self.normals = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.faces))
        self.vertices.from_numpy(mesh.vertices)
        self.faces.from_numpy(mesh.faces)
        self.normals.from_numpy(mesh.face_normals)

        self.gravity = ti.Vector(gravity)
        self.density = density
        self.volume = mesh.volume
        self.mass = self.density * self.volume
        self.inertia_tensor = ti.Matrix(mesh.moment_inertia) * self.density

        self.voxel = np.ascontiguousarray(mesh.voxelized(0.01).fill().points, np.float32)
        self.num_particles = self.voxel.shape[0]
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_particles,))
        self.positions.from_numpy(self.voxel)

        self.position = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.orientation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.position[None] = cm + offset
        self.velocity[None] = velocity
        self.orientation[None] = orientation
        self.angular_velocity[None] = angular_velocity

        self.force = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.torque = ti.Vector.field(3, dtype=ti.f32, shape=())  # torque relative to the center of mass
        self.angular_momentum = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.fixed = fixed

        print(f"Initialized rigid with {self.num_particles} particles")

    def write(self, i):
        with open(self.obj_prefix + str(i) + ".obj", "w") as file:
            pos = self.position[None]
            orient = self.orientation[None]
            for vi in range(self.vertices.shape[0]):
                v2 = orient @ self.vertices[vi] + pos
                file.write(f"v {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            for fi in range(self.faces.shape[0]):
                f = self.faces[fi]
                file.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

    @ti.func
    def apply_force(self, force, pos):
        self.force[None] += force
        self.torque[None] += ti.math.cross(pos - self.position[None], force)

    # @ti.kernel
    def update(self, time_step: ti.f32):
        if not self.fixed:
            self.update_pos_unfixed(time_step)
        elif self.traj != None:
            epsilon = 1e-5
            self.position[None] += self.traj(self.t[None] + time_step) - self.traj(self.t[None])
            self.t[None] += time_step
            self.velocity[None] = (self.traj(self.t[None] + epsilon) - self.traj(self.t[None])) / epsilon

    @ti.kernel
    def update_pos_unfixed(self, time_step: ti.f32):
        acceleration = self.force[None] / self.mass
        self.velocity[None] += (acceleration + self.gravity) * time_step
        self.position[None] += self.velocity[None] * time_step

        orient = self.orientation[None]
        angv = self.angular_velocity[None]
        it = orient @ self.inertia_tensor @ orient.transpose()
        beta = it.inverse() @ (self.torque[None] - ti.math.cross(angv, it @ angv))
        self.angular_velocity[None] += beta * time_step

        angv = self.angular_velocity[None]
        l = angv.norm()
        exp_A = ti.Matrix.identity(ti.f32, 3)
        if l > 1e-6:
            M = (
                ti.Matrix(
                    [
                        [0, -angv[2], angv[1]],
                        [angv[2], 0, -angv[0]],
                        [-angv[1], angv[0], 0],
                    ]
                )
                / l
            )

            exp_A = ti.Matrix.identity(ti.f32, 3) + M * ti.sin(l * time_step) + M @ M * (1 - ti.cos(l * time_step))
        # print(exp_A)
        self.orientation[None] = exp_A @ self.orientation[None]

        self.force[None] = ti.Vector([0.0, 0.0, 0.0])
        self.torque[None] = ti.Vector([0.0, 0.0, 0.0])
