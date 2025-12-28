from rigid import *


@ti.data_oriented
class Collision_solver:
    def __init__(self, rigids: list[Rigid], threshold):
        self.rigids = rigids
        self.nr = len(rigids)
        self.threshold = threshold
        self.orientations = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.nr)
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.nr)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.nr)
        self.angular_velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.nr)

        self.before_sizes = [0]
        self.before_faces = [0]
        self.before_vertices = [0]
        n = 0
        f = 0
        v = 0
        for rigid in rigids:
            n += rigid.num_particles
            f += rigid.faces.shape[0]
            v += rigid.vertices.shape[0]
            self.before_sizes.append(n)
            self.before_faces.append(f)
            self.before_vertices.append(v)

        self.particle_positions = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=f)
        self.normals = ti.Vector.field(3, dtype=ti.f32, shape=f)
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=v)
        self.init_self()

        self.point = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.normal = ti.Vector.field(3, dtype=ti.f32, shape=())
        print(f"Initialized collsion solver")

    @ti.kernel
    def init_self(self):
        for i in ti.static(range(self.nr)):
            ri = self.rigids[i]
            for j in range(self.before_sizes[i], self.before_sizes[i + 1]):
                self.particle_positions[j] = ri.positions[j - self.before_sizes[i]]
            for j in range(self.before_faces[i], self.before_faces[i + 1]):
                self.faces[j] = ri.faces[j - self.before_faces[i]]
                self.normals[j] = ri.normals[j - self.before_faces[i]]
            for j in range(self.before_vertices[i], self.before_vertices[i + 1]):
                self.vertices[j] = ri.vertices[j - self.before_vertices[i]]

    def update(self, time_step):
        for i in range(self.nr):
            self.orientations[i] = self.rigids[i].orientation[None]
            self.positions[i] = self.rigids[i].position[None]
            self.velocities[i] = self.rigids[i].velocity[None]
            self.angular_velocities[i] = self.rigids[i].angular_velocity[None]
        for i in range(self.nr):
            self.rigids[i].update(time_step)
        self.update_velocity()

    @ti.func
    def in_triangle(self, A, B, C, P):
        nA = ti.math.cross(B - P, C - P)
        nB = ti.math.cross(C - P, A - P)
        nC = ti.math.cross(A - P, B - P)
        return ti.math.dot(nA, nB) > 0 and ti.math.dot(nB, nC) > 0

    @ti.kernel
    def is_colliding(self, i: int, j: int, bsj: int, bsj1: int, bfi: int, bfi1: int, bvi: int) -> bool:
        ori = self.orientations[i]
        pi = self.positions[i]
        vi = self.velocities[i]
        angvi = self.angular_velocities[i]

        orj = self.orientations[j]
        pj = self.positions[j]
        vj = self.velocities[j]
        angvj = self.angular_velocities[j]

        self.point[None] = ti.Vector([0.0, 0.0, 0.0])
        self.normal[None] = ti.Vector([0.0, 0.0, 0.0])

        t = 0.0
        num = 0
        ret = False
        for nj in range(bsj, bsj1):
            t = 100.0
            nm = ti.Vector([0.0, 0.0, 0.0])
            pos_j = orj @ self.particle_positions[nj] + pj
            vel_j = vj + ti.math.cross(angvj, pos_j - pj) - vi - ti.math.cross(angvi, pos_j - pi)
            for fi in range(bfi, bfi1):
                n = ori @ self.normals[fi]
                denom = ti.math.dot(vel_j, n)
                ver_i0 = ori @ self.vertices[self.faces[fi][0] + bvi] + pi
                ver_i1 = ori @ self.vertices[self.faces[fi][1] + bvi] + pi
                ver_i2 = ori @ self.vertices[self.faces[fi][2] + bvi] + pi
                numer = ti.math.dot(ver_i0 - pos_j, n)
                if denom < -1e-5 and numer < -1e-5:
                    tij = numer / denom
                    if tij < t and tij * vel_j.norm() < self.threshold and self.in_triangle(ver_i0, ver_i1, ver_i2, pos_j + tij * vel_j):
                        t = tij
                        nm = n
            if t < 99.999 and nm.norm() > 0.99:
                num += 1
                self.point[None] += pos_j
                self.normal[None] += nm
        if num != 0:
            # print("---------------------------------------------")
            # print(t, i, j)
            # print(self.point[None]/num, self.normal[None]/num, num)
            self.point[None] /= num
            self.normal[None] /= num
            ret = True
            # print(f"Collision at {self.point[None]} with normal {self.normal[None]} between rigids {i} and {j}")
        return ret

    def update_velocity(self):
        for i in range(self.nr):
            for j in range(self.nr):
                if not self.rigids[j].fixed and (self.rigids[i].fixed or i < j):
                    if self.is_colliding(
                        i,
                        j,
                        self.before_sizes[j],
                        self.before_sizes[j + 1],
                        self.before_faces[i],
                        self.before_faces[i + 1],
                        self.before_vertices[i],
                    ):
                        ori = self.rigids[i].orientation[None]
                        pi = self.rigids[i].position[None]
                        vi = self.rigids[i].velocity[None]
                        angvi = self.rigids[i].angular_velocity[None]
                        mi = self.rigids[i].mass
                        iti = self.rigids[i].inertia_tensor

                        orj = self.rigids[j].orientation[None]
                        pj = self.rigids[j].position[None]
                        vj = self.rigids[j].velocity[None]
                        angvj = self.rigids[j].angular_velocity[None]
                        mj = self.rigids[j].mass
                        itj = self.rigids[j].inertia_tensor

                        point = self.point[None]
                        normal = self.normal[None]

                        RHS = 2 * np.dot(
                            vi - vj + np.cross(angvi, point - pi) - np.cross(angvj, point - pj),
                            normal,
                        )
                        inertia_i = iti
                        inertia_j = (orj @ itj @ orj.transpose()).inverse().to_numpy()
                        LHS = np.dot(
                            np.cross(inertia_j @ np.cross(point - pj, normal), point - pj) + normal / mj,
                            normal,
                        )
                        if not self.rigids[i].fixed:
                            inertia_i = (ori @ iti @ ori.transpose()).inverse().to_numpy()
                            LHS += np.dot(
                                np.cross(inertia_i @ np.cross(point - pi, normal), point - pi) + normal / mi,
                                normal,
                            )
                        A = RHS / LHS
                        self.rigids[j].velocity[None] = vj + A * normal / mj
                        self.rigids[j].angular_velocity[None] = angvj + A * (inertia_j @ np.cross(point - pj, normal))
                        if not self.rigids[i].fixed:
                            self.rigids[i].velocity[None] = vi - A * normal / mi
                            self.rigids[i].angular_velocity[None] = angvi - A * (inertia_i @ np.cross(point - pi, normal))
                        # print(rj.velocity[None], rj.angular_velocity[None])
