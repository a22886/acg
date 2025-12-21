from rigid import *
import time
@ti.data_oriented
class Collision_solver:
    def __init__(self, rigids: list[Rigid], threshold):
        self.rigids = rigids
        self.nr = len(rigids)
        self.threshold = threshold

    def update(self, time_step):
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
    def update_velocity(self):
        # compute collisions in a specific order
        for i in ti.static(range(self.nr)):
            ri = self.rigids[i]
            ori = ri.orientation[None]
            pi = ri.position[None]
            vi = ri.velocity[None]
            angvi = ri.angular_velocity[None]
            for j in ti.static(range(self.nr)):
                num = 0
                point = ti.Vector([0.0, 0.0, 0.0])
                normal = ti.Vector([0.0, 0.0, 0.0])
                rj = self.rigids[j]
                if not rj.fixed and (ri.fixed or i < j):
                    t = 0.0
                    orient = rj.orientation[None]
                    pos = rj.position[None]
                    vel = rj.velocity[None]
                    ang_v = rj.angular_velocity[None]
                    for nj in range(rj.num_particles):
                        t = 100.0
                        nm = ti.Vector([0.0, 0.0, 0.0])
                        pos_j = orient @ rj.positions[nj] + pos
                        vel_j = vel + ti.math.cross(ang_v, pos_j - pos)
                        for fi in range(ri.faces.shape[0]):
                            n = ri.normals[fi]
                            denom = ti.math.dot(vel_j, n)
                            ver_i = ori @ ri.vertices[ri.faces[fi][0]] + pi
                            ver_i1 = ori @ ri.vertices[ri.faces[fi][1]] + pi
                            ver_i2 = ori @ ri.vertices[ri.faces[fi][2]] + pi
                            numer = ti.math.dot(ver_i - pos_j, n)
                            # print(denom, numer)
                            # time.sleep(3)
                            if denom < -1e-5 and numer < -1e-5:
                                tij = numer / denom
                                if tij < t and tij * vel_j.norm() < self.threshold and self.in_triangle(ver_i, ver_i1, ver_i2, pos_j + tij * vel_j):
                                    t = tij
                                    nm = n
                        if t < 99.999 and nm.norm() > 0.99:
                            num += 1
                            point += pos_j
                            normal += nm
                    if num != 0:
                        print("---------------------------------------------")
                        print(t, i, j)
                        print(point/num, normal/num, num)
                        point /= num
                        normal /= num
                        # if ri.fixed:
                        RHS = 2 * ti.math.dot(vi - vel + ti.math.cross(angvi, point - pi) - ti.math.cross(ang_v, point - pos), normal)
                        inertia_i = ri.inertia_tensor
                        inertia_j = (orient @ rj.inertia_tensor @ orient.transpose()).inverse()
                        LHS = ti.math.dot(ti.math.cross(inertia_j @ ti.math.cross(point - pos, normal), point - pos) + normal / rj.mass, normal)
                        if not ri.fixed:
                            inertia_i = (ori @ ri.inertia_tensor @ ori.transpose()).inverse()
                            LHS += ti.math.dot(ti.math.cross(inertia_i @ ti.math.cross(point - pi, normal), point - pi) + normal / ri.mass, normal)
                        A = RHS / LHS
                        rj.velocity[None] = vel + A * normal / rj.mass
                        rj.angular_velocity[None] = ang_v + A * (inertia_j @ ti.math.cross(point - pos, normal))
                        if not ri.fixed:
                            ri.velocity[None] = vi - A * normal / ri.mass
                            ri.angular_velocity[None] = angvi - A * (inertia_i @ ti.math.cross(point - pi, normal))
                        # print(rj.velocity[None], rj.angular_velocity[None])