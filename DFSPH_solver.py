from fluid import *
from rigid import *
from main import visc_from_shear_func
visc_from_shear = visc_from_shear_func()

@ti.data_oriented
class DFSPH_solver:
    def __init__(self, dimensions, center, fluid: Fluid, rigids: list[Rigid]):
        assert(fluid != None)
        self.dimensions = ti.Vector(dimensions)
        self.fluid = fluid
        self.center = ti.Vector(center)
        self.h = fluid.h
        self.fn = fluid.num_particles
        # self.rigid = rigid
        self.rigids = rigids
        self.max_k = ti.field(dtype=ti.f32, shape=())
        self.before_sizes = [0]
        self.rn = 0
        for rigid in rigids:
            self.rn += rigid.num_particles
            self.before_sizes.append(self.rn)
        self.nr = len(rigids)
        self.tn = self.fn + self.rn
        print(f"Total number of particles: {self.tn} with {self.before_sizes} of length {self.nr}")

        self.grid_size = 0.02
        self.grid_nums = np.array(self.dimensions / self.grid_size).astype(np.int32)
        
        self.idx_to_grid = ti.Vector.field(3, dtype=ti.i32, shape=(self.tn,))
        
        self.grid = ti.field(dtype=ti.i32)
        self.grid_num = ti.field(dtype=ti.i32)
        ti.root.dense(ti.ijk, self.grid_nums).dynamic(ti.l, 1024).place(self.grid)
        ti.root.dense(ti.ijk, self.grid_nums).place(self.grid_num)
        
        # self.neighbour = ti.field(dtype=ti.i32)
        self.mj_nablaWij = ti.Vector.field(4, dtype=ti.f32)
        # ti.root.dense(ti.i, self.max_num_particles).dynamic(ti.j, 2048).place(self.neighbour)
        ti.root.dense(ti.i, self.fn).dynamic(ti.j, 1000).place(self.mj_nablaWij)
        self.neighbour_num = ti.field(dtype=ti.i32, shape=self.fn)

        self.rigid_positions = ti.Vector.field(3, dtype=ti.f32, shape=ti.max(1, self.rn))
        self.rigid_velocities = ti.Vector.field(3, dtype=ti.f32, shape=ti.max(1, self.rn))      

        self.alpha = ti.field(dtype=ti.f32, shape=self.fn)
        self.kappa = ti.field(dtype=ti.f32, shape=self.fn)
        self.rho_derivative = ti.field(dtype=ti.f32, shape=self.fn)
        self.k_a_over_rho = ti.field(dtype=ti.f32, shape=self.fn)

        self.surf_tens = ti.Vector.field(3, dtype=ti.f32, shape=self.fn)

        self.div_max_it = 1000
        self.den_max_it = 1000

        self.div_max_err = 0.0001
        self.den_max_err = 0.0001

        self.p_mass = self.fluid.p_mass
        self.max_v = ti.field(dtype=ti.f32, shape=())
        self.max_non_gravity_force = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def get_rigid_pos(self):
        for i in ti.static(range(self.nr)):
            rigid = self.rigids[i]
            orient = rigid.orientation[None]
            pos = rigid.position[None]
            vel = rigid.velocity[None]
            ang_v = rigid.angular_velocity[None]
            bef = self.before_sizes[i]
            for j in range(bef, self.before_sizes[i+1]):
                self.rigid_positions[j] = orient @ rigid.positions[j - bef] + pos
                self.rigid_velocities[j] = vel + ti.math.cross(ang_v, self.rigid_positions[j] - pos)

    @ti.func
    def pos(self, i: int):
        ret = ti.Vector([0.0, 0.0, 0.0])
        if i < self.fn:
            ret = self.fluid.positions[i]
        else:
            ret = self.rigid_positions[i - self.fn]
        return ret
    
    @ti.func
    def vel(self, i: int):
        ret = ti.Vector([0.0, 0.0, 0.0])
        if i < self.fn:
            ret = self.fluid.velocities[i]
        else:
            ret = self.rigid_velocities[i - self.fn]
        return ret

    
    @ti.kernel
    def set_boundary_conditions(self):
        left = self.center - self.dimensions / 2 + self.h
        right = self.center + self.dimensions / 2 - self.h
        self.max_v[None] = 0.0
        for p_i in range(self.fn):
            cn = ti.Vector([0.0, 0.0, 0.0])
            pos = self.fluid.positions[p_i]
            for i in ti.static(range(3)):
                if pos[i] < left[i]:
                    self.fluid.positions[p_i][i] = left[i]
                    cn[i] += 1.0
                if pos[i] > right[i]:
                    self.fluid.positions[p_i][i] = right[i]
                    cn[i] -= 1.0

            cn_ns = cn.norm_sqr()
            if cn_ns > 0.5:
                recover_coef = 0.3
                self.fluid.velocities[p_i] -= (1 + recover_coef) * ti.math.dot(self.fluid.velocities[p_i], cn) * cn / cn_ns
                # self.simulate_collisions(
                #         p_i, collision_normal / collision_normal_length)
            # self.max_v[None] = max(self.max_v[None], self.fluid.velocities[p_i].norm())
    
    @ti.kernel
    def compute_N_rho_alpha(self):
        for x, y, z in ti.ndrange(self.grid_nums[0], self.grid_nums[1], self.grid_nums[2]):
            self.grid[int(x), int(y), int(z)].deactivate()
            self.grid_num[x, y, z] = 0
        for i in range(self.fn):
            self.mj_nablaWij[int(i)].deactivate()
            self.neighbour_num[int(i)] = 0
        for p_i in range(self.tn):
            pos = self.pos(p_i) - self.center + self.dimensions / 2
            x_id = int(pos[0] / self.grid_size)
            y_id = int(pos[1] / self.grid_size)
            z_id = int(pos[2] / self.grid_size)
            if 0 <= x_id < self.grid_nums[0] and 0 <= y_id < self.grid_nums[1] and 0 <= z_id < self.grid_nums[2]:
                self.grid[x_id, y_id, z_id].append(p_i)
                self.grid_num[x_id, y_id, z_id] += 1
                self.idx_to_grid[p_i] = ti.Vector([x_id, y_id, z_id])
        for p_i in range(self.fn):
            grid_idx = self.idx_to_grid[p_i]
            ret = 0.0
            st = ti.Vector([0.0, 0.0, 0.0])
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0, 0.0, 0.0])
            for offset in ti.grouped(ti.ndrange((-2, 3), (-2, 3), (-2, 3))):
                n_id = grid_idx + offset
                if 0 <= n_id[0] < self.grid_nums[0] and 0 <= n_id[1] < self.grid_nums[1] and 0 <= n_id[2] < self.grid_nums[2]:
                    for j in range(self.grid_num[n_id]):
                        p_j = self.grid[n_id,j]
                        r = self.fluid.positions[p_i] - self.pos(p_j)
                        r_len = r.norm()
                        if r_len <= self.h:
                            # self.neighbour[p_i].append(p_j)
                            self.neighbour_num[p_i] += 1
                            mj_nij = self.p_mass * self.fluid.kernel_grad(r)
                            self.mj_nablaWij[p_i].append([mj_nij[0], mj_nij[1], mj_nij[2], p_j])
                            ret += self.p_mass * self.fluid.kernel_func(r_len)
                            grad_p_i += mj_nij
                            if p_j < self.fn:
                                st += self.fluid.surface_tension * r * self.fluid.kernel_func(max(r_len, self.fluid.particle_diameter))
                                sum_grad_p_k += mj_nij.norm_sqr()
            self.fluid.densities[p_i] = ret
            self.surf_tens[p_i] = st * self.p_mass
            sum_grad_p_k += grad_p_i.norm_sqr()
            self.alpha[p_i] = self.fluid.densities[p_i] / sum_grad_p_k if sum_grad_p_k > 1e-5 else 0.0
        
    @ti.kernel
    def compute_density_derivative(self):
        """
        compute (D rho / Dt) / rho_0 for each particle
        """
        for i in range(self.fn):
            if self.neighbour_num[i] < 20:
                self.rho_derivative[i] = 0.0
                continue
            ret = 0.0
            vi = self.fluid.velocities[i]
            for j in range(self.neighbour_num[i]):
                mn_pj = self.mj_nablaWij[i, j]
                mn = mn_pj[:3]
                pj = int(mn_pj[3])
                ret += ti.math.dot(vi - self.vel(pj), mn)
                
            # self.rho_derivative[i] = ti.max(ret / self.fluid.rest_density, 0.0)
            self.rho_derivative[i] = ret / self.fluid.rest_density

    @ti.kernel
    def compute_k_divergence(self):
        for i in range(self.fn):
            self.kappa[i] = ti.max(self.rho_derivative[i], 0.0)
            # self.k_a_over_rho[i] = self.kappa[i] * self.alpha[i] / self.fluid.densities[i]

    @ti.kernel
    def compute_k_density(self):
        for i in range(self.fn):
            self.kappa[i] = ti.max(self.rho_derivative[i] + (self.fluid.densities[i] / self.fluid.rest_density - 1) / self.fluid.time_step, 0.0)
            # self.k_a_over_rho[i] = self.kappa[i] * self.alpha[i] / self.fluid.densities[i]
            # k/den = max(((rho_i - restrho)/restrho * t + Drho/Dt), 0) * alpha / den
    
    @ti.kernel
    def update_vel_according_to_k(self):
        for i in range(self.fn):
            self.k_a_over_rho[i] = self.kappa[i] * self.alpha[i] / self.fluid.densities[i]
            self.max_k[None] = max(self.max_k[None], self.kappa[i])
        for i in range(self.fn):
            ret = ti.Vector([0.0, 0.0, 0.0])
            ka_rhoi = self.k_a_over_rho[i]
            for j in range(self.neighbour_num[i]):
                mn_pj = self.mj_nablaWij[i, j]
                mn = mn_pj[:3]
                pj = int(mn_pj[3])
                tmp = mn * ka_rhoi
                if pj < self.fn:
                    ret += tmp + mn * self.k_a_over_rho[pj]
                else:
                    ret += tmp
                    # print(mn, ka_rhoi)
                    pj -= self.fn
                    for ind in ti.static(range(self.nr)):
                        if self.before_sizes[ind] <= pj < self.before_sizes[ind + 1]:
                            self.rigids[ind].apply_force(tmp * self.p_mass / self.fluid.time_step, self.rigid_positions[pj])
            self.fluid.velocities[i] -= ret

    @ti.kernel
    def iteration_error_k(self) -> ti.f32:
        error = 0.0
        for i in range(self.fn):
            error += self.kappa[i] * self.fluid.time_step
        return error / self.fn

    def DFSPH_solver_template(self, compute_k, max_iter, max_err):
        self.compute_density_derivative()
        compute_k()
        # self.max_k[None] = 0.0
        for _ in range(max(1, max_iter)):
            self.update_vel_according_to_k()
            self.compute_density_derivative()
            compute_k()
            if self.iteration_error_k() <= max_err:
                break
        # print(f"self.max_k[None]: {self.max_k[None]}")
    
    @ti.kernel
    def compute_non_pressure_forces(self):
        for i in range(self.fn):
            ret = ti.Vector([0.0, 0.0, 0.0])
            v_i = self.fluid.velocities[i]
            pos_i = self.fluid.positions[i]
            # den_i = self.fluid.densities[i]
            visc_i = 10 * self.fluid.viscosity[i] / self.fluid.densities[i] * self.p_mass
            for j in range(self.neighbour_num[i]):
                mn_pj = self.mj_nablaWij[i, j]
                p_j = int(mn_pj[3])
                mn = mn_pj[:3]
                r = pos_i - self.pos(p_j)
                v_xy = ti.math.dot(v_i - self.vel(p_j), r)
                # viscosity_force = 2 * 5 * visc_i * mn / den_i / (r.norm_sqr() + 0.01 * self.h ** 2) * v_xy
                viscosity_force = visc_i * mn / (max(r.norm(), self.fluid.particle_diameter)**2) * v_xy
                ret += viscosity_force
                if p_j >= self.fn:
                    p_j -= self.fn
                    # force_j = - viscosity_force * self.fluid.mass[i]
                    pos_j = self.rigid_positions[p_j]
                    for ind in ti.static(range(self.nr)):
                        if self.before_sizes[ind] <= p_j < self.before_sizes[ind + 1]:
                            self.rigids[ind].apply_force(-viscosity_force, pos_j)
            self.fluid.forces[i] = (ret - self.surf_tens[i])
            # self.max_non_gravity_force[None] = max(self.fluid.forces[i].norm(), self.max_non_gravity_force[None])

    @ti.func
    def out_prod(self, a, b):
        ret = ti.Matrix.zero(ti.f32, 3, 3)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                ret[i, j] = a[i] * b[j]
        return ret
    
    @ti.func
    def mat3norm2(self, x):
        ret = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            ret += x[i, j] * x[i, j]
        return ti.sqrt(ret / 2)

    @ti.kernel
    def update_viscosity(self):
        # avg_shear = 0.0
        for i in range(self.fn):
            # self.fluid.viscosity[i] = 0.0
            strain = ti.Matrix([[0.0] * 3] * 3)
            vi = self.fluid.velocities[i]
            for j in range(self.neighbour_num[i]):
                pj = int(self.mj_nablaWij[i, j][3])
                mn = self.mj_nablaWij[i, j][:3]
                if pj < self.fn:
                    vij = vi - self.fluid.velocities[pj]
                    strain += self.out_prod(vij, mn) / self.fluid.densities[pj]
                    # strain += self.mn_vij[i, j]
            # strain = (strain.transpose() + strain) / 2
            shear = self.mat3norm2(strain + strain.transpose())
            self.fluid.viscosity[i] = visc_from_shear(shear)
            # avg_shear += shear
        # print("avg_shear:", avg_shear / self.fn)
    
    def test_v(self):
        pass
        
    
    def update(self):
        self.get_rigid_pos()
        # self.compute_mn_vij()
        self.update_viscosity()
        self.compute_non_pressure_forces()
        self.fluid.update_velocity()
        # self.density_solver() 
        self.DFSPH_solver_template(self.compute_k_density, self.den_max_it, self.den_max_err)
        
        self.fluid.update_position()
        self.set_boundary_conditions()
        
        self.compute_N_rho_alpha()
        # self.divergence_solver()
        self.DFSPH_solver_template(self.compute_k_divergence, self.div_max_it, self.div_max_err)
