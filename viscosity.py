import taichi as ti
@ti.func
def visc_from_shear(shear):
    # Curiously shear is exactly around 1.0
    return ti.min(0.05 * ti.max(1.0, shear ** 4), 0.5) # Non-Newtonian