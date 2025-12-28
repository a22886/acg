import bpy, sys, os, time
from main import fluid_obj_prefix, rigids_obj_prefix, output_png_prefix, has_fluid, has_rigids


def add_mesh(path, material):
    bpy.ops.wm.obj_import(filepath=path)
    mesh = bpy.context.selected_objects[0]
    if mesh.data.materials:
        mesh.data.materials[0] = material
    else:
        mesh.data.materials.append(material)
    mesh.rotation_euler = (0, 0, 0)


def proc_mesh(i: int):
    stdout = os.dup(1)
    os.close(1)
    os.open(os.devnull, os.O_WRONLY)

    if has_fluid:
        add_mesh(fluid_obj_prefix + str(i) + ".obj", bpy.data.materials.get("Fluid"))
    for j in range(len(has_rigids)):
        if has_rigids[j]:
            add_mesh(rigids_obj_prefix[j] + str(i) + ".obj", bpy.data.materials.get("Rigid"))

    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = output_png_prefix + str(i)
    bpy.ops.render.render(write_still=True)
    os.close(1)
    os.dup(stdout)
    os.close(stdout)
    print(f"Saved {bpy.context.scene.render.filepath}.png at {time.strftime('%H:%M:%S')}")

    for obj in bpy.data.objects:
        if obj.type == "MESH":
            bpy.data.meshes.remove(obj.data)


def init_bpy():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Create a new camera
    bpy.ops.object.camera_add(align="WORLD", location=[-2, 4, 8], rotation=(-0.471, -0.227, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    # Set background color
    bpy.context.scene.world.use_nodes = True
    world = bpy.context.scene.world
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    for node in nodes:
        nodes.remove(node)

    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.location = (200, 0)
    bg_node.inputs["Color"].default_value = (0, 0, 0, 1)

    output_node = nodes.new(type="ShaderNodeOutputWorld")
    output_node.location = (400, 0)

    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

    bg_node.inputs["Color"].default_value = (0.52, 0.8, 0.98, 1)  # RGBA (Blueish tone)
    bg_node.inputs["Strength"].default_value = 1.0  # Set the strength

    # Create a new light source
    bpy.ops.object.light_add(type="POINT", location=(5, 7, 5))
    light = bpy.context.object
    light.data.energy = 1000

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # or 'OPENCL' depending on your GPU
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for device in bpy.context.preferences.addons["cycles"].preferences.devices:
        device.use = True

    fluid_material = bpy.data.materials.new(name="Fluid")
    fluid_material.use_nodes = True
    nodes = fluid_material.node_tree.nodes
    links = fluid_material.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    noise_tex = nodes.new(type="ShaderNodeTexNoise")
    noise_tex.location = (-400, 0)
    noise_tex.inputs["Scale"].default_value = 10

    bump = nodes.new(type="ShaderNodeBump")
    bump.location = (-200, 0)
    bump.inputs["Strength"].default_value = 0.3

    links.new(noise_tex.outputs["Fac"], bump.inputs["Height"])

    # Connect Bump node to BSDF Normal input
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    if bsdf:
        bsdf.inputs["Roughness"].default_value = 0.1
        bsdf.inputs["Transmission Weight"].default_value = 0.3
        bsdf.inputs["IOR"].default_value = 1.33
        # bsdf.inputs['Clearcoat'].default_value = 1.0
        bsdf.inputs["Base Color"].default_value = (0.1, 0.3, 0.9, 1)
        bsdf.inputs["Alpha"].default_value = 0.7  # Alpha value = 0.5: half transparent
        fluid_material.blend_method = "BLEND"  # Set blend method
        fluid_material.shadow_method = "HASHED"  # Set shadow method
    else:
        pass

    rigid_material = bpy.data.materials.new(name="Rigid")
    rigid_material.use_nodes = True
    bsdf = rigid_material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.486, 0.988, 0.0, 1.0)  # RGBA for yellow


init_bpy()
for j in range(int(sys.argv[1]), int(sys.argv[2])):
    proc_mesh(j)
