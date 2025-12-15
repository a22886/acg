import bpy
import math
import sys
import os
import time
def add_fluid(i: int, path):
    if not os.path.exists(path):
        return None
    bpy.ops.wm.obj_import(filepath=path)
    fluid_mesh = bpy.context.selected_objects[0]
    fluid_material = bpy.data.materials.get("Fluid")
    if fluid_mesh.data.materials:
        fluid_mesh.data.materials[0] = fluid_material
    else:
        fluid_mesh.data.materials.append(fluid_material)
        
    # Set the alpha value of the fluid material
    
    fluid_mesh.rotation_euler = (0, 0, 0) # ensure (x,y,z) equals (x,y,z) in blender
    return fluid_mesh
def add_rigid(i: int, path):
    if not os.path.exists(path):
        return None
    bpy.ops.wm.obj_import(filepath=path)
    rigid_mesh = bpy.context.selected_objects[0]    
    rigid_mesh.rotation_euler = (0, 0, 0) # ensure (x,y,z) equals (x,y,z) in blender
    return rigid_mesh

def proc_mesh(i: int):
    # bpy.ops.object.select_all(action='DESELECT')
    # for obj in bpy.data.objects:
    #     if obj.type == 'MESH':
    #         obj.select_set(True)
    # bpy.ops.object.delete()
    stdout = os.dup(1)
    os.close(1)
    os.open(os.devnull, os.O_WRONLY)

    fluid_mesh = add_fluid(i, sys.argv[4]+str(i)+".obj")
    rigid_mesh = add_rigid(i, sys.argv[5]+str(i)+".obj")
    rigid_mesh2 = add_rigid(i, sys.argv[6]+str(i)+".obj")
    
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = sys.argv[3]+str(i)
    bpy.ops.render.render(write_still=True)
    os.close(1)
    os.dup(stdout)
    os.close(stdout)
    print(f"Saved {bpy.context.scene.render.filepath}.png at {time.strftime('%H:%M:%S')}")

    if fluid_mesh is not None:
        bpy.data.meshes.remove(fluid_mesh.data)
    if rigid_mesh is not None:
        bpy.data.meshes.remove(rigid_mesh.data)
    if rigid_mesh2 is not None:
        bpy.data.meshes.remove(rigid_mesh2.data)
    # fluid_mesh.select_set(True)
    # bpy.ops.object.delete()

def init_bpy():
    # Renderer = render.Render()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Create a new camera
    bpy.ops.object.camera_add(align='WORLD', location=[-3+0.87-0.4-0.433*2, 3.8-0.9+0.5*2-0.2, 1-1.5+0.75*2], rotation=(math.radians(-30),math.radians(-30),0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    
    # Set background color
    bpy.context.scene.world.use_nodes = True
    world = bpy.context.scene.world
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add Background node
    bg_node = nodes.new(type='ShaderNodeBackground')
    bg_node.location = (200, 0)
    bg_node.inputs['Color'].default_value = (0,0,0,1)

    # Add Environment Texture node
    # env_texture_node = nodes.new(type='ShaderNodeTexEnvironment')
    # env_texture_node.location = (-200, 0)
    # env_texture_node.image = bpy.data.images.load('background.hdr')
    
    # Add Output node
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    output_node.location = (400, 0)
    
    # Link nodes
    # links.new(env_texture_node.outputs['Color'], bg_node.inputs['Color'])
    links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    bg_node.inputs["Color"].default_value = (0.52, 0.8, 0.98, 1)  # RGBA (Blueish tone)
    bg_node.inputs["Strength"].default_value = 1.0  # Set the strength

    # Create a new light source
    bpy.ops.object.light_add(type='POINT', location=(4,0,4))
    light = bpy.context.object
    light.data.energy = 2000

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # or 'OPENCL' depending on your GPU
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        device.use = True

    fluid_material = bpy.data.materials.new(name="Fluid")
    fluid_material.use_nodes = True
    nodes = fluid_material.node_tree.nodes
    links = fluid_material.node_tree.links
    bsdf = nodes.get('Principled BSDF')

    noise_tex = nodes.new(type='ShaderNodeTexNoise')
    noise_tex.location = (-400, 0)
    noise_tex.inputs['Scale'].default_value = 10 
    
    bump = nodes.new(type='ShaderNodeBump')
    bump.location = (-200, 0)
    bump.inputs['Strength'].default_value = 0.3
    
    links.new(noise_tex.outputs['Fac'], bump.inputs['Height'])

    # Connect Bump node to BSDF Normal input
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    if bsdf:
        bsdf.inputs['Roughness'].default_value = 0.1
        bsdf.inputs['Transmission Weight'].default_value = 0.3
        bsdf.inputs['IOR'].default_value = 1.33
        # bsdf.inputs['Clearcoat'].default_value = 1.0
        bsdf.inputs['Base Color'].default_value = (0.1, 0.3, 0.9, 1)
        bsdf.inputs['Alpha'].default_value = 0.7  # Alpha value = 0.5: half transparent
        fluid_material.blend_method = 'BLEND'     # Set blend method
        fluid_material.shadow_method = 'HASHED'    # Set shadow method
    else:
        pass

def main():       
    init_bpy()
    for j in range(int(sys.argv[1]), int(sys.argv[2])):
        proc_mesh(j)
    
if __name__ == '__main__':
    main()
