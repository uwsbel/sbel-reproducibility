import bpy
import math
import random
import sys

jobid = int(sys.argv[4])
start_frame = jobid*1 + 1
end_frame = jobid*1 + 2

for k in range(start_frame,end_frame,1):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.objects.keys()

    root_dir = "/srv/home/whu59/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1001/chrono-dev-io-test-random-motion/curiosity-523-fewSPH-C-Wi=0.8/Animation_Blender/"

    file_fluid = root_dir + "granular_file/"
    file_mesh = root_dir + "mesh_file/"

    file_loc = file_fluid + 'fluid'+str(k)+'.obj'
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
    obj_object = bpy.context.object
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    for j in range(1, 16, 1):
        file_loc_j = file_mesh + 'Body_' + str(j) + '_' + str(k) + '.obj'
        imported_object_0 = bpy.ops.import_scene.obj(filepath=file_loc_j)
        bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    #ob = bpy.data.objects[0]
    #mat = bpy.data.materials.new(name="Material")
    #mat.diffuse_color = (1, 0, 0, 0)
    #ob.data.materials.append(mat)
    #ob.active_material = mat

    bpy.ops.mesh.primitive_plane_add(size=200.0, calc_uvs=True, enter_editmode=False, align='WORLD',
                                     location=(0.0, 0.0, 0.0))

    # bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=(-8.699, -11.285, 4.6723),
    #                       rotation=(1.2548917, 0.0139800873, -0.6300638), scale=(5.0, 5.0, 5.0))
    bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=(8.199*1.3, -11.285*1.3, 4.2723*1.56),
                              rotation=(1.2548917, -0.0139800873, 0.6300638), scale=(5.0, 5.0, 5.0))
    scene.camera = bpy.context.object
    scene.cycles.device = 'GPU'

    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences

    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            break
        except TypeError:
            pass

    # Enable all CPU and GPU devices
    cprefs.get_devices()
    for device in cprefs.devices:
        device.use = True

    # create light datablock, set attributes
    light_data = bpy.data.lights.new(name="light_2.80", type='POINT')
    light_data.energy = 13000

    # create new object with our light datablock
    light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)

    # link light object
    bpy.context.collection.objects.link(light_object)

    # make it active
    bpy.context.view_layer.objects.active = light_object

    # change location
    light_object.location = (10, 10, 15)

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    # bpy.context.scene.render.resolution_percentage = 200
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.render.resolution_x = 3840
    bpy.context.scene.render.resolution_y = 2160
    bpy.context.scene.render.filepath = root_dir + "image_file/"+str(k)+".png"
    # bpy.context.scene.render.image_settings.compression = 50
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
