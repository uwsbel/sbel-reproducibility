import bpy
import math
import random

start_frame = 0
end_frame = 401

for k in range(start_frame,end_frame,1):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.objects.keys()

    root_dir = "/srv/home/whu59/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1001/chrono-dev-io-Viper/18/Rendering/"

    file_fluid = root_dir + "granular_file/"
    file_mesh = root_dir + "mesh_file/"

    file_loc = file_fluid + 'fluid'+str(k)+'.obj'
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
    obj_object = bpy.context.object
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_0 = file_mesh + 'body_'+str(k)+'.obj'
    imported_object_0 = bpy.ops.import_scene.obj(filepath=file_loc_0)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_1 = file_mesh + 'wheel_1_'+str(k)+'.obj'
    imported_object_1 = bpy.ops.import_scene.obj(filepath=file_loc_1)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_2 = file_mesh + 'wheel_2_'+str(k)+'.obj'
    imported_object_2 = bpy.ops.import_scene.obj(filepath=file_loc_2)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_3 = file_mesh + 'wheel_3_'+str(k)+'.obj'
    imported_object_3 = bpy.ops.import_scene.obj(filepath=file_loc_3)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_4 = file_mesh + 'wheel_4_'+str(k)+'.obj'
    imported_object_4 = bpy.ops.import_scene.obj(filepath=file_loc_4)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_5 = file_mesh + 'lowerRod_1_'+str(k)+'.obj'
    imported_object_5 = bpy.ops.import_scene.obj(filepath=file_loc_5)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_6 = file_mesh + 'lowerRod_2_'+str(k)+'.obj'
    imported_object_6 = bpy.ops.import_scene.obj(filepath=file_loc_6)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_7 = file_mesh + 'lowerRod_3_'+str(k)+'.obj'
    imported_object_7 = bpy.ops.import_scene.obj(filepath=file_loc_7)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_8 = file_mesh + 'lowerRod_4_'+str(k)+'.obj'
    imported_object_8 = bpy.ops.import_scene.obj(filepath=file_loc_8)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_5 = file_mesh + 'upperRod_1_'+str(k)+'.obj'
    imported_object_5 = bpy.ops.import_scene.obj(filepath=file_loc_5)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_6 = file_mesh + 'upperRod_2_'+str(k)+'.obj'
    imported_object_6 = bpy.ops.import_scene.obj(filepath=file_loc_6)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_7 = file_mesh + 'upperRod_3_'+str(k)+'.obj'
    imported_object_7 = bpy.ops.import_scene.obj(filepath=file_loc_7)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_8 = file_mesh + 'upperRod_4_'+str(k)+'.obj'
    imported_object_8 = bpy.ops.import_scene.obj(filepath=file_loc_8)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_5 = file_mesh + 'steerRod_1_'+str(k)+'.obj'
    imported_object_5 = bpy.ops.import_scene.obj(filepath=file_loc_5)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_6 = file_mesh + 'steerRod_2_'+str(k)+'.obj'
    imported_object_6 = bpy.ops.import_scene.obj(filepath=file_loc_6)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_7 = file_mesh + 'steerRod_3_'+str(k)+'.obj'
    imported_object_7 = bpy.ops.import_scene.obj(filepath=file_loc_7)
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    file_loc_8 = file_mesh + 'steerRod_4_'+str(k)+'.obj'
    imported_object_8 = bpy.ops.import_scene.obj(filepath=file_loc_8)
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
    bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=(-8.699*1.0, -11.285*1.0, 4.6723*1.2),
                          rotation=(1.2548917, 0.0139800873, -0.6300638), scale=(5.0, 5.0, 5.0))
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
