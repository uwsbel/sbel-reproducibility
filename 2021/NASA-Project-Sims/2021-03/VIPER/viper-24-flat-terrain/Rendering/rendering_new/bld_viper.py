import bpy
import math
import random
import csv
import sys
time = []

# index number:
# 0 - body
# 1 - L sus 1
# 2 - L sus 2
# 3 - L sus 3
# 4 - L sus 4
# 5 - U sus 1
# 6 - U sus 2
# 7 - U sus 3
# 8 - U sus 4
# 9 - TI 1
# 10 - TI 2
# 11 - TI 3
# 12 - TI 4
# 13 - Wheel 1
# 14 - Wheel 2
# 15 - Wheel 3
# 16 - Wheel 4
dis = []
rot = []
home_root = "/srv/home/whu59/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1001/chrono-dev-io-test-random-motion/viper-24-C-Wi=0.8/"
data_in = home_root + "DEMO_OUTPUT/FSI_Viper/Rover/"
file_in = home_root + "Rendering/rendering_new/"
for i in range(17):
    with open(data_in + "body_pos_rot_vel"+str(i+1)+".csv", 'r') as file:
        dis_temp = []
        rot_temp = []
        reader = csv.reader(file)
        i = 0
        for row in reader:
            i=i+1
            if i!=1:
                time.append(float(row[0]))
                dis_buff = []
                dis_buff.append(float(row[1]))
                dis_buff.append(float(row[2]))
                dis_buff.append(float(row[3]))
                rot_buff = []
                rot_buff.append(float(row[4]))
                rot_buff.append(float(row[5]))
                rot_buff.append(float(row[6]))
                rot_buff.append(float(row[7]))
                dis_temp.append(dis_buff)
                rot_temp.append(rot_buff)

        dis.append(dis_temp)
        rot.append(rot_temp)

base_loc = []
base_loc.append(dis[0][0][0])
base_loc.append(dis[0][0][1])
base_loc.append(dis[0][0][2])


jobid = int(sys.argv[4])
start_frame = jobid*1 + 1
end_frame = jobid*1 + 2
for i in range(start_frame, end_frame, 1):

    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.objects.keys()

    file_loc_0 = file_in + 'body_new_test.obj'
    imported_object_0 = bpy.ops.import_scene.obj(filepath=file_loc_0)
    obj_object_0 = bpy.context.object
    bpy.data.objects["body"].location.x += dis[0][i][0]
    bpy.data.objects["body"].location.y += dis[0][i][1]
    bpy.data.objects["body"].location.z += dis[0][i][2]  
    bpy.data.objects["body"].rotation_mode = 'QUATERNION'
    q_0 = (rot[0][i][0],rot[0][i][1],rot[0][i][2],rot[0][i][3])
    bpy.data.objects["body"].rotation_quaternion = q_0

    file_loc_1 = file_in + 'wheelSimplified.obj'
    imported_object_1 = bpy.ops.import_scene.obj(filepath=file_loc_1)
    obj_object_1 = bpy.context.object
    bpy.data.objects["wheelSimplified"].location.x += dis[1][i][0]
    bpy.data.objects["wheelSimplified"].location.y += dis[1][i][1]
    bpy.data.objects["wheelSimplified"].location.z += dis[1][i][2]  
    bpy.data.objects["wheelSimplified"].rotation_mode = 'QUATERNION'
    q_1 = (rot[1][i][0],rot[1][i][1],rot[1][i][2],rot[1][i][3])
    bpy.data.objects["wheelSimplified"].rotation_quaternion = q_1

    file_loc_2 = file_in + 'wheelSimplified.obj'
    imported_object_2 = bpy.ops.import_scene.obj(filepath=file_loc_2)
    obj_object_2 = bpy.context.object
    bpy.data.objects["wheelSimplified.001"].location.x += dis[2][i][0]
    bpy.data.objects["wheelSimplified.001"].location.y += dis[2][i][1]
    bpy.data.objects["wheelSimplified.001"].location.z += dis[2][i][2]  
    bpy.data.objects["wheelSimplified.001"].rotation_mode = 'QUATERNION'
    q_2 = (rot[2][i][0],rot[2][i][1],rot[2][i][2],rot[2][i][3])
    bpy.data.objects["wheelSimplified.001"].rotation_quaternion = q_2

    file_loc_3 = file_in + 'wheelSimplified.obj'
    imported_object_3 = bpy.ops.import_scene.obj(filepath=file_loc_3)
    obj_object_3 = bpy.context.object
    bpy.data.objects["wheelSimplified.002"].location.x += dis[3][i][0]
    bpy.data.objects["wheelSimplified.002"].location.y += dis[3][i][1]
    bpy.data.objects["wheelSimplified.002"].location.z += dis[3][i][2]  
    bpy.data.objects["wheelSimplified.002"].rotation_mode = 'QUATERNION'
    q_3 = (rot[3][i][0],rot[3][i][1],rot[3][i][2],rot[3][i][3])
    bpy.data.objects["wheelSimplified.002"].rotation_quaternion = q_3

    file_loc_4 = file_in + 'wheelSimplified.obj'
    imported_object_4 = bpy.ops.import_scene.obj(filepath=file_loc_4)
    obj_object_4 = bpy.context.object
    bpy.data.objects["wheelSimplified.003"].location.x += dis[4][i][0]
    bpy.data.objects["wheelSimplified.003"].location.y += dis[4][i][1]
    bpy.data.objects["wheelSimplified.003"].location.z += dis[4][i][2]  
    bpy.data.objects["wheelSimplified.003"].rotation_mode = 'QUATERNION'
    q_4 = (rot[4][i][0],rot[4][i][1],rot[4][i][2],rot[4][i][3])
    bpy.data.objects["wheelSimplified.003"].rotation_quaternion = q_4

    file_loc_5 = file_in + 'lowerRod_new.obj'
    imported_object_5 = bpy.ops.import_scene.obj(filepath=file_loc_5)
    obj_object_5 = bpy.context.object
    bpy.data.objects["lowerRod"].location.x += dis[5][i][0]
    bpy.data.objects["lowerRod"].location.y += dis[5][i][1]
    bpy.data.objects["lowerRod"].location.z += dis[5][i][2]  
    bpy.data.objects["lowerRod"].rotation_mode = 'QUATERNION'
    q_5 = (rot[5][i][0],rot[5][i][1],rot[5][i][2],rot[5][i][3])
    bpy.data.objects["lowerRod"].rotation_quaternion = q_5

    file_loc_6 = file_in + 'lowerRod_new.obj'
    imported_object_6 = bpy.ops.import_scene.obj(filepath=file_loc_6)
    obj_object_6 = bpy.context.object
    bpy.data.objects["lowerRod.001"].location.x += dis[6][i][0]
    bpy.data.objects["lowerRod.001"].location.y += dis[6][i][1]
    bpy.data.objects["lowerRod.001"].location.z += dis[6][i][2]  
    bpy.data.objects["lowerRod.001"].rotation_mode = 'QUATERNION'
    q_6 = (rot[6][i][0],rot[6][i][1],rot[6][i][2],rot[6][i][3])
    bpy.data.objects["lowerRod.001"].rotation_quaternion = q_6

    file_loc_7 = file_in + 'lowerRod_new.obj'
    imported_object_7 = bpy.ops.import_scene.obj(filepath=file_loc_7)
    obj_object_7 = bpy.context.object
    bpy.data.objects["lowerRod.002"].location.x += dis[7][i][0]
    bpy.data.objects["lowerRod.002"].location.y += dis[7][i][1]
    bpy.data.objects["lowerRod.002"].location.z += dis[7][i][2]  
    bpy.data.objects["lowerRod.002"].rotation_mode = 'QUATERNION'
    q_7 = (rot[7][i][0],rot[7][i][1],rot[7][i][2],rot[7][i][3])
    bpy.data.objects["lowerRod.002"].rotation_quaternion = q_7

    file_loc_8 = file_in + 'lowerRod_new.obj'
    imported_object_8 = bpy.ops.import_scene.obj(filepath=file_loc_8)
    obj_object_8 = bpy.context.object
    bpy.data.objects["lowerRod.003"].location.x += dis[8][i][0]
    bpy.data.objects["lowerRod.003"].location.y += dis[8][i][1]
    bpy.data.objects["lowerRod.003"].location.z += dis[8][i][2]  
    bpy.data.objects["lowerRod.003"].rotation_mode = 'QUATERNION'
    q_8 = (rot[8][i][0],rot[8][i][1],rot[8][i][2],rot[8][i][3])
    bpy.data.objects["lowerRod.003"].rotation_quaternion = q_8

    file_loc_9 = file_in + 'upperRod_new.obj'
    imported_object_9 = bpy.ops.import_scene.obj(filepath=file_loc_9)
    obj_object_9 = bpy.context.object
    bpy.data.objects["upperRod"].location.x += dis[9][i][0]
    bpy.data.objects["upperRod"].location.y += dis[9][i][1]
    bpy.data.objects["upperRod"].location.z += dis[9][i][2]  
    bpy.data.objects["upperRod"].rotation_mode = 'QUATERNION'
    q_9 = (rot[9][i][0],rot[9][i][1],rot[9][i][2],rot[9][i][3])
    bpy.data.objects["upperRod"].rotation_quaternion = q_9

    file_loc_10 = file_in + 'upperRod_new.obj'
    imported_object_10 = bpy.ops.import_scene.obj(filepath=file_loc_10)
    obj_object_10 = bpy.context.object
    bpy.data.objects["upperRod.001"].location.x += dis[10][i][0]
    bpy.data.objects["upperRod.001"].location.y += dis[10][i][1]
    bpy.data.objects["upperRod.001"].location.z += dis[10][i][2]  
    bpy.data.objects["upperRod.001"].rotation_mode = 'QUATERNION'
    q_10 = (rot[10][i][0],rot[10][i][1],rot[10][i][2],rot[10][i][3])
    bpy.data.objects["upperRod.001"].rotation_quaternion = q_10

    file_loc_11 = file_in + 'upperRod_new.obj'
    imported_object_11 = bpy.ops.import_scene.obj(filepath=file_loc_11)
    obj_object_11 = bpy.context.object
    bpy.data.objects["upperRod.002"].location.x += dis[11][i][0]
    bpy.data.objects["upperRod.002"].location.y += dis[11][i][1]
    bpy.data.objects["upperRod.002"].location.z += dis[11][i][2]  
    bpy.data.objects["upperRod.002"].rotation_mode = 'QUATERNION'
    q_11 = (rot[11][i][0],rot[11][i][1],rot[11][i][2],rot[11][i][3])
    bpy.data.objects["upperRod.002"].rotation_quaternion = q_11

    file_loc_12 = file_in + 'upperRod_new.obj'
    imported_object_12 = bpy.ops.import_scene.obj(filepath=file_loc_12)
    obj_object_12 = bpy.context.object
    bpy.data.objects["upperRod.003"].location.x += dis[12][i][0]
    bpy.data.objects["upperRod.003"].location.y += dis[12][i][1]
    bpy.data.objects["upperRod.003"].location.z += dis[12][i][2]  
    bpy.data.objects["upperRod.003"].rotation_mode = 'QUATERNION'
    q_12 = (rot[12][i][0],rot[12][i][1],rot[12][i][2],rot[12][i][3])
    bpy.data.objects["upperRod.003"].rotation_quaternion = q_12

    file_loc_13 = file_in + 'steeringRod.obj'
    imported_object_13 = bpy.ops.import_scene.obj(filepath=file_loc_13)
    obj_object_13 = bpy.context.object
    bpy.data.objects["steeringRod"].location.x += dis[13][i][0]
    bpy.data.objects["steeringRod"].location.y += dis[13][i][1]
    bpy.data.objects["steeringRod"].location.z += dis[13][i][2]  
    bpy.data.objects["steeringRod"].rotation_mode = 'QUATERNION'
    q_13 = (rot[13][i][0],rot[13][i][1],rot[13][i][2],rot[13][i][3])
    bpy.data.objects["steeringRod"].rotation_quaternion = q_13

    file_loc_14 = file_in + 'steeringRod.obj'
    imported_object_14 = bpy.ops.import_scene.obj(filepath=file_loc_14)
    obj_object_14 = bpy.context.object
    bpy.data.objects["steeringRod.001"].location.x += dis[14][i][0]
    bpy.data.objects["steeringRod.001"].location.y += dis[14][i][1]
    bpy.data.objects["steeringRod.001"].location.z += dis[14][i][2]  
    bpy.data.objects["steeringRod.001"].rotation_mode = 'QUATERNION'
    q_14 = (rot[14][i][0],rot[14][i][1],rot[14][i][2],rot[14][i][3])
    bpy.data.objects["steeringRod.001"].rotation_quaternion = q_14

    file_loc_15 = file_in + 'steeringRod.obj'
    imported_object_15 = bpy.ops.import_scene.obj(filepath=file_loc_15)
    obj_object_15 = bpy.context.object
    bpy.data.objects["steeringRod.002"].location.x += dis[15][i][0]
    bpy.data.objects["steeringRod.002"].location.y += dis[15][i][1]
    bpy.data.objects["steeringRod.002"].location.z += dis[15][i][2]  
    bpy.data.objects["steeringRod.002"].rotation_mode = 'QUATERNION'
    q_15 = (rot[15][i][0],rot[15][i][1],rot[15][i][2],rot[15][i][3])
    bpy.data.objects["steeringRod.002"].rotation_quaternion = q_15

    file_loc_16 = file_in + 'steeringRod.obj'
    imported_object_16 = bpy.ops.import_scene.obj(filepath=file_loc_16)
    obj_object_16 = bpy.context.object
    bpy.data.objects["steeringRod.003"].location.x += dis[16][i][0]
    bpy.data.objects["steeringRod.003"].location.y += dis[16][i][1]
    bpy.data.objects["steeringRod.003"].location.z += dis[16][i][2]  
    bpy.data.objects["steeringRod.003"].rotation_mode = 'QUATERNION'
    q_16 = (rot[16][i][0],rot[16][i][1],rot[16][i][2],rot[16][i][3])
    bpy.data.objects["steeringRod.003"].rotation_quaternion = q_16

    # file_loc_17 = file_in + 'obstacle.obj'
    # imported_object_17 = bpy.ops.import_scene.obj(filepath=file_loc_17)
    # obj_object_17 = bpy.context.object
    # bpy.data.objects["obstacle"].location.x += dis[17][i][0]
    # bpy.data.objects["obstacle"].location.y += dis[17][i][1]
    # bpy.data.objects["obstacle"].location.z += dis[17][i][2]  
    # bpy.data.objects["obstacle"].rotation_mode = 'QUATERNION'
    # q_17 = (rot[17][i][0],rot[17][i][1],rot[17][i][2],rot[17][i][3])
    # bpy.data.objects["obstacle"].rotation_quaternion = q_17

    # file_loc_18 = file_in + 'obstacle.obj'
    # imported_object_18 = bpy.ops.import_scene.obj(filepath=file_loc_18)
    # obj_object_18 = bpy.context.object
    # bpy.data.objects["obstacle.001"].location.x += dis[18][i][0]
    # bpy.data.objects["obstacle.001"].location.y += dis[18][i][1]
    # bpy.data.objects["obstacle.001"].location.z += dis[18][i][2]  
    # bpy.data.objects["obstacle.001"].rotation_mode = 'QUATERNION'
    # q_18 = (rot[18][i][0],rot[18][i][1],rot[18][i][2],rot[18][i][3])
    # bpy.data.objects["obstacle.001"].rotation_quaternion = q_18

    terrain_loc = home_root + 'Rendering/granular_file/fluid'+str(i)+'.obj'
    imported_object = bpy.ops.import_scene.obj(filepath=terrain_loc)
    obj_object = bpy.context.object
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle

    #bpy.ops.transform.rotate(value=(math.pi * 0.5), orient_axis='X')  # value = Angle
    bpy.ops.mesh.primitive_plane_add(size=200.0, calc_uvs=True, enter_editmode=False, align='WORLD',
                             location=(0.0, 0.0, 0.0))
    # vehicle back
    #bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=(11.617, 10.797, 8.4221),
    #                    rotation=(1.06988683, 0.0130201562, 2.23402144), scale=(1.0, 1.0, 1.0))

    # test
    camera_dis = []
    camera_dis.append(float(dis[0][i][0]-base_loc[0]))
    camera_dis.append(float(dis[0][i][1]-base_loc[1]))
    camera_dis.append(float(dis[0][i][2]-base_loc[2]))

    # bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=(-5.2157+camera_dis[0], 3.4817+camera_dis[1], 1.9782+camera_dis[2]),
    #                     rotation=(1.4992378275, 0.013648474751, 4.3109632524), scale=(1.0, 1.0, 1.0))
    bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=(-8.899*0.9, -11.285*0.9, 4.6723*1.08),
                              rotation=(1.2548917,  0.0139800873, -0.6300638), scale=(5.0, 5.0, 5.0))
    # bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=( 8.199*1.0, -11.285*1.0, 4.2723*1.2),
    #                           rotation=(1.2548917, -0.0139800873,  0.6300638), scale=(5.0, 5.0, 5.0))

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
    light_object.location = (-10, 10, 15)

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.render.resolution_x = 3840
    bpy.context.scene.render.resolution_y = 2160
    bpy.context.scene.render.filepath = home_root + "Rendering/image_file/fixed_camera/"+str(i)+".png"
    #bpy.context.scene.render.image_settings.compression = 50
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
