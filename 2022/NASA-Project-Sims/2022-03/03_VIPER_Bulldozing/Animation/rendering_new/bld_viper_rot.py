import bpy
import math
import random
import mathutils
import csv
import sys
import os.path
time = []

# index number in each rover, 3 rovers in total:
#  0: chassis_body
#  1: wheel_LF_body
#  2: upper_arm_body
#  3: lower_arm_body
#  4: upright_body
#  5: wheel_RF_body
#  6: upper_arm_body
#  7: lower_arm_body
#  8: upright_body
#  9: wheel_LB_body
# 10: upper_arm_body
# 11: lower_arm_body
# 12: upright_body
# 13: wheel_RB_body
# 14: upper_arm_body
# 15: lower_arm_body
# 16: upright_body
# 17: blade
num_body = 18

def point_at(obj, target, roll=0):
    """
    Rotate obj to look at target

    :arg obj: the object to be rotated. Usually the camera
    :arg target: the location (3-tuple or Vector) to be looked at
    :arg roll: The angle of rotation about the axis from obj to target in radians. 
    """
    if not isinstance(target, mathutils.Vector):
        target = mathutils.Vector(target)
    loc = obj.location
    # direction points from the object to the target
    direction = target - loc

    quat = direction.to_track_quat('-Z', 'Y')
    
    # /usr/share/blender/scripts/addons/add_advanced_objects_menu/arrange_on_curve.py
    quat = quat.to_matrix().to_4x4()
    rollMatrix = mathutils.Matrix.Rotation(roll, 4, 'Z')

    # remember the current location, since assigning to obj.matrix_world changes it
    loc = loc.to_tuple()
    #obj.matrix_world = quat * rollMatrix
    # in blender 2.8 and above @ is used to multiply matrices
    # using * still works but results in unexpected behaviour!
    obj.matrix_world = quat @ rollMatrix
    obj.location = loc

def rot_pos(center, radius, angle=0):
    """
    return (x,y,z) points on a circle centered at (center) with a radius of (radius) with an angle of (angle)
    """
    # convert to radian
    angle_pi = angle/180*math.pi
    # generate point on circle
    return (center[0]+radius*math.sin(angle_pi),center[1]+radius*math.cos(angle_pi),center[2])

#============ specify the directory of csv, obj, image, and such
home_root = "/srv/home/whu59/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1008/viper_sim/"
data_in = home_root + "DEMO_OUTPUT/FSI_VIPER/Rover_simple_wheel_75/"
file_in = home_root + "Animation_Blender/Rover_simple_wheel_5X-7X/rendering_new/chrono_viper_data/render/"
image_in = home_root + "Animation_Blender/Rover_simple_wheel_5X-7X/Image_texture_rot_cam/75/"

#============ load position, velocity, rotation of each body at different time
dis = []
rot = []
for i in range(54):
    with open(data_in + "body_pos_rot_vel" + str(i+1) + ".csv", 'r') as file:
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

# base_loc = []
# base_loc.append(dis[0][0][0])
# base_loc.append(dis[0][0][1])
# base_loc.append(dis[0][0][2])


jobid = int(sys.argv[4])
start_frame = jobid*1 + 0
end_frame = jobid*1 + 1
for i in range(start_frame, end_frame, 1):
    #========= check if the png file exits or not
    image_path = image_in + str(i) + ".png"
    file_exists = os.path.exists(image_path)
    if file_exists:
        sys.exit()
        
    #===========================================
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.objects.keys()

    #=========================================== load obj file
    for n_r in range(3):
        obj_name = ""
        obj_name_spe = ""
        for n in range(num_body):
            if n==0:
                obj_name = "viper_chassis"
                obj_name_spe = "viper_chassis"
                if n_r > 0:
                    obj_name_spe = "viper_chassis." + str(n_r).zfill(3)
            if n==1:
                obj_name = "viper_simplewheel"
                obj_name_spe = "viper_simplewheel"
                if n_r > 0:
                    obj_name_spe = "viper_simplewheel." + str(4 * n_r).zfill(3)
            if n==2:
                obj_name = "viper_simplewheel"
                obj_name_spe = "viper_simplewheel." + str(1).zfill(3)
                if n_r > 0:
                    obj_name_spe = "viper_simplewheel." + str(4 * n_r + 1).zfill(3)
            if n==3:
                obj_name = "viper_simplewheel"
                obj_name_spe = "viper_simplewheel." + str(2).zfill(3)
                if n_r > 0:
                    obj_name_spe = "viper_simplewheel." + str(4 * n_r + 2).zfill(3)
            if n==4:
                obj_name = "viper_simplewheel"
                obj_name_spe = "viper_simplewheel." + str(3).zfill(3)
                if n_r > 0:
                    obj_name_spe = "viper_simplewheel." + str(4 * n_r + 3).zfill(3)
            if n==5:
                obj_name = "viper_L_steer"
                obj_name_spe = "viper_L_steer"
                if n_r > 0:
                    obj_name_spe = "viper_L_steer." + str(2 * n_r).zfill(3)
            if n==6:
                obj_name = "viper_R_steer"
                obj_name_spe = "viper_R_steer"
                if n_r > 0:
                    obj_name_spe = "viper_R_steer." + str(2 * n_r).zfill(3)
            if n==7:
                obj_name = "viper_L_steer"
                obj_name_spe = "viper_L_steer." + str(1).zfill(3)
                if n_r > 0:
                    obj_name_spe = "viper_L_steer." + str(2 * n_r + 1).zfill(3)
            if n==8:
                obj_name = "viper_R_steer"
                obj_name_spe = "viper_R_steer." + str(1).zfill(3)
                if n_r > 0:
                    obj_name_spe = "viper_R_steer." + str(2 * n_r + 1).zfill(3)
            if n==9:
                obj_name = "viper_L_bt_sus"
                obj_name_spe = "viper_L_bt_sus"
                if n_r > 0:
                    obj_name_spe = "viper_L_bt_sus." + str(2 * n_r).zfill(3)
            if n==10:
                obj_name = "viper_R_bt_sus"
                obj_name_spe = "viper_R_bt_sus"
                if n_r > 0:
                    obj_name_spe = "viper_R_bt_sus." + str(2 * n_r ).zfill(3)
            if n==11:
                obj_name = "viper_L_bt_sus"
                obj_name_spe = "viper_L_bt_sus." + str(1).zfill(3)
                if n_r > 0:
                    obj_name_spe = "viper_L_bt_sus." + str(2 * n_r + 1).zfill(3)
            if n==12:
                obj_name = "viper_R_bt_sus"
                obj_name_spe = "viper_R_bt_sus." + str(1).zfill(3)
                if n_r > 0:
                    obj_name_spe = "viper_R_bt_sus." + str(2 * n_r + 1).zfill(3)
            if n==13:
                obj_name = "viper_L_up_sus"
                obj_name_spe = "viper_L_up_sus"
                if n_r > 0:
                    obj_name_spe = "viper_L_up_sus." + str(2 * n_r ).zfill(3)
            if n==14:
                obj_name = "viper_R_up_sus"
                obj_name_spe = "viper_R_up_sus"
                if n_r > 0:
                    obj_name_spe = "viper_R_up_sus." + str(2 * n_r ).zfill(3)
            if n==15:
                obj_name = "viper_L_up_sus"
                obj_name_spe = "viper_L_up_sus." + str(1).zfill(3)
                if n_r > 0:
                    obj_name_spe = "viper_L_up_sus." + str(2 * n_r + 1).zfill(3)
            if n==16:
                obj_name = "viper_R_up_sus"
                obj_name_spe = "viper_R_up_sus." + str(1).zfill(3)
                if n_r > 0:
                    obj_name_spe = "viper_R_up_sus." + str(2 * n_r + 1).zfill(3)
            if n==17:
                obj_name = "plate"
                obj_name_spe = "plate"
                if n_r > 0:
                    obj_name_spe = "plate.00" + str(n_r)

            file_loc = file_in + obj_name + ".obj"
            imported_object = bpy.ops.import_scene.obj(filepath = file_loc)
            obj_object = bpy.context.object
            body_id = n + n_r * num_body
            bpy.data.objects[obj_name_spe].location.x += dis[body_id][i][0]
            bpy.data.objects[obj_name_spe].location.y += dis[body_id][i][1]
            bpy.data.objects[obj_name_spe].location.z += dis[body_id][i][2]  
            bpy.data.objects[obj_name_spe].rotation_mode = 'QUATERNION'
            q = (rot[body_id][i][0],rot[body_id][i][1],rot[body_id][i][2],rot[body_id][i][3])
            bpy.data.objects[obj_name_spe].rotation_quaternion = q

    bpy.context.view_layer.update()


    #===========================================
    #==================== Load SPH particle file
    #===========================================
    radius_particle = 0.005

    positions = []
    dir = data_in + "fluid" + str(i) + ".csv"
    count = 0
    for line in open(dir):
        if count == 0:
            count = count + 1
            continue
        else:
            # you have to parse "x", "y", "z" and "r" from the variable "line"
            line_seg = line.split(",")
            x, y, z = line_seg[0], line_seg[1], line_seg[2]
            position_buff = (float(x), float(y), float(z))
            positions.append(position_buff)
            count = count + 1
    # print("total number of particles")
    # print(count)

    """ -------------- PARTICLE SYSTEM START-------------- """
    context = bpy.context
    # instance object
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1, location=(50,50,50))
    pippo = radius_particle
    ico = context.object

    # cube with ps
    bpy.ops.mesh.primitive_cube_add(size=0.0001)
    cube = context.object

    # ps
    ps = cube.modifiers.new("SomeName", 'PARTICLE_SYSTEM').particle_system
    psname = ps.name
    ps.settings.count = count-1
    ps.settings.lifetime = 1000
    ps.settings.frame_start = ps.settings.frame_end = 1
    ps.settings.render_type = "OBJECT"
    ps.settings.instance_object = ico

    def particle_handler(scene, depsgraph):
        ob = depsgraph.objects.get(cube.name)
        if ob:
            ps = ob.particle_systems[psname]
            f = scene.frame_current
            for m, particle in enumerate(ps.particles):
                setattr(particle, "location", positions[m])
                setattr(particle, "size", radius_particle)

    # Clear the post frame handler
    bpy.app.handlers.frame_change_post.clear()

    # Register our particleSetter with the post frame handler
    bpy.app.handlers.frame_change_post.append(particle_handler)

    # Trigger frame update
    bpy.context.scene.frame_current = 2
    """ -------------- PARTICLE SYSTEM END -------------- """

    bpy.context.view_layer.update()
    #===========================================
    #===========================================
    #===========================================


    #===========================================
    #========================= Rendering setting
    #===========================================
    bpy.ops.transform.rotate(value=(-math.pi * 0.5), orient_axis='X')  # value = Angle
    #bpy.ops.transform.rotate(value=(math.pi * 0.5), orient_axis='X')  # value = Angle

    bpy.ops.mesh.primitive_plane_add(size=200.0, calc_uvs=True, enter_editmode=False, 
                                     align='WORLD', location=(0.0, 0.0, 0.0))
    # test
    # camera_dis = []
    # camera_dis.append(float(dis[0][i][0]-base_loc[0]))
    # camera_dis.append(float(dis[0][i][1]-base_loc[1]))
    # camera_dis.append(float(dis[0][i][2]-base_loc[2]))

    # bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=(-5.2157+camera_dis[0], 3.4817+camera_dis[1], 1.9782+camera_dis[2]),
    #                     rotation=(1.4992378275, 0.013648474751, 4.3109632524), scale=(1.0, 1.0, 1.0))
    # bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=(-8.699*1.0, -11.285*1.0, 4.6723*1.2),
    #                           rotation=(1.2548917, 0.0139800873, -0.6300638), scale=(5.0, 5.0, 5.0))
    # bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=(8.199*1.0, -11.285*1.0, 4.2723*1.2),
    #                           rotation=(1.2548917, -0.0139800873, 0.6300638), scale=(5.0, 5.0, 5.0))
    bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', scale=(5.0, 5.0, 5.0))

    scene.camera = bpy.context.object

    # Set up rotational camera
    cam = bpy.data.objects["Camera"]
    # add rotational angle by 1
    ini_rad = 45
    cur_rad = 90 #ini_rad + i*0.5
    cam.location = rot_pos((dis[0][i][0],dis[0][i][1],dis[0][i][2]+6),10,cur_rad)
    point_at(cam, bpy.data.objects["viper_chassis"].location, roll=math.radians(0))

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
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.render.resolution_x = 3840
    bpy.context.scene.render.resolution_y = 2160
    bpy.context.scene.render.filepath = image_in + str(i) + ".png"
    #bpy.context.scene.render.image_settings.compression = 50
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
