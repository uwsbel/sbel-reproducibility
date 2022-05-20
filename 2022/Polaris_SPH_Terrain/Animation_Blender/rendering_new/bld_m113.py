import bpy
import math
import random
import mathutils
import csv
import sys
import os.path

# index number in each polaris:
# Body 1 is: MRZR chassis body
# Body 2 is: MRZR Pitman Arm Steering_link
# Body 3 is: MRZR Pitman Arm Steering_arm
# Body 4 is: MRZR DoubleWishbone_spindle_L
# Body 5 is: MRZR DoubleWishbone_upright_L
# Body 6 is: MRZR DoubleWishbone_UCA_L
# Body 7 is: MRZR DoubleWishbone_LCA_L
# Body 8 is: MRZR DoubleWishbone_spindle_R
# Body 9 is: MRZR DoubleWishbone_upright_R
# Body 10 is: MRZR DoubleWishbone_UCA_R
# Body 11 is: MRZR DoubleWishbone_LCA_R
# Body 12 is: MRZR ThreeLinkIRS_spindle_L
# Body 13 is: MRZR ThreeLinkIRS_arm_L
# Body 14 is: MRZR ThreeLinkIRS_upper_L
# Body 15 is: MRZR ThreeLinkIRS_lower_L
# Body 16 is: MRZR ThreeLinkIRS_spindle_R
# Body 17 is: MRZR ThreeLinkIRS_arm_R
# Body 18 is: MRZR ThreeLinkIRS_upper_R
# Body 19 is: MRZR ThreeLinkIRS_lower_R
# Body 20 is: MRZR RSD Antirollbar_arm_left
# Body 21 is: MRZR RSD Antirollbar_arm_right



num_body = 21
num_vehicle = 1
num_rock = 0
rock_scale = 0.3
radius_particle = 0.01
time = []

#============ specify the directory of csv, obj, image, and such
home_root = "/srv/home/whu59/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1010/chrono_radu/"
data_in = home_root + "chrono_build/bin/DEMO_OUTPUT/FSI_POLARIS/Polaris_02/"
file_in = home_root + "Polaris/Animation_Blender/rendering_new/chrono_polaris_data/"
image_in = home_root + "Polaris/Animation_Blender/Image/02/new/"

#============ find the position that the camera points at
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

#============ find the position and rotation of the camera
def rot_pos(center, radius, angle=0):
    """
    return (x,y,z) points on a circle centered at (center) with a radius of (radius) with an angle of (angle)
    """
    # convert to radian
    angle_pi = angle/180*math.pi
    # generate point on circle
    return (center[0]+radius*math.sin(angle_pi),center[1]+radius*math.cos(angle_pi),center[2])

#============ quaternion multiply
def quaternion_multiply(quaternion0, quaternion1):
    w0 = quaternion0[0]
    x0 = quaternion0[1]
    y0 = quaternion0[2]
    z0 = quaternion0[3]
    w1 = quaternion1[0]
    x1 = quaternion1[1]
    y1 = quaternion1[2]
    z1 = quaternion1[3]
    return (-x1*x0 - y1*y0 - z1*z0 + w1*w0,
             x1*w0 + y1*z0 - z1*y0 + w1*x0,
            -x1*z0 + y1*w0 + z1*x0 + w1*y0,
             x1*y0 - y1*x0 + z1*w0 + w1*z0)

#============ load position, velocity, rotation of each body at different time
dis = []
rot = []
for i in range(num_body * num_vehicle + num_rock):
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

#===========================================
#============================ Start the loop
#===========================================
jobid = int(sys.argv[4])
start_frame = jobid*1 + 0
end_frame = jobid*1 + 1
for i in range(start_frame, end_frame, 1):
    #===========================================
    #======== check if the png file exits or not
    #===========================================
    image_path = image_in + str(i) + ".png"
    file_exists = os.path.exists(image_path)
    if file_exists:
        sys.exit()
        
    #===========================================
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.objects.keys()

    #===========================================
    #===================== load vehicle obj file
    #===========================================
    for n_r in range(num_vehicle):
        obj_name = ""
        obj_name_spe = ""
        n_Roller_L = 0
        n_Roller_R = 0
        for nn in range(2):
            for n in range(num_body):
                q_rot = (1, 0, 0, 0)
                if nn == 0:
                    if n == 0:
                        # continue
                        obj_name = "Polaris_chassis"
                        obj_name_spe = "Polaris_chassis"
                    elif n == 3:
                        obj_name = "Polaris_tire"
                        obj_name_spe = "Polaris_tire"
                    elif n == 7:
                        obj_name = "Polaris_tire"
                        obj_name_spe = "Polaris_tire.001"
                    elif n == 11:
                        obj_name = "Polaris_tire"
                        obj_name_spe = "Polaris_tire.002"
                    elif n == 15:
                        obj_name = "Polaris_tire"
                        obj_name_spe = "Polaris_tire.003"
                    else:
                        continue
                if nn == 1:
                    if n == 3:
                        obj_name = "Polaris_wheel"
                        obj_name_spe = "Polaris_wheel"
                    elif n == 7:
                        obj_name = "Polaris_wheel"
                        obj_name_spe = "Polaris_wheel.001"
                        q_rot = (0, 1, 0, 0)
                    elif n == 11:
                        obj_name = "Polaris_wheel"
                        obj_name_spe = "Polaris_wheel.002"
                    elif n == 15:
                        obj_name = "Polaris_wheel"
                        obj_name_spe = "Polaris_wheel.003"
                        q_rot = (0, 1, 0, 0)
                    else:
                        continue

                file_loc = file_in + obj_name + ".obj"
                imported_object = bpy.ops.import_scene.obj(filepath = file_loc)
                obj_object = bpy.context.object
                body_id = n + n_r * num_body
                bpy.data.objects[obj_name_spe].location.x += dis[body_id][i][0]
                bpy.data.objects[obj_name_spe].location.y += dis[body_id][i][1]
                bpy.data.objects[obj_name_spe].location.z += dis[body_id][i][2]  
                bpy.data.objects[obj_name_spe].rotation_mode = 'QUATERNION'
                q = (rot[body_id][i][0],rot[body_id][i][1],rot[body_id][i][2],rot[body_id][i][3])
                bpy.data.objects[obj_name_spe].rotation_quaternion = quaternion_multiply(q, q_rot)

    #===========================================
    #======================== load rock obj file
    #===========================================
    for n_r in range(num_rock):
        obj_name = "rock3"
        obj_name_spe = "rock3"
        if n_r > 0:
            obj_name_spe = "rock3." + str(n_r).zfill(3)

        file_loc = file_in + obj_name + ".obj"
        imported_object = bpy.ops.import_scene.obj(filepath = file_loc)
        bpy.ops.transform.resize(value=(rock_scale, rock_scale, rock_scale))
        obj_object = bpy.context.object
        body_id = n_r + num_vehicle * num_body
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
    bpy.ops.mesh.primitive_plane_add(size=10000.0, calc_uvs=True, enter_editmode=False, 
                                     align='WORLD', location=(0.0, 0.0, -0.3))
    
    #======== create a camera and settings
    bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', scale=(5.0, 5.0, 5.0))
    scene.camera = bpy.context.object
    # Set up rotational camera
    cam = bpy.data.objects["Camera"]
    # add rotational angle by 1
    ini_rad = 135
    cur_rad = ini_rad - i*0.5
    # cam.location = rot_pos((0, 0, 10), 30, cur_rad)
    # point_at(cam, (0, 0, 2), roll=math.radians(0))
    cam.location = rot_pos((dis[0][i][0], dis[0][i][1], dis[0][i][2] + 8),15,cur_rad)
    point_at(cam, (dis[0][i][0] - 1.5, dis[0][i][1], dis[0][i][2] + 0.5), roll=math.radians(0))

    #======== create light datablock, set attributes
    light_data = bpy.data.lights.new(name="light_2.80", type='POINT') # type='SUN'
    light_data.energy = 5000
    # create new object with our light datablock
    light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)
    # link light object
    bpy.context.collection.objects.link(light_object)
    # make it active
    bpy.context.view_layer.objects.active = light_object
    # change location
    angle_pi = (45 + i*0.5) / 180 * math.pi
    light_object.location = ( 14 * math.cos(angle_pi) + dis[0][i][0], 14 * math.sin(angle_pi) + dis[0][i][1], 15)

    #======== create another light datablock, set attributes
    light_data1 = bpy.data.lights.new(name="light_top", type='POINT')
    light_data1.energy = 5000
    # create new object with our light datablock
    light_object1 = bpy.data.objects.new(name="light_top", object_data=light_data1)
    # link light object
    bpy.context.collection.objects.link(light_object1)
    # make it active
    bpy.context.view_layer.objects.active = light_object1
    # change location
    light_object1.location = ( dis[0][i][0],  dis[0][i][1], 15)

    #======== use GPU if available
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

    #=======================
    #======== ouput settings
    #=======================
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
