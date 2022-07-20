import bpy
import math
import random
import mathutils
import csv
import sys
import os.path

# index number in each m113:
# Body 1 is: Chassis body
# Body 2 is: M113_SprocketLeft_gear
# Body 3 is: M113_IdlerLeft_wheel
# Body 4 is: M113_IdlerLeft_carrier  --- don't render now
# Body 5 is: M113_SuspensionLeft_0_arm --- don't render now
# Body 6 is: M113_RoadWheelLeft_0_wheel
# Body 7 is: M113_SuspensionLeft_1_arm --- don't render now
# Body 8 is: M113_RoadWheelLeft_1_wheel
# Body 9 is: M113_SuspensionLeft_2_arm --- don't render now
# Body 10 is: M113_RoadWheelLeft_2_wheel
# Body 11 is: M113_SuspensionLeft_3_arm --- don't render now
# Body 12 is: M113_RoadWheelLeft_3_wheel
# Body 13 is: M113_SuspensionLeft_4_arm --- don't render now
# Body 14 is: M113_RoadWheelLeft_4_wheel
# Body 15-77 is: M113_TrackShoeLeft_0_shoe - M113_TrackShoeLeft_62_shoe
# Body 78 is: M113_SprocketRight_gear
# Body 79 is: M113_IdlerRight_wheel
# Body 80 is: M113_IdlerRight_carrier --- don't render now
# Body 81 is: M113_SuspensionRight_0_arm --- don't render now
# Body 82 is: M113_RoadWheelRight_0_wheel
# Body 83 is: M113_SuspensionRight_1_arm --- don't render now
# Body 84 is: M113_RoadWheelRight_1_wheel
# Body 85 is: M113_SuspensionRight_2_arm --- don't render now
# Body 86 is: M113_RoadWheelRight_2_wheel
# Body 87 is: M113_SuspensionRight_3_arm --- don't render now
# Body 88 is: M113_RoadWheelRight_3_wheel
# Body 89 is: M113_SuspensionRight_4_arm --- don't render now
# Body 90 is: M113_RoadWheelRight_4_wheel
# Body 91-154 is: M113_TrackShoeRight_0_shoe - M113_TrackShoeRight_63_shoe


num_body = 154
num_vehicle = 1
num_rock = 25
rock_scale = 0.5
radius_particle = 0.01
time = []

#============ specify the directory of csv, obj, image, and such
home_root = "/srv/home/whu59/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1010/track_vehicle/"
data_in = home_root + "DEMO_OUTPUT/FSI_M113/M113_Rock_14/"
file_in = home_root + "Animation_Blender/rendering_new/chrono_m113_data/M113_render/"
image_in = home_root + "Animation_Blender/Image/M113_Rock_14/with_chassis_rot_texture_two_lights_another_direction/"

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
        for n in range(num_body):
            if n == 0:
                # continue
                obj_name = "chassis"
                obj_name_spe = "chassis"
            elif n == 1: # M113_SprocketLeft_gear
                obj_name = "sprocket_L"
                obj_name_spe = "sprocket_L"
            elif n == 2: # M113_IdlerLeft_wheel
                obj_name = "idler_L"
                obj_name_spe = "idler_L"
            elif n in range(5, 14, 2): # M113_RoadWheelLeft_wheel 1-4
                if n == 5:
                    obj_name = "roller_L"
                    obj_name_spe = "roller_L"
                else:
                    n_Roller_L = n_Roller_L + 1
                    obj_name = "roller_L"
                    obj_name_spe = "roller_L." + str(n_Roller_L).zfill(3)
            elif n in range(14, 77): # M113_TrackShoeLeft_shoe 1-63
                if n == 14:
                    obj_name = "trackshoe"
                    obj_name_spe = "trackshoe"
                else:
                    obj_name = "trackshoe"
                    obj_name_spe = "trackshoe." + str(n-14).zfill(3)
            elif n == 77: # M113_SprocketRight_gear
                obj_name = "sprocket_R"
                obj_name_spe = "sprocket_R"
            elif n == 78: # M113_IdlerRight_wheel
                obj_name = "idler_R"
                obj_name_spe = "idler_R"
            elif n in range(81, 90, 2): # M113_RoadWheelRight_wheel 1-4
                if n == 81:
                    obj_name = "roller_R"
                    obj_name_spe = "roller_R"
                else:
                    n_Roller_R = n_Roller_R + 1
                    obj_name = "roller_R"
                    obj_name_spe = "roller_R." + str(n_Roller_R).zfill(3)
            elif n in range(90, 154): # M113_TrackShoeRight_shoe 1-64
                obj_name = "trackshoe"
                obj_name_spe = "trackshoe." + str(n-90+63).zfill(3)
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
            bpy.data.objects[obj_name_spe].rotation_quaternion = q

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
                                     align='WORLD', location=(0.0, 0.0, 0.0))
    
    #======== create a camera and settings
    bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', scale=(5.0, 5.0, 5.0))
    scene.camera = bpy.context.object
    # Set up rotational camera
    cam = bpy.data.objects["Camera"]
    # add rotational angle by 1
    ini_rad = 135
    cur_rad = ini_rad - i*0.5
    cam.location = rot_pos((0, 0, 10), 30, cur_rad)
    point_at(cam, (0, 0, 2), roll=math.radians(0))
    # cam.location = rot_pos((dis[0][i][0],dis[0][i][1],dis[0][i][2]+4),15,cur_rad)
    # point_at(cam, (dis[0][i][0]-2,dis[0][i][1],dis[0][i][2]+0), roll=math.radians(0))

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
    light_object.location = ( 14 * math.cos(angle_pi), 14 * math.sin(angle_pi), 15)

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
    light_object1.location = ( dis[0][i][0] - 1.5,  dis[0][i][1] - 1.5, 15)

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
