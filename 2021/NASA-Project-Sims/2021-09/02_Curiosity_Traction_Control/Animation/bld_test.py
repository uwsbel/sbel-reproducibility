import bpy
import math
import random

radius_particle = 0.005
        

for idx in range(0,750,1):

    positions = []
    dir = "/srv/home/groups/sbel/fsi/diff_mu/711_flat_up_flat_real_mass/DEMO_OUTPUT/FSI_Curiosity/Curiosity/fluid"+str(idx)+".csv"
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
    print("total number of particles")
    print(count)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.objects.keys()

    out_dir = "/srv/home/zzhou292/blender/res_data/711/"

    bpy.ops.mesh.primitive_plane_add(size=200.0, calc_uvs=True, enter_editmode=False, align='WORLD',
                                    location=(0.0, 0.0, 0.0))

    for i in range(15):
        file_loc = "/srv/home/groups/sbel/fsi/diff_mu/711_flat_up_flat_real_mass/DEMO_OUTPUT/FSI_Curiosity/Curiosity/Body_"+str(i+1)+"_"+str(idx)+".obj"
        imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
        ov=bpy.context.copy()
        ov['area']=[a for a in bpy.context.screen.areas if a.type=="VIEW_3D"][0]
        bpy.ops.transform.rotate(ov,value=(math.pi * 0.5), orient_axis='X')  # value = Angle




    """ -------------------------PARTICLE SYSTEM TEST------------------------------------------------------ """
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
    """ -----------PARTICLE SYSTEM TEST END-------------------------------------------- """

    bpy.context.view_layer.update()


    bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=(8.53977, -12.4916, 5.6382),
                        rotation=(1.2688578569, -0.014040161541, 0.50421514893), scale=(5.0, 5.0, 5.0))
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
    light_data.energy = 14000

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
    #bpy.context.scene.render.resolution_percentage = 200
    bpy.context.scene.cycles.samples = 512
    bpy.context.scene.view_layers['View Layer'].cycles.use_denoising
    bpy.context.scene.render.resolution_x = 3840
    bpy.context.scene.render.resolution_y = 2160
    bpy.context.scene.render.filepath = out_dir + str(idx)+".png"
    # bpy.context.scene.render.image_settings.compression = 50
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
