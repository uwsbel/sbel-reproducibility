import meshio

for i in range(0,501,1):
    home_root = "/srv/home/whu59/research/sbel/"
    file_root = home_root + "d_chrono_fsi_granular/chrono_fsi_granular_102_Rover_Wheel/chrono-dev-io/17/DEMO_OUTPUT/FSI_Rover/Rover/"

    file_mesh = "mesh_file/"

    filename_1 = file_root + "wheel_1_"+str(i)+".vtk"
    out_1 = file_mesh + "wheel_1_"+str(i)+".obj"
    mesh = meshio.read(filename_1,file_format="vtk")
    meshio.write(out_1,mesh,file_format="obj")
    
    filename_2 = file_root + "wheel_2_"+str(i)+".vtk"
    out_2 = file_mesh + "wheel_2_"+str(i)+".obj"
    mesh = meshio.read(filename_2,file_format="vtk")
    meshio.write(out_2,mesh,file_format="obj")
    
    filename_3 = file_root + "wheel_3_"+str(i)+".vtk"
    out_3 = file_mesh + "wheel_3_"+str(i)+".obj"
    mesh = meshio.read(filename_3,file_format="vtk")
    meshio.write(out_3,mesh,file_format="obj")
    
    filename_4 = file_root + "wheel_4_"+str(i)+".vtk"
    out_4 = file_mesh + "wheel_4_"+str(i)+".obj"
    mesh = meshio.read(filename_4,file_format="vtk")
    meshio.write(out_4,mesh,file_format="obj")
    
    filename_5 = file_root + "wheel_5_"+str(i)+".vtk"
    out_5 = file_mesh + "wheel_5_"+str(i)+".obj"
    mesh = meshio.read(filename_5,file_format="vtk")
    meshio.write(out_5,mesh,file_format="obj")
    
    filename_6 = file_root + "wheel_6_"+str(i)+".vtk"
    out_6 = file_mesh + "wheel_6_"+str(i)+".obj"
    mesh = meshio.read(filename_6,file_format="vtk")
    meshio.write(out_6,mesh,file_format="obj")
    
    filename_7 = file_root + "wheel_7_"+str(i)+".vtk"
    out_7 = file_mesh + "wheel_7_"+str(i)+".obj"
    mesh = meshio.read(filename_7,file_format="vtk")
    meshio.write(out_7,mesh,file_format="obj")
    
    filename_8 = file_root + "wheel_8_"+str(i)+".vtk"
    out_8 = file_mesh + "wheel_8_"+str(i)+".obj"
    mesh = meshio.read(filename_8,file_format="vtk")
    meshio.write(out_8,mesh,file_format="obj")
    
    filename_9 = file_root + "Rover_body_"+str(i)+".vtk"
    out_9 = file_mesh + "body_"+str(i)+".obj"
    mesh = meshio.read(filename_9,file_format="vtk")
    meshio.write(out_9,mesh,file_format="obj")