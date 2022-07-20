import meshio
import multiprocessing

total_frame = 401
first_frame = 0
num_cpu_core = 40 #multiprocessing.cpu_count()

def process(core_idx,num_cpu_core,total_frame):
    seg = int(total_frame / num_cpu_core)
    start_frame = (int)(seg * core_idx) + first_frame
    end_frame = total_frame + first_frame
    if core_idx!=num_cpu_core-1:
        end_frame = seg * (core_idx+1) + first_frame

    for i in range(start_frame, end_frame, 1):
        home_root = "/srv/home/whu59/research/sbel/d_chrono_fsi_granular/"
        file_root = home_root + "chrono_fsi_granular_1001/chrono-dev-io-Viper/18/DEMO_OUTPUT/FSI_Viper/Rover/"

        file_mesh = "mesh_file/"

        filename_0 = file_root + "body_"+str(i)+".vtk"
        out_0 = file_mesh + "body_"+str(i)+".obj"
        mesh = meshio.read(filename_0,file_format="vtk")
        meshio.write(out_0,mesh,file_format="obj")

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
        
        filename_5 = file_root + "lowerRod_1_"+str(i)+".vtk"
        out_5 = file_mesh + "lowerRod_1_"+str(i)+".obj"
        mesh = meshio.read(filename_5,file_format="vtk")
        meshio.write(out_5,mesh,file_format="obj")
        
        filename_6 = file_root + "lowerRod_2_"+str(i)+".vtk"
        out_6 = file_mesh + "lowerRod_2_"+str(i)+".obj"
        mesh = meshio.read(filename_6,file_format="vtk")
        meshio.write(out_6,mesh,file_format="obj")
        
        filename_7 = file_root + "lowerRod_3_"+str(i)+".vtk"
        out_7 = file_mesh + "lowerRod_3_"+str(i)+".obj"
        mesh = meshio.read(filename_7,file_format="vtk")
        meshio.write(out_7,mesh,file_format="obj")
        
        filename_8 = file_root + "lowerRod_4_"+str(i)+".vtk"
        out_8 = file_mesh + "lowerRod_4_"+str(i)+".obj"
        mesh = meshio.read(filename_8,file_format="vtk")
        meshio.write(out_8,mesh,file_format="obj")

        filename_9 = file_root + "upperRod_1_"+str(i)+".vtk"
        out_9 = file_mesh + "upperRod_1_"+str(i)+".obj"
        mesh = meshio.read(filename_9,file_format="vtk")
        meshio.write(out_9,mesh,file_format="obj")
        
        filename_10 = file_root + "upperRod_2_"+str(i)+".vtk"
        out_10 = file_mesh + "upperRod_2_"+str(i)+".obj"
        mesh = meshio.read(filename_10,file_format="vtk")
        meshio.write(out_10,mesh,file_format="obj")
        
        filename_11 = file_root + "upperRod_3_"+str(i)+".vtk"
        out_11 = file_mesh + "upperRod_3_"+str(i)+".obj"
        mesh = meshio.read(filename_11,file_format="vtk")
        meshio.write(out_11,mesh,file_format="obj")
        
        filename_12 = file_root + "upperRod_4_"+str(i)+".vtk"
        out_12 = file_mesh + "upperRod_4_"+str(i)+".obj"
        mesh = meshio.read(filename_12,file_format="vtk")
        meshio.write(out_12,mesh,file_format="obj")
        
        filename_13 = file_root + "steerRod_1_"+str(i)+".vtk"
        out_13 = file_mesh + "steerRod_1_"+str(i)+".obj"
        mesh = meshio.read(filename_13,file_format="vtk")
        meshio.write(out_13,mesh,file_format="obj")
        
        filename_14 = file_root + "steerRod_2_"+str(i)+".vtk"
        out_14 = file_mesh + "steerRod_2_"+str(i)+".obj"
        mesh = meshio.read(filename_14,file_format="vtk")
        meshio.write(out_14,mesh,file_format="obj")
        
        filename_15 = file_root + "steerRod_3_"+str(i)+".vtk"
        out_15 = file_mesh + "steerRod_3_"+str(i)+".obj"
        mesh = meshio.read(filename_15,file_format="vtk")
        meshio.write(out_15,mesh,file_format="obj")
        
        filename_16 = file_root + "steerRod_4_"+str(i)+".vtk"
        out_16 = file_mesh + "steerRod_4_"+str(i)+".obj"
        mesh = meshio.read(filename_16,file_format="vtk")
        meshio.write(out_16,mesh,file_format="obj")

if __name__ == '__main__':
    if num_cpu_core>total_frame:
        num_cpu_core = total_frame

    for core_idx in range(num_cpu_core):
        print("Process "+str(core_idx)+" started")
        p = multiprocessing.Process(target=process, args=(core_idx,num_cpu_core,total_frame))
        p.start()