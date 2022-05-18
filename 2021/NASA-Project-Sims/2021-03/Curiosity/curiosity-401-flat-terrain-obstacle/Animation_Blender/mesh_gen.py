import meshio
import multiprocessing

total_frame = 400
first_frame = 1
num_cpu_core = 20 #multiprocessing.cpu_count()

def process(core_idx,num_cpu_core,total_frame):
    seg = int(total_frame / num_cpu_core)
    start_frame = (int)(seg * core_idx) + first_frame
    end_frame = total_frame + first_frame
    if core_idx!=num_cpu_core-1:
        end_frame = seg * (core_idx+1) + first_frame

    for i in range(start_frame, end_frame, 1):
        home_root = "/srv/home/whu59/research/sbel/d_chrono_fsi_granular/"
        file_root = home_root + "chrono_fsi_granular_1001/chrono-dev-io-test-random-motion/curiosity-401-C-Wi=0.8/DEMO_OUTPUT/FSI_Curiosity/Curiosity/"

        file_mesh = "mesh_file/"

        for j in range(1, 16, 1):
            filename = file_root + "Body_"  + str(j) + "_" + str(i) + ".vtk"
            out = file_mesh + "Body_" + str(j) + "_" + str(i) + ".obj"
            mesh = meshio.read(filename,file_format="vtk")
            meshio.write(out,mesh,file_format="obj")

        
if __name__ == '__main__':
    if num_cpu_core>total_frame:
        num_cpu_core = total_frame

    for core_idx in range(num_cpu_core):
        print("Process "+str(core_idx)+" started")
        p = multiprocessing.Process(target=process, args=(core_idx,num_cpu_core,total_frame))
        p.start()