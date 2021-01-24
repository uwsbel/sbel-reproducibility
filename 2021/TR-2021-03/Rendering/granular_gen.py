import multiprocessing

radius = 0.005
total_frame = 401
first_frame = 0
num_cpu_core = 20 #multiprocessing.cpu_count()

def process(core_idx,num_cpu_core,total_frame):
    seg = int(total_frame / num_cpu_core)
    start_frame = (int)(seg * core_idx) + first_frame
    end_frame = total_frame + first_frame
    if core_idx!=num_cpu_core-1:
        end_frame = seg * (core_idx+1) + first_frame

    for k in range(start_frame, end_frame, 1):
        positions = []
        home_root = "/srv/home/whu59/research/sbel/d_chrono_fsi_granular/"
        file_in = home_root + "chrono_fsi_granular_1001/chrono-dev-io-Viper/24/DEMO_OUTPUT/FSI_Viper/Rover/"
        file_out = "granular_file/"
        dir = file_in + "fluid" + str(k) + ".csv"
        # filepath_ver = "res_ver_obj_" + str(k) + ".obj"
        # filepath_face = "res_fac_obj_" + str(k) + ".obj"
        out_mesh = file_out + "fluid" + str(k) + ".obj"
        count = 0
        for line in open(dir):
            if count == 0:
                count = count + 1
                continue
            else:
                # you have to parse "x", "y", "z" and "r" from the variable "line"
                line_seg = line.split(",")
                x, y, z = line_seg[0], line_seg[1], line_seg[2]
                position_buff = []
                position_buff.append(float(x))
                position_buff.append(float(y))
                position_buff.append(float(z))
                positions.append(position_buff)
                count = count + 1

        sphere_vertices = []
        sphere_faces = []

        f = open("pyramid.obj")
        for line in f:
            if line[:2] == "v ":
                index1 = line.find(" ") + 1
                index2 = line.find(" ", index1 + 1)
                index3 = line.find(" ", index2 + 1)

                vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                sphere_vertices.append(vertex)

            elif line[0] == "f":
                string = line.replace("//", "/")
                ##
                i = string.find(" ") + 1
                face = []
                for item in range(string.count(" ")):
                    if string.find(" ", i) == -1:
                        face.append(int(string[i:-1]))
                        break
                    face.append(int(string[i:string.find(" ", i)]))
                    i = string.find(" ", i) + 1
                ##
                sphere_faces.append(tuple(face))

        f.close()

        sphere_ver_post = []
        for i in sphere_vertices:
            buff = []
            buff.append(float(i[0]) * float(radius))
            buff.append(float(i[1]) * float(radius))
            buff.append(float(i[2]) * float(radius))
            sphere_ver_post.append(buff)

        face_post = []
        for i in sphere_faces:
            buff = []
            buff.append(int(i[0]))
            buff.append(int(i[1]))
            buff.append(int(i[2]))
            face_post.append(buff)

        # f_1 = open(filepath_ver, 'w')
        # f_2 = open(filepath_face, 'w')

        count_ver = 0
        count = 0

        tot_ver = []
        tot_face = []

        for i in positions:
            final_ver = []
            final_face = []
            for j in face_post:
                face_buff = []
                face_buff.append(count_ver + int(j[0]))
                face_buff.append(count_ver + int(j[1]))
                face_buff.append(count_ver + int(j[2]))
                final_face.append(face_buff)
            for j in sphere_ver_post:
                ver_buff = []
                ver_buff.append(float(j[0]) + float(i[0]))
                ver_buff.append(float(j[1]) + float(i[1]))
                ver_buff.append(float(j[2]) + float(i[2]))
                final_ver.append(ver_buff)
            count = count + 1
            count_ver = count_ver + int(len(final_ver))
            for a in range(len(final_ver)):
                tot_ver.append(final_ver[a])
                # f_1.write("v "+str(round(final_ver[a][0],5))+" "+str(round(final_ver[a][1],5))+" "+str(round(final_ver[a][2],5))+"\n")

            for a in range(len(final_face)):
                tot_face.append(final_face[a])
                # f_2.write("f "+str(final_face[a][0])+" "+str(final_face[a][1])+" "+str(final_face[a][2])+"\n")

            # del(final_ver)
            # del(final_face)

        # f_1.close()
        # f_2.close()
        f = open(out_mesh, 'w')
        for b in range(len(tot_ver)):
            f.write("v " + str(round(tot_ver[b][0], 5)) + " " + str(round(tot_ver[b][1], 5)) + " " + str(
                round(tot_ver[b][2], 5)) + "\n")

        for b in range(len(tot_face)):
            f.write("f " + str(tot_face[b][0]) + " " + str(tot_face[b][1]) + " " + str(tot_face[b][2]) + "\n")

        f.close()
        # Python program to
        # demonstrate merging of
        # two files

        # Creating a list of filenames
        # filenames = [filepath_ver, filepath_face]

        # Open file3 in write mode
        # with open(out_mesh, 'w') as outfile:

        # Iterate through list
        # for names in filenames:

        # Open each file in read mode
        # with open(names) as infile:

        # read the data from file1 and
        # file2 and write it in file3
        # outfile.write(infile.read())

        # Add '\n' to enter data of file2
        # from next line
        # outfile.write("\n")


if __name__ == '__main__':
    if num_cpu_core>total_frame:
        num_cpu_core = total_frame

    for core_idx in range(num_cpu_core):
        print("Process "+str(core_idx)+" started")
        p = multiprocessing.Process(target=process, args=(core_idx,num_cpu_core,total_frame))
        p.start()
