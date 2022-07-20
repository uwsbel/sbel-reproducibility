import meshio

filename_2 = "wheel_grouser_150.vtk"
out_2 = "wheel_grouser_150.obj"
mesh = meshio.read(filename_2,file_format="vtk")
meshio.write(out_2,mesh,file_format="obj")