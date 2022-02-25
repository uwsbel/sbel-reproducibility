import json
import numpy as np

bodies = [2, 4, 6, 8, 16, 32]

for numBodies in bodies:    

    numSphericalConstraints = numBodies
    file_name = "pendulum_nb_{}_init_omega".format(numBodies)
    
    # list of bodies
    list_of_bodies = []
    
    # ground object
    body_ground = {}
    body_ground["name"] = "ground"
    body_ground["id"] = 0
    
    list_of_bodies = []
    list_of_bodies.append(body_ground)
    
    # define values that are constant
    r_dot = [0, 0, 0]
    A = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    
    for i in range(0, numBodies):
        body_id = i+1
        r = [0, 0, -(2*i+1)]
        name = "body_{}".format(i+1)
        body_dict = {}
        body_dict["name"] = name
        body_dict["id"] = body_id
        body_dict["r"] = r
        body_dict["r_dot"] = r_dot
        body_dict["A"] = A
        omega = [0, 0, np.random.uniform(-10, 10)]
        body_dict["omega"] = omega
        list_of_bodies.append(body_dict)
    
    # list of constraints
    list_of_constraints = []
    
    # constant quantities for constraint
    c_i = [1, 0 ,0]
    c_j = [0, 1 ,0]
    c_k = [0, 0 ,1]
    c_mat = [c_i, c_j, c_k]
    cons_type = "CD"
    zero_str = "{}".format(0)
    xyz = ['i', 'j', 'k']
    zero_array = [0, 0, 0]
    pos_one = [1, 0, 0]
    neg_one = [-1, 0, 0]
    
    
    for sph_constr_itr in range(0, numSphericalConstraints):
        for cord in range(0, 3):
            constr = {}
            constr["name"] = "spherical_{}_CD_{}".format(sph_constr_itr+1, xyz[cord])
            constr["type"] = "CD"
            constr["c"] = c_mat[cord]
            constr["f"] = zero_str
            constr["f_dot"] = zero_str
            constr["f_ddot"] = zero_str
            if sph_constr_itr == 0:
                constr["body_i"] = 1
                constr["body_j"] = 0
                constr["s_bar_p_i"] = [-1, 0, 0]
                constr["s_bar_q_j"] = [ 0, 0, 0]
            else:
                constr["body_i"] = sph_constr_itr
                constr["body_j"] = sph_constr_itr + 1
                constr["s_bar_p_i"] = [ 1, 0, 0]
                constr["s_bar_q_j"] = [-1, 0, 0]
            
            list_of_constraints.append(constr)
            
    
    d = {}
    d["name"] = file_name
    d["bodies"] = list_of_bodies
    d["constraints"] = list_of_constraints
    
    # encoding to JSON
    data = json.dumps(d)
    
    # write to a file
    with open("{}.json".format(file_name),"w") as f:
      f.write(data)