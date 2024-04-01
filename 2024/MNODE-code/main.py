# author: Jingquan Wang
# This is the main file for the project
# import all the packages needed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import matplotlib.colors as colors
from torch.utils.data import TensorDataset, DataLoader
from Model.model import *
from Model.utils import *
from Model.force_fun import *
from Model.Data_generator import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Configuration dictionary
config = {
    "test_case": "Slider_Crank",
    "generation_numerical_methods": "fe",
    "numerical_methods": "fe",
    "dt": 0.01,
    "num_steps": 400,
    "training_size": 300,
    "num_epochs":400,
    "num_steps_test": 400,
    "dt_test": 0.01,
    "step_delay": 2,
    "random_seed": 5,
    "if_symplectic": False,
    "model_string": "PNODE",
}
def main(config):
    set_seed(config["random_seed"])
    Ground_trajectory = generate_training_data(test_case=config["test_case"],numerical_methods=config["generation_numerical_methods"],dt=config["dt"],num_steps=config["num_steps"])
    save_data(Ground_trajectory,config["test_case"],"Ground_truth",config["training_size"],config["num_steps_test"],config["dt_test"])
    num_body = Ground_trajectory.shape[1]
    model = PNODE(num_bodys=num_body).to(device)
    trained_model = train_trajectory_PNODE(test_case=config["test_case"],numerical_methods=config["numerical_methods"],model=model,body_tensor=Ground_trajectory, step_delay=config["step_delay"],training_size=config["training_size"], num_epochs=config["num_epochs"], dt=config["dt"])
    #save_model(trained_model,config["test_case"],config["numerical_methods"],config["model_string"],config["training_size"],config["num_epochs"],config["dt"],config["step_delay"])
    #trained_model=train_sampling_PNODE(test_case=config["test_case"],numerical_methods=config["numerical_methods"],model=model,training_pair_input=training_pair_input,training_pair_output=training_pair_output,training_size=config["training_size"], num_epochs=config["num_epochs"], dt=config["dt"])
    #trained_model=train_trajectory_PNODE_symplectic(test_case=config["test_case"],numerical_methods=config["numerical_methods"],model=model,body_tensor=a, training_size=config["training_size"], num_epochs=config["num_epochs"], dt=config["dt"])
    #test_trajectory = test_PNODE(numerical_methods=config["numerical_methods"],model=trained_model, body=training_pair_input[0,:], num_steps=config["num_steps_test"],dt=config["dt_test"])
    test_trajectory = test_PNODE(numerical_methods=config["numerical_methods"], model=trained_model,body=Ground_trajectory[0, :], num_steps=config["num_steps_test"],dt=config["dt_test"])
    #calculate the mse between the test trajectory and the ground truth trajectory
    mse = torch.nn.MSELoss()
    loss = mse(test_trajectory[0:400], Ground_trajectory)
    print("The MSE between the test trajectory and the ground truth trajectory is:", loss)
    #also save the test trajectory with the same label
    plot_energy_trajectory(test_case=config["test_case"], numerical_methods=config["numerical_methods"],
                           dt=config["dt_test"], data=test_trajectory, test=True)
    save_data(test_trajectory,config["test_case"],config["model_string"],config["training_size"],config["num_steps_test"],config["dt_test"])
    #test_trajectory = test_PNODE_symplectic(numerical_methods=config["numerical_methods"],model=trained_model, body=a[0], num_steps=config["num_steps_test"],dt=config["dt_test"])
    #plot_energy_trajectory(test_case=config["test_case"], numerical_methods=config["numerical_methods"], dt=config["dt_test"], data=test_trajectory,test=True)
    compare_trajectories(test_case=config["test_case"], numerical_methods=config["numerical_methods"],num_training=config["training_size"],model_string="PNODE",if_symplectic=config["if_symplectic"],dt=config["dt_test"], data1=Ground_trajectory, data2=test_trajectory)

    #compare_line_trajectories(test_case=config["test_case"], numerical_methods=config["numerical_methods"],dt=config["dt_test"], data1=Ground_trajectory, data2=test_trajectory)
if __name__ == "__main__":
    main(config)
