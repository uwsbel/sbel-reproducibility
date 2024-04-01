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
    "num_epochs":600,
    "num_steps_test": 400,
    "dt_test": 0.01,
    "step_delay": 2,
    "random_seed": 0,
    "if_symplectic": False,
    "model_string": "FCNN",
}

def main(config):
    set_seed(config["random_seed"])
    Ground_trajectory = generate_training_data(test_case=config["test_case"],numerical_methods=config["generation_numerical_methods"],dt=config["dt"],num_steps=config["num_steps"])
    #Ground_trajectory = Ground_trajectory[:, :, 0:2]
    #print(Ground_trajectory.shape)

    #plot_energy_trajectory(test_case=config["test_case"],numerical_methods=config["numerical_methods"],dt=config["dt"],data=Ground_trajectory)
    num_body = Ground_trajectory.shape[1]
    FCNNmodel = FCNN(num_bodys=num_body,d_input_interest=0).to(device)
    #get parameters number
    calculate_model_parameters(FCNNmodel)
    t_array=np.arange(0, config["num_steps"] * config["dt"], config["dt"])
    #reshape to (num_steps,1)
    t_array = t_array.reshape(-1, 1)
    aug_t_tensor = torch.from_numpy(t_array).float().to(device)
    aug_t_tensor_test = torch.from_numpy(np.arange(0, config["num_steps_test"] * config["dt_test"], config["dt_test"]).reshape(-1,1)).float().to(device)
    trained_model=train_FCNN(test_case=config["test_case"],model=FCNNmodel,aug_t_tensor=aug_t_tensor,aug_body_tensor=Ground_trajectory,dt=config["dt"],training_size=config["training_size"],num_epochs=config["num_epochs"])
    save_model(trained_model,config["test_case"],config["numerical_methods"],config["model_string"],config["training_size"],config["num_epochs"],config["dt"],config["step_delay"])
    test_trajectory= test_FCNN(test_case=config["test_case"],model=trained_model,aug_t_tensor=aug_t_tensor_test,dt=config["dt_test"],num_step=config["num_steps_test"],num_body=num_body)
    save_data(test_trajectory,config["test_case"],config["model_string"],config["training_size"],config["num_steps_test"],config["dt_test"])
    plot_energy_trajectory(test_case=config["test_case"], numerical_methods=config["numerical_methods"], dt=config["dt_test"], data=test_trajectory,test=True)
    compare_trajectories(test_case=config["test_case"], numerical_methods=config["numerical_methods"],if_symplectic=config["if_symplectic"],model_string=config["model_string"],num_training=config["training_size"],
                         dt=config["dt_test"], data1=Ground_trajectory, data2=test_trajectory)
    #plt.figure()
# run the main function
if __name__ == "__main__":
    main(config)
