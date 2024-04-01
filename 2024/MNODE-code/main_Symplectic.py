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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Configuration dictionary
config = {
    "test_case": "Single_Mass_Spring_Symplectc",
    "generation_numerical_methods": "sep_sv",
    "numerical_methods": "sep_sv",
    "dt": 0.01,
    "num_steps": 3000,
    "training_size": 300,
    "num_epochs": 450,
    "num_steps_test": 3000,
    "dt_test": 0.01,
    "step_delay": 10,
    "random_seed": 0,
    "if_symplectic": True,
    "model_string": "PNODE_Symplectic",
}
def main(config):
    set_seed(config["random_seed"])
    # generate the data for the test case
    Ground_trajectory = generate_training_data(
        test_case=config["test_case"],
        numerical_methods=config["generation_numerical_methods"],
        dt=config["dt"],
        num_steps=config["num_steps"],
    )
    save_data(Ground_trajectory,config["test_case"],"Ground_truth_sym",config["training_size"],config["num_steps_test"],config["dt_test"])
    plot_energy_trajectory(
        test_case=config["test_case"],
        numerical_methods=config["numerical_methods"],
        dt=config["dt"],
        data=Ground_trajectory
    )
    num_body = Ground_trajectory.shape[1]
    print(f"Number of bodies: {num_body}")
    model = PNODE_Symplectic(num_bodys=num_body).to(device)
    calculate_model_parameters(model)
    trained_model=train_trajectory_PNODE_symplectic2(test_case=config["test_case"],
                                                    numerical_methods=config["numerical_methods"],
                                                    model=model,
                                                    body_tensor=Ground_trajectory,
                                                    training_size=config["training_size"],
                                                    num_epochs=config["num_epochs"],
                                                    dt=config["dt"])
    test_trajectory = test_PNODE_symplectic(numerical_methods=config["numerical_methods"],
                                            model=trained_model,
                                            body=Ground_trajectory[0],
                                            num_steps=config["num_steps_test"],
                                            dt=config["dt_test"])
    save_data(test_trajectory,config["test_case"],config["model_string"],config["training_size"],config["num_steps_test"],config["dt_test"])
    # Visualize the data
    plot_energy_trajectory(
        test_case=config["test_case"],
        numerical_methods=config["numerical_methods"],
        dt=config["dt"],
        data=test_trajectory
    )
    compare_trajectories(test_case=config["test_case"],numerical_methods=config["numerical_methods"],if_symplectic=config["if_symplectic"],model_string=config["model_string"],num_training=config["training_size"],dt=config["dt_test"], data1=Ground_trajectory, data2=test_trajectory)

    #plt.figure()
# run the main function
if __name__ == "__main__":
    main(config)
