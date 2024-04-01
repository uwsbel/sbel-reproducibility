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
    "test_case": "Single_Mass_Spring",
    "generation_numerical_methods": "rk4",
    "numerical_methods": "rk4",
    "dt": 0.01,
    "num_steps": 3000,
    "training_size": 300,
    "num_epochs":300,
    "num_steps_test": 3000,
    "dt_test": 0.01,
    "step_delay": 2,
    "random_seed": 0,
    "if_symplectic": False,
    "model_string": "PNODE",
}
def main(config):
    set_seed(config["random_seed"])
    Ground_trajectory = generate_training_data(test_case=config["test_case"],numerical_methods=config["generation_numerical_methods"],dt=config["dt"],num_steps=config["num_steps"])
    save_data(Ground_trajectory,config["test_case"],"RK4",config["training_size"],config["num_steps_test"],config["dt_test"])

    #compare_line_trajectories(test_case=config["test_case"], numerical_methods=config["numerical_methods"],dt=config["dt_test"], data1=Ground_trajectory, data2=test_trajectory)
if __name__ == "__main__":
    main(config)
