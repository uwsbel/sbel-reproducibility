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
    "step_delay": 16,
    "random_seed": 0,
    "if_symplectic": False,
    "model_string": "LSTM",
}

def main(config):
    set_seed(config["random_seed"])
    Ground_trajectory = generate_training_data(test_case=config["test_case"],numerical_methods=config["generation_numerical_methods"],dt=config["dt"],num_steps=config["num_steps"])
    print(Ground_trajectory.shape)
    #plot_energy_trajectory(test_case=config["test_case"],numerical_methods=config["numerical_methods"],dt=config["dt"],data=Ground_trajectory)
    num_body = Ground_trajectory.shape[1]
    LSTMmodel = LSTMModel(num_body=num_body).to(device)
    #get parameters number
    calculate_model_parameters(LSTMmodel)
    scaler,trained_LSTM=train_trajectory_LSTM(test_case=config["test_case"],model=LSTMmodel,body_tensor=Ground_trajectory, step_delay=config["step_delay"],training_size=config["training_size"], num_epochs=config["num_epochs"], dt=config["dt"])
    test_trajectory = Infer_LSTM(test_case=config["test_case"],body_tensor=Ground_trajectory[0:config["step_delay"]],scaler=scaler,num_steps=config["num_steps_test"],model=LSTMmodel,step_delay=config["step_delay"])
    save_data(test_trajectory,config["test_case"],config["model_string"],config["training_size"],config["num_steps_test"],config["dt_test"])
    save_model(trained_LSTM,config["test_case"],config["numerical_methods"],config["model_string"],config["training_size"],config["num_epochs"],config["dt"],config["step_delay"])
    plot_energy_trajectory(test_case=config["test_case"], numerical_methods=config["numerical_methods"], dt=config["dt_test"], data=test_trajectory,test=True)
    compare_trajectories(test_case=config["test_case"], numerical_methods=config["numerical_methods"],if_symplectic=config["if_symplectic"],model_string=config["model_string"],num_training=config["training_size"],dt=config["dt_test"], data1=Ground_trajectory, data2=test_trajectory)

if __name__ == "__main__":
    main(config)
