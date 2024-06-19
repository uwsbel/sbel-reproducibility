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
    "test_case": "Single_Pendulum_P",
    "generation_numerical_methods": "midpoint",
    "numerical_methods": "midpoint",
    "dt": 0.01,
    "num_steps": 300000,
    "training_size": 300000,
    "num_epochs":400,
    "num_steps_test": 300000,
    "dt_test": 0.01,
    "step_delay": 2,
    "random_seed": 5,
    "if_symplectic": False,
    "model_string": "PNODE",
}
def main(config):
    set_seed(config["random_seed"])
    body_tensor, p_tensor, accel_tensor = generate_sampling_training_data(test_case=config["test_case"],
                                                                          numerical_methods=config[
                                                                              "generation_numerical_methods"],
                                                                          dt=config["dt"],
                                                                          num_steps=config["num_steps"])
    print(body_tensor.shape)
    print(p_tensor.shape)
    print(accel_tensor.shape)
    num_body = body_tensor.shape[1]
    model = PNODE(num_bodys=num_body, d_interest=1).to(device)
    trained_model = train_force_batch_PNODE(test_case=config["test_case"],
                                            numerical_methods=config["numerical_methods"], model=model,
                                            body_tensor=body_tensor, force_tensor=p_tensor, accel_tensor=accel_tensor,
                                            training_size=config["training_size"], num_epochs=config["num_epochs"],
                                            dt=config["dt"])
    # save the trained model
    torch.save(trained_model, "trained_model_SP3.pt")
if __name__ == "__main__":
    main(config)
