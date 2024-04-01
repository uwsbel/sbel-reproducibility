# author: Jingquan Wang
# This is the main file for the project
# import all the packages needed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint
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
    "generation_numerical_methods": "hf",
    "numerical_methods": "rk4",
    "dt": 0.01,
    "num_steps": 3000,
    "training_size": 300,
    "num_epochs":400,
    "num_steps_test": 3000,
    "dt_test": 0.01,
    "step_delay": 2,
    "random_seed": 0,
    "if_symplectic": False,
    "model_string": "LNN",
}
def main(config):
    set_seed(config["random_seed"])
    Ground_trajectory = generate_training_data(test_case=config["test_case"],numerical_methods=config["generation_numerical_methods"],dt=config["dt"],num_steps=config["num_steps"])
    plot_energy_trajectory(test_case=config["test_case"],numerical_methods=config["numerical_methods"],dt=config["dt"],data=Ground_trajectory)
    num_body = Ground_trajectory.shape[1]
    model = LNN(num_bodys=num_body).to(device)
    print(model)
    t_train = torch.tensor(np.linspace(0, config["dt"]*config["training_size"], config["training_size"])).float()
    t_test = torch.tensor(np.linspace(0, config["dt"]*config["num_steps"], config["num_steps"])).float()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    best_loss = float("inf")
    b=Ground_trajectory.clone().cpu().detach().numpy()
    derivatives = get_xt_anal(b.reshape(3000,1,2), t_test)
    derivatives=torch.tensor(derivatives).float().to(device)
    print("derivatives shape:",derivatives.shape)
    plt.figure()
    a=derivatives.clone().cpu().detach().numpy()
    plt.plot(a[:,:,0],label="dx=v")
    plt.plot(a[:,:,1],label="dv=a")
    plt.legend()
    plt.figure()
    plt.plot(b[:,:,0],label="x")
    plt.plot(b[:,:,1],label="v")
    plt.plot(a[:,:,0]-b[:,:,1],label="dx=v-v")
    plt.legend()
    plt.show()
    print(derivatives)
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0
        for i in range(1, config["training_size"]):
            optimizer.zero_grad()
            inputs = Ground_trajectory[i,:,:]
            target = derivatives[i,:,:]
            pred_derivative = model(inputs)
            #print(pred_derivative.shape)
            #print(target, pred_derivative)
            #print(pred_derivative,target)
            loss = criterion(pred_derivative, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('[%d] loss: %.10f' %
              (epoch + 1, epoch_loss))
        scheduler.step()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            trained_model = model

    nn_test = lnn_solve_ode(model, Ground_trajectory[0,0,:], t_test)
    save_data_np(nn_test,config["test_case"],config["model_string"],config["training_size"],config["num_steps_test"],config["dt_test"])
    plt.figure()
    #plt.plot(t_train, x_train[:, 0], label='analytical_train')
    plt.plot(t_test, nn_test[:, 0], label='nn_train')
    plt.legend()
    plt.title('Training data')

    # plt.figure()
    # plt.plot(t_test, x_test[:, 0], label='analytical_test')
    # nn_test2 = nn_solve_ode(model, x_test[0], t_test)
    # plt.plot(t_test, nn_test2[:, 0], label='nn_test')
    # plt.title('Test data')
    # plt.legend()
    save_model(trained_model,config["test_case"],config["numerical_methods"],config["model_string"],config["training_size"],config["num_epochs"],config["dt"],config["step_delay"])
    plt.show()
if __name__ == "__main__":
    main(config)
