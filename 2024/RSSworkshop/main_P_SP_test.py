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
from tqdm import tqdm
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
    "dt": 0.05,
    "num_steps": 100,
    "training_size": 100,
    "num_epochs":400,
    "num_steps_test": 10,
    "dt_test": 0.05,
    "step_delay": 2,
    "random_seed": 3,
    "if_symplectic": False,
    "model_string": "PNODE",
}
def main(config):
    set_seed(config["random_seed"])
    trained_model = torch.load("trained_model_SP2.pt")
    # Ensure the model is in evaluation mode
    trained_model.eval()
    print(trained_model)
    body_init = torch.tensor([[1, 0]], dtype=torch.float32).to(device)
    # generate the random force and torque in the range of theta_min = 0 theta_max = 2*np.pi theta_dot_min=-8theta_dot_max=8 F_min = -5 F_max = 5 Tao_min = -5   Tao_max = 5
    p_tensor = torch.tensor([[0.4]], dtype=torch.float32).to(device)
    p_value_np = p_tensor.cpu().detach().numpy()
    #reshape the p_value_np
    p_value_np = p_value_np.reshape(-1)

    current_state = torch.cat((body_init, p_tensor), 1)
    # print(current_state)
    ground_current_state = torch.cat((body_init, p_tensor), 1)
    trajectorys = []
    ground_trajectorys = []
    forward_euler = True
    # print(current_state.shape)
    for i in tqdm(range(config["num_steps_test"] - 1)):
        # use the trained model to predict the acceleration
        if forward_euler == True:
            print("current_state:",current_state)
            print("ground_current_state:",ground_current_state)
            pred_accel_tensor = trained_model(current_state) / 1
            ground_accel_tensor = p_force_sp(ground_current_state,p_tensor)
            next_theta_dot = current_state[:, 1] + pred_accel_tensor[:, 0] * config["dt_test"]
            ground_next_theta_dot = ground_current_state[:, 1] + ground_accel_tensor * config["dt_test"]
            next_theta = current_state[:, 0] + current_state[:, 1] * config["dt_test"]
            ground_next_theta = ground_current_state[:, 0] + ground_current_state[:, 1] * config["dt_test"]
            # if theta is out of the range, then reset it to the range
            next_theta = torch.where(next_theta > np.pi, next_theta - 2 * np.pi, next_theta)
            next_theta = torch.where(next_theta < -np.pi, next_theta + 2 * np.pi, next_theta)
            next_state = torch.cat(
                (next_theta.reshape(1, 1), next_theta_dot.reshape(1, 1), p_tensor), 1)
            ground_next_theta = torch.where(ground_next_theta >np.pi, ground_next_theta - 2 * np.pi,
                                            ground_next_theta)
            ground_next_theta = torch.where(ground_next_theta < -np.pi, ground_next_theta + 2 * np.pi,
                                            ground_next_theta)
            ground_next_state = torch.cat((ground_next_theta.reshape(1, 1), ground_next_theta_dot.reshape(1, 1),
                                           p_tensor), 1)
            current_state = next_state
            ground_current_state = ground_next_state
            trajectorys.append(current_state)
            ground_trajectorys.append(ground_current_state)
    #visualize the trajectory, compare the ground truth and the predicted trajectory
    trajectorys = torch.cat(trajectorys, 0)
    ground_trajectorys = torch.cat(ground_trajectorys, 0)
    """
    print(trajectorys.shape)
    print(ground_trajectorys.shape)
    plt.figure()
    plt.plot(ground_trajectorys[:, 0], ground_trajectorys[:, 1], label="ground truth trajectory",color="red",linestyle="solid")
    plt.plot(trajectorys[:, 0], trajectorys[:, 1], label="predicted trajectory",color="blue",linestyle="dashed")
    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.legend()
    plt.show()
    """
    # we need to define the number of iterations
    num_iterations = 50
    loss_record = []
    l_record=[]

    current_xv = torch.tensor([[1.0, 0.0]], requires_grad=True,dtype=torch.float32).to(device)
    current_xv = nn.Parameter(current_xv.to(device))
    current_l=torch.tensor([[0.5]], requires_grad=True,dtype=torch.float32).to(device)
    current_l = nn.Parameter(current_l.to(device))

    #optimizer = optim.Adam([current_l], lr=1.0)
    #optimizer = optim.LBFGS([current_l,current_xv], lr=0.1)
    optimizer = optim.LBFGS([current_l], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    external_loss_value = None
    #loss_record.append(loss)
    for i in tqdm(range(num_iterations)):
        def closure():
            nonlocal external_loss_value
            optimizer.zero_grad()
            current_state = torch.cat((current_xv, current_l), 1)
            loss=0
            for i in range(config["num_steps_test"] - 1):
                pred_accel_tensor = trained_model(current_state) / 1
                next_theta_dot = current_state[:, 1] + pred_accel_tensor[:, 0] * config["dt_test"]
                next_theta = current_state[:, 0] + current_state[:, 1] * config["dt_test"]
                next_theta = torch.where(next_theta > np.pi, next_theta - 2 * np.pi, next_theta)
                next_theta = torch.where(next_theta < -np.pi, next_theta + 2 * np.pi, next_theta)
                next_xv = torch.cat((next_theta.reshape(1, 1), next_theta_dot.reshape(1, 1)), 1)
                next_state = torch.cat((next_theta.reshape(1, 1), next_theta_dot.reshape(1, 1), p_tensor), 1)
                current_state = next_state
                #print(next_xv,ground_trajectorys[i,0:2])
                mse_loss = nn.MSELoss()
                loss1=mse_loss(next_xv,ground_trajectorys[i,0:2])
                loss=loss+loss1
            loss.backward()
            external_loss_value = loss.item()
            print("next_xv:",next_xv)
            print("ground_trajectorys[i,0:2]:",ground_trajectorys[i,0:2])
            #print("loss value:", external_loss_value)
            return loss
        print("loss value:", external_loss_value)
        loss_record.append(external_loss_value)
        optimizer.step(closure)
        """
        optimizer.zero_grad()
        current_state = torch.cat((current_xv, current_l), 1)
        loss=0
        trajectorys = []
        for i in range(config["num_steps_test"] - 1):
            pred_accel_tensor = trained_model(current_state) / 1
            next_theta_dot = current_state[:, 1] + pred_accel_tensor[:, 0] * config["dt_test"]
            next_theta = current_state[:, 0] + current_state[:, 1] * config["dt_test"]
            next_theta = torch.where(next_theta > 2 * np.pi, next_theta - 2 * np.pi, next_theta)
            next_xv = torch.cat((next_theta.reshape(1, 1), next_theta_dot.reshape(1, 1)), 1)
            next_state = torch.cat((next_theta.reshape(1, 1), next_theta_dot.reshape(1, 1), p_tensor), 1)
            current_state = next_state
            #print(next_xv,ground_trajectorys[i,0:2])
            mse_loss = nn.MSELoss()  # Instantiate the MSE loss function
            loss1=mse_loss(next_xv,ground_trajectorys[i,0:2])
            loss=loss+loss1
        loss_record.append(loss)
        loss.backward()
        optimizer.step()
        """
        #scheduler.step()
        #print the loss value and the parameter l
        #print("loss value:",loss)
        print("l:",current_l)
        print("xv:",current_xv)
        l_record.append(current_l.item())
    print(loss_record)


    trajectorys = trajectorys.cpu().detach().numpy()
    ground_trajectorys = ground_trajectorys.cpu().detach().numpy()


    #make the three figures in one figure using subplot
    plt.figure(figsize=(10, 10))
    plt.subplot(3,1,1)
    plt.plot(loss_record, color="blue", linestyle="dashed")
    plt.xlabel("iteration")
    plt.ylabel("loss value")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot([p_value_np]*num_iterations,color="red",linestyle="solid",label=r"ground truth $L$")
    plt.plot(l_record, color="blue", linestyle="dashed", label=r"optimized $L$")
    plt.xlabel("iteration")
    plt.ylabel(r"$L$")
    plt.grid()
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(ground_trajectorys[:, 0], ground_trajectorys[:, 1], label="ground truth trajectory",color="red",linestyle="solid")
    plt.plot(trajectorys[:, 0], trajectorys[:, 1], label="predicted trajectory",color="blue",linestyle="dashed")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.legend()
    plt.grid()
    plt.savefig("Single_Pendulum_P3.png")
    plt.show()
    #print the optimized l
    print("optimized l:",current_l)
    #print the ground truth l
    print("ground truth l:",p_tensor)




if __name__ == "__main__":
    main(config)
