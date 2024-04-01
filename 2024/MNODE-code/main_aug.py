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
    "random_seed": 42,
    "if_symplectic": False,
    "model_string": "PNODE",
}
def main(config):
    set_seed(config["random_seed"])
    t_test=np.linspace(0,config["dt_test"]*config["num_steps_test"],config["num_steps_test"])
    Ground_trajectory = generate_training_data(test_case=config["test_case"],numerical_methods=config["generation_numerical_methods"],dt=config["dt"],num_steps=config["num_steps"])
    print(Ground_trajectory.shape)
    #visualize the ground truth trajectory
    plt.figure()
    plt.plot(t_test,Ground_trajectory.clone().cpu().detach().numpy()[:,0,0],label="theta")
    plt.plot(t_test,Ground_trajectory.clone().cpu().detach().numpy()[:,0,1],label="omega")
    plt.plot(t_test,Ground_trajectory.clone().cpu().detach().numpy()[:,1,0],label="x")
    plt.plot(t_test,Ground_trajectory.clone().cpu().detach().numpy()[:,1,1],label="dx")
    #plt.plot(t_test,Ground_trajectory.clone().cpu().detach().numpy()[:,0,2],label="t1")
    plt.legend()
    plt.show()
    #compute the constraint loss for the whole trajectory
    theta_values = Ground_trajectory.clone().cpu().detach().numpy()[:,0,0]
    omega_values = Ground_trajectory.clone().cpu().detach().numpy()[:,0,1]
    x_values = Ground_trajectory.clone().cpu().detach().numpy()[:,1,0]
    dxdt_values = Ground_trajectory.clone().cpu().detach().numpy()[:,1,1]
    r = 1.0  # length of crank
    l = 4.0  # length of connecting rod
    r_constant = 2.0  # Example value for r
    for i in range(0,config["num_steps_test"]):
        constraint_loss2 = (-dxdt_values[i]-omega_values[i]*r*np.sin(theta_values[i]))**2
        constraint_loss1 = (x_values[i]-r*np.cos(theta_values[i])-np.sqrt(l**2-r**2*np.sin(theta_values[i])**2))**2
        print("constraint loss1 is "+str(constraint_loss1))
        print("constraint loss2 is "+str(constraint_loss2))
    save_data(Ground_trajectory,config["test_case"],"Ground_truth",config["training_size"],config["num_steps_test"],config["dt_test"])
    num_body = Ground_trajectory.shape[1]
    print("The number of bodies is:",num_body)
    model = PNODE(num_bodys=2,d_interest=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0006)
    mu=0.000001
    mu_mul=0.99
    lam1=0
    lam2=0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    print("optimizer is " + str(optimizer), "scheduler is " + str(scheduler))
    best_loss = float("inf")
    model_dir = f"models/{config['test_case']}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    Ground_trajectory=Ground_trajectory[:,:,0:2]
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0
        print("epoch is " + str(epoch))
        epoch_constraint_loss1=0
        epoch_constraint_loss2 = 0
        for i in range(0, config["training_size"] - config["step_delay"]):
            optimizer.zero_grad()
            inputs = torch.tensor(Ground_trajectory[i, :, :].clone().to(device), dtype=torch.float32,requires_grad=True).to(device)
            inputs=inputs.reshape(1,4)
            inputs2=torch.cat((inputs,torch.tensor([[i*config["dt"]]],dtype=torch.float32,requires_grad=True).to(device)),dim=1)
            target = Ground_trajectory[i + config["step_delay"] - 1, :, :].clone().to(device)
            accels = model(inputs2)
            Integrated_pred = torch.zeros_like(target).to(device)
            inputs=inputs.reshape(2,2)
            #using the forward euler method to integrate the acceleration to get the velocity and position
            Integrated_pred[0,1]=inputs[0,1]+config["dt_test"]*accels[0,0]
            Integrated_pred[1,1]=inputs[1,1]+config["dt_test"]*accels[0,1]
            Integrated_pred[0, 0] = inputs[0, 0] + config["dt_test"] * inputs[0, 1]
            Integrated_pred[1,0]=inputs[1,0]+config["dt_test"]*inputs[1,1]
            Integrated_pred[0,0]=torch.fmod(Integrated_pred[0,0],2*np.pi)
            loss_main=criterion(Integrated_pred,target)
            #constrain loss because we know that the x and dx has relationship with the theta and dtheta
            #x_values[i] = r * np.cos(theta_values[i]) + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_values[i]) ** 2)
            #dxdt_values[i] = omega_values[i] * r * np.sin(theta_values[i])
            #here we use the constraint that x_values[i] = r * np.cos(theta_values[i]) + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_values[i]) ** 2)
            constraint_loss1_1= torch.square(Integrated_pred[1,0]-r*torch.cos(Integrated_pred[0,0])-torch.sqrt(l**2-r**2*torch.sin(Integrated_pred[0,0])**2))
            constraint_loss1_2= -(Integrated_pred[1,0]-r*torch.cos(Integrated_pred[0,0])-torch.sqrt(l**2-r**2*torch.sin(Integrated_pred[0,0])**2))
            #here we use the constraint that dxdt_values[i] = omega_values[i] * r * np.sin(theta_values[i])
            constraint_loss2_1= torch.square(Integrated_pred[1,1]+Integrated_pred[0,1]*r*torch.sin(Integrated_pred[0,0]))
            constraint_loss2_2= -(Integrated_pred[1,1]+Integrated_pred[0,1]*r*torch.sin(Integrated_pred[0,0]))

            #loss=loss_main+constraint_loss1_1*mu+lam1*constraint_loss1_2+constraint_loss2_1*mu+lam2*constraint_loss2_2
            loss=loss_main
            epoch_loss += loss.item()
            epoch_constraint_loss1+=constraint_loss1_2.item()
            epoch_constraint_loss2+=constraint_loss2_2.item()
            loss.backward()
            optimizer.step()
        print("loss is " + str(epoch_loss),"constraint loss is " + str(epoch_constraint_loss1),"constraint loss2 is " + str(epoch_constraint_loss2))
        print("lam1 is " + str(lam1),"lam2 is" +str(lam2), "mu is " + str(mu))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("best loss is " + str(best_loss))
            model_path = f'{model_dir}/best_model_dt_epoch{epoch}.pth'
            torch.save(model.state_dict(), model_path)
        scheduler.step()
        mu=mu*mu_mul
        lam1=lam1+2*mu*epoch_constraint_loss1
        lam2=lam2+2*mu*epoch_constraint_loss2
    save_model(model,config["test_case"],config["numerical_methods"],config["model_string"],config["training_size"],config["num_epochs"],config["dt"],config["step_delay"])

    test_trajectory = torch.zeros(config["num_steps_test"],num_body,2).to(device)
    test_trajectory[0,:,:]=Ground_trajectory[0,:,:]
    for i in range(0,config["num_steps_test"]-1):
        inputs = test_trajectory[i, :, :].clone().to(device)
        Integrated_pred = torch.zeros_like(inputs).to(device)
        # reshape the inputs to be (1,6)
        #inputs = inputs.reshape(1, 6)
        inputs = inputs.reshape(1, 4)
        # concatenate the inputs with the time
        inputs2 = torch.cat((inputs, torch.tensor([[i * config["dt_test"]]], dtype=torch.float32, requires_grad=True).to(device)), dim=1)
        accels = model(inputs2)
        inputs = inputs.reshape(2, 2)
        Integrated_pred[0,1]=inputs[0,1]+config["dt_test"]*accels[0,0]
        Integrated_pred[0,0]=inputs[0,0]+config["dt_test"]*inputs[0,1]
        Integrated_pred[1,1]=inputs[1,1]+config["dt_test"]*accels[0,1]
        Integrated_pred[1,0]=inputs[1,0]+config["dt_test"]*inputs[1,1]
        Integrated_pred[0,0]=torch.fmod(Integrated_pred[0,0],2*np.pi)
        test_trajectory[i+1,:,:]=Integrated_pred[:,:]
        #convert the theta to be in the range of [0,2pi]
        test_trajectory[i+1,0,0]=torch.fmod(test_trajectory[i+1,0,0],2*np.pi)
    mse = torch.nn.MSELoss()
    loss = mse(test_trajectory, Ground_trajectory)
    print("The MSE between the test trajectory and the ground truth trajectory is:", loss)
    save_data(test_trajectory,config["test_case"],config["model_string"],config["training_size"],config["num_steps_test"],config["dt_test"])
    #we also calculate the single step simulation, which means we use the ground truth to predict the next step
    test_trajectory2 = torch.zeros(config["num_steps_test"],num_body,2).to(device)
    test_trajectory2[0,:,:]=Ground_trajectory[0,:,:]
    for i in range(0,config["num_steps_test"]-1):
        #input is the ground truth of the previous step
        inputs = Ground_trajectory[i, :, :].clone().to(device)
        Integrated_pred = torch.zeros_like(inputs).to(device)
        # reshape the inputs to be (1,6)
        #inputs = inputs.reshape(1, 6)
        inputs = inputs.reshape(1, 4)
        # concatenate the inputs with the time
        inputs2 = torch.cat((inputs, torch.tensor([[i * config["dt_test"]]], dtype=torch.float32, requires_grad=True).to(device)), dim=1)
        accels = model(inputs2)
        inputs = inputs.reshape(2, 2)
        Integrated_pred[0,1]=inputs[0,1]+config["dt_test"]*accels[0,0]
        Integrated_pred[0,0]=inputs[0,0]+config["dt_test"]*inputs[0,1]
        Integrated_pred[1,1]=inputs[1,1]+config["dt_test"]*accels[0,1]
        Integrated_pred[1,0]=inputs[1,0]+config["dt_test"]*inputs[1,1]
        Integrated_pred[0,0]=torch.fmod(Integrated_pred[0,0],2*np.pi)
        test_trajectory2[i+1,:,:]=Integrated_pred[:,:]
        #convert the theta to be in the range of [0,2pi]
        test_trajectory2[i+1,0,0]=torch.fmod(test_trajectory2[i+1,0,0],2*np.pi)
    mse = torch.nn.MSELoss()
    loss = mse(test_trajectory2, Ground_trajectory)
    print("The MSE between the test trajectory and the ground truth trajectory is:", loss)
    save_data(test_trajectory2,config["test_case"],config["model_string"]+"single_step",config["training_size"],config["num_steps_test"],config["dt_test"])
    plt.figure()
    plt.plot(t_test,test_trajectory.clone().cpu().detach().numpy()[:,0,0],label=r"$\theta$")
    plt.plot(t_test,test_trajectory.clone().cpu().detach().numpy()[:,0,1],label=r"$\omega$")
    plt.plot(t_test,test_trajectory.clone().cpu().detach().numpy()[:,1,0],label=r"$x$")
    plt.plot(t_test,test_trajectory.clone().cpu().detach().numpy()[:,1,1],label=r"$v$")
    #plt.plot(t_test,test_trajectory.clone().cpu().detach().numpy()[:,0,2],label="t1")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main(config)
