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
    "num_epochs":500,
    "num_steps_test": 400,
    "dt_test": 0.01,
    "step_delay": 2,
    "random_seed": 0,
    "if_symplectic": False,
    "model_string": "PNODE_con",
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
    r=1.0
    for i in range(0,config["num_steps_test"]):
        constraint_loss = (-dxdt_values[i]-omega_values[i]*r*np.sin(theta_values[i]))**2
        print("constraint loss is "+str(constraint_loss))
    save_data(Ground_trajectory,config["test_case"],"Ground_truth",config["training_size"],config["num_steps_test"],config["dt_test"])
    num_body = Ground_trajectory.shape[1]
    print("The number of bodies is:",num_body)
    #model = PNODE(num_bodys=num_body,d_interest=1).to(device)
    model = PNODE(num_bodys=2,d_interest=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0007)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    print("optimizer is " + str(optimizer), "scheduler is " + str(scheduler))
    best_loss = float("inf")
    model_dir = f"models/{config['test_case']}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    losses = []
    r = 1.0  # length of crank
    l = 4.0  # length of connecting rod
    r_constant = 2.0  # Example value for r
    Ground_trajectory=Ground_trajectory[:,:,0:2]
    loss_main1_array = np.zeros(config["num_epochs"])
    loss_main2_array = np.zeros(config["num_epochs"])
    loss_main3_array = np.zeros(config["num_epochs"])
    loss_main4_array = np.zeros(config["num_epochs"])
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0
        print("epoch is " + str(epoch))
        for i in range(0, config["training_size"] - config["step_delay"]):
            optimizer.zero_grad()
            inputs = torch.tensor(Ground_trajectory[i, :, :].clone().to(device), dtype=torch.float32,requires_grad=True).to(device)
            inputs=inputs.reshape(1,4)
            #concatenate the inputs with the time
            inputs2=torch.cat((inputs,torch.tensor([[i*config["dt"]]],dtype=torch.float32,requires_grad=True).to(device)),dim=1)
            target = Ground_trajectory[i + config["step_delay"] - 1, :, :].clone().to(device)
            #accels is ((dd\theta),(ddx))
            accels = model(inputs2)
            Integrated_pred = torch.zeros_like(target).to(device)
            inputs=inputs.reshape(2,2)
            Integrated_pred[0,1]=inputs[0,1]+config["dt_test"]*accels[0,0]
            Integrated_pred[0,0]=inputs[0,0]+config["dt_test"]*inputs[0,1]
            #x_values[i] = r * np.cos(theta_values[i]) + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_values[i]) ** 2)
            #dxdt_values[i] = omega_values[i] * r * np.sin(theta_values[i])
            #if hard constraints applied, we need to use the following code
            theta_values=Integrated_pred[0,0].clone()
            omega_values=Integrated_pred[0,1].clone()
            Integrated_pred[1,0]= r*torch.cos(theta_values)+torch.sqrt(l**2-r**2*torch.sin(theta_values)**2)
            Integrated_pred[1,1]=-omega_values*r*torch.sin(theta_values)
            #otherwise, we use the soft constraints
            #Integrated_pred[1,1]=inputs[1,1]+config["dt_test"]*accels[0,1]
            #Integrated_pred[1,0]=inputs[1,0]+config["dt_test"]*inputs[1,1]
            Integrated_pred[0,0]=torch.fmod(Integrated_pred[0,0],2*np.pi)
            loss_main=criterion(Integrated_pred,target)
            #constrain loss because we know that the x and dx has relationship with the theta and dtheta
            #x_values[i] = r * np.cos(theta_values[i]) + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_values[i]) ** 2)
            #dxdt_values[i] = omega_values[i] * r * np.sin(theta_values[i])
            #here we use the constraint that x_values[i] = r * np.cos(theta_values[i]) + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_values[i]) ** 2)
            constraint_loss= torch.square(Integrated_pred[1,0]-r*torch.cos(Integrated_pred[0,0])-torch.sqrt(l**2-r**2*torch.sin(Integrated_pred[0,0])**2))
            #constraint_loss=criterion(-Integrated_pred[1,1],Integrated_pred[1,0]*r*torch.sin(Integrated_pred[0,0]))
            #print the constraint loss for the ground truth trajectory
            #we only optimize the constraint loss after 200 epochs
            #if epoch>400:
            #    loss=loss_main+constraint_loss
                #loss=loss_main
            #else:
            loss=loss_main
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("loss is " + str(epoch_loss))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("best loss is " + str(best_loss))
            model_path = f'{model_dir}/best_model_dt_epoch{epoch}.pth'
            torch.save(model.state_dict(), model_path)
        scheduler.step()
    save_model(model,config["test_case"],config["numerical_methods"],config["model_string"],config["training_size"],config["num_epochs"],config["dt"],config["step_delay"])
    #plot the loss_main1,loss_main2,loss_main3,loss_main4
    plt.figure()
    plt.plot(np.arange(0,config["num_epochs"]),loss_main1_array,label="loss_main1")
    plt.plot(np.arange(0,config["num_epochs"]),loss_main2_array,label="loss_main2")
    plt.plot(np.arange(0,config["num_epochs"]),loss_main3_array,label="loss_main3")
    plt.plot(np.arange(0,config["num_epochs"]),loss_main4_array,label="loss_main4")
    plt.legend()
    plt.show()

    #simulate the model to get the test trajectory
    #test_trajectory = torch.zeros(config["num_steps_test"], num_body, 3).to(device)
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
        Integrated_pred[1, 0] = r * torch.cos(Integrated_pred[0, 0]) + torch.sqrt(l ** 2 - r ** 2 * torch.sin(Integrated_pred[0, 0]) ** 2)
        Integrated_pred[1, 1] = -Integrated_pred[0, 1] * r * torch.sin(Integrated_pred[0, 0])
        #Integrated_pred[1,1]=inputs[1,1]+config["dt_test"]*accels[0,1]
        #Integrated_pred[1,0]=inputs[1,0]+config["dt_test"]*inputs[1,1]
        Integrated_pred[0,0]=torch.fmod(Integrated_pred[0,0],2*np.pi)
        test_trajectory[i+1,:,:]=Integrated_pred[:,:]
        #convert the theta to be in the range of [0,2pi]
        test_trajectory[i+1,0,0]=torch.fmod(test_trajectory[i+1,0,0],2*np.pi)
    mse = torch.nn.MSELoss()
    loss = mse(test_trajectory, Ground_trajectory)
    print("The MSE between the test trajectory and the ground truth trajectory is:", loss)
    save_data(test_trajectory,config["test_case"],config["model_string"],config["training_size"],config["num_steps_test"],config["dt_test"])
    #visualize the test trajectory
    plt.figure()
    plt.plot(t_test,test_trajectory.clone().cpu().detach().numpy()[:,0,0],label=r"$\theta$")
    plt.plot(t_test,test_trajectory.clone().cpu().detach().numpy()[:,0,1],label=r"$\omega$")
    plt.plot(t_test,test_trajectory.clone().cpu().detach().numpy()[:,1,0],label=r"$x$")
    plt.plot(t_test,test_trajectory.clone().cpu().detach().numpy()[:,1,1],label=r"$v$")
    #plt.plot(t_test,test_trajectory.clone().cpu().detach().numpy()[:,0,2],label="t1")
    plt.legend()
    plt.show()


    #compare_line_trajectories(test_case=config["test_case"], numerical_methods=config["numerical_methods"],dt=config["dt_test"], data1=Ground_trajectory, data2=test_trajectory)
if __name__ == "__main__":
    main(config)
