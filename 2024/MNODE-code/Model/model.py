#Author: Jingquan Wang
#This file contains all the basic functions for the parameterized Nerual ODE model for multiple body system project
#This file is organized as follows:
#1. The integrator used for the project: Contains the forward Euler method and the RK4 method, and the symplectic integrators
#2. The force function for numerical tests for the multiple body system
#3. The function used to generate the training and testing data for the multiple body system, maybe with external force and noise
#4.The Pytorch module for the parameterized Nerual ODE model for multiple body system and the baseline model like
# the Hamiltonian Neural Network, Lagrangian Neural Network, BiLSTM model and the Feedforward Neural Network model
#5. The function used to train the model
#6. The visualization function used to visualize the training and testing results
#7. The function used to calculate the error for the model

import numpy as np
from numpy import cos, sin
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint
from functools import partial
from torch.autograd.functional import jacobian, hessian
from torchdiffeq import odeint as tor_odeint
from torchdiffeq import odeint_adjoint as tor_odeintadj
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import fsolve
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import scipy
from utils import *
from force_fun import *
from Integrator import *
from Data_generator import *
from sklearn.preprocessing import MinMaxScaler
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#The function used to generate the training and testing data for the multiple body system, maybe with external force and noise
#Input: test_case is a string that indicates which test case we are using
#       num_bodys is the number of bodys we want to simulate
#       T is the time we want to simulate
#       dt is the time step we want to use
#       n_samples is the number of samples we want to generate
#       if_noise is a boolean variable that indicates whether we want to add noise to the data
#       if_external_force is a boolean variable that indicates whether we want to add external force to the data
#       external_force_function is a function that takes in time and returns the external force
#       if_return_tensor is a boolean variable that indicates whether we want to return tensor or numpy array
#Output: The training or testing data, with shape (n_samples,num_bodys,2), if if_return_tensor is True
#        The training or testing data, with shape (n_samples,num_bodys,2), if if_return_tensor is False


#Now we start to define the models
#The first is our model, the parameterized Nerual ODE model for multiple body system
#The model is a Pytorch module, with the following parameters:
#       num_bodys is the number of bodys we want to simulate
class PNODE(nn.Module):
    def __init__(self, num_bodys=1, layers=3, width=256, d_interest=0,activation='tanh', initializer='xavier'):
        super(PNODE, self).__init__()

        # Determine activation
        self.act = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }[activation]

        # Construct the neural network layers
        self.d_interest=d_interest
        self.layers = layers
        self.num_bodys = num_bodys
        self.dim = 2 * num_bodys+self.d_interest
        self.width = width
        #we want to use layer normalization here
        module_list = [nn.Linear(self.dim, self.width), self.act]
        for _ in range(layers - 2):
            module_list.extend([nn.Linear(self.width, self.width), self.act])

        module_list.append(nn.Linear(self.width, self.num_bodys))

        self.network = nn.Sequential(*module_list)

        # Apply initializer if specified
        if initializer == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        if initializer == 'kaiming':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.network(x)
class LSTMModel(nn.Module):
    def __init__(self,num_body=1, hidden_size=256, num_layers=3):
        super(LSTMModel, self).__init__()
        self.num_body = num_body
        self.input_dim = 2 * num_body
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = 2 * num_body
        #Use unbatched LSTM

        self.lstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,bias=True)
        self.fc = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        #print("out shape is " + str(out.shape))
        #print("out[-1,:] shape is "+str(out[-1,:].shape))
        out = self.fc(out[ -1,:])
        #print("out shape is "+str(out.shape))
        return out

class FCNN(nn.Module):
    def __init__(self, num_bodys=1, layers=3, width=256,d_input_interest=0,activation='tanh', initializer='xavier'):
        super(FCNN, self).__init__()
        # Determine activation
        self.act = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }[activation]
        # Construct the neural network layers
        self.layers = layers
        self.num_bodys = num_bodys
        self.d_input_interest=d_input_interest
        self.dim = 1+self.d_input_interest
        self.width = width
        #we want to use layer normalization here
        module_list = [nn.Linear(self.dim, self.width), self.act]
        for _ in range(layers - 2):
            module_list.extend([nn.Linear(self.width, self.width), self.act])
        module_list.append(nn.Linear(self.width, self.num_bodys*2))
        self.network = nn.Sequential(*module_list)
        # Apply initializer if specified
        if initializer == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        if initializer == 'kaiming':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
    def forward(self, x):
        return self.network(x)

class PNODE_Symplectic(nn.Module):
    def __init__(self, num_bodys=1, layers=3, width=200, activation='relu', initializer='xavier'):
        super(PNODE_Symplectic, self).__init__()

        # Determine activation
        self.act = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }[activation]

        # Construct the neural network layers
        self.layers = layers
        self.num_bodys = num_bodys
        self.dim = 2 * num_bodys
        self.width = width

        module_list = [nn.Linear(self.dim, self.width), self.act]
        for _ in range(self.layers - 2):
            module_list.extend([nn.Linear(self.width, self.width), self.act])
        module_list.append(nn.Linear(self.width, self.dim))
        self.network = nn.Sequential(*module_list)
        # Apply initializer if specified
        if initializer == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
    def forward(self, x):
        return self.network(x)
def neural_network_force_function_PNODE(body_tensor,model):
    #print("entern neural_network_force_function_PNODE function")
    #mass=torch.tensor([100,10,1],dtype=torch.float32,device=device)
    mass=torch.tensor([10],dtype=torch.float32,device=device)
    #mass=torch.tensor([1,1],dtype=torch.float32,device=device)
    inputs = body_tensor.view( -1).clone().to(device).requires_grad_(True)
    #print(inputs.shape)
    f_pred = model(inputs)
    acceleration_pred = f_pred/mass
    return acceleration_pred
def neural_network_force_function_PNODE_external(body_tensor,model,external_force=None):
    #print("entern neural_network_force_function_PNODE function")
    #mass=torch.tensor([100,10,1],dtype=torch.float32,device=device)
    mass=torch.tensor([10],dtype=torch.float32,device=device)
    #mass=torch.tensor([1,1],dtype=torch.float32,device=device)
    inputs = body_tensor.view( -1).clone().to(device).requires_grad_(True)
    #print(inputs.shape)
    f_pred = model(inputs)
    acceleration_pred = f_pred/mass+external_force/mass
    return acceleration_pred

def neural_network_force_function_PNODE_symplectic_dH_dq(body_tensor,model):
    #body_tensor is a tensor with shape (num_bodys,2),the first column is the generalized coordinate, the second column is the generalized momentum
    #print("entern neural_network_force_function_PNODE_symplectic_coordinate function")
    #print("body_tensor shape is "+str(body_tensor.shape))
    body_num=body_tensor.shape[0]
    inputs = body_tensor.view(-1).clone().to(device).requires_grad_(True)
    #print(inputs.shape)
    #print("inputs is "+str(inputs),"inputs shape is "+str(inputs.shape))
    #print("model is "+str(model))
    f_pred = model(inputs)
    #extract the gradient of the Hamiltonian with respect to the generalized coordinate
    return f_pred[0:body_num]
def neural_network_force_function_PNODE_symplectic_dH_dp(body_tensor,model):
    #body_tensor is a tensor with shape (num_bodys,2),the first column is the generalized coordinate, the second column is the generalized momentum
    #print("entern neural_network_force_function_PNODE_symplectic_momentum function")
    #print("body_tensor shape is "+str(body_tensor.shape))
    body_num=body_tensor.shape[0]
    #print("body_num is "+str(body_num),"body_tensor is "+str(body_tensor))
    inputs = body_tensor.view(-1).clone().to(device).requires_grad_(True)
    #print("model is "+str(model))
    #print(inputs.shape)
    f_pred = model(inputs)
    #extract the gradient of the Hamiltonian with respect to the generalized momentum
    return f_pred[body_num:2*body_num]
def train_trajectory_PNODE(test_case, numerical_methods, model, body_tensor, training_size,
                           step_delay=2, num_epochs=450, dt=0.01, device='cuda', verbose=True):
    # Setup logging
    def log(message):
        if verbose:
            print(message)
    log("Starting training of the PNODE model")
    log(f"body_tensor shape: {body_tensor.shape}")
    log(f"Model: {model}")
    criterion = nn.MSELoss()
    #optimizer = optim.RAdam(model.parameters(), lr=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    print("optimizer is "+str(optimizer),"scheduler is "+str(scheduler))
    best_loss = float("inf")
    # Numerical methods mapping
    methods = {
        "fe": forward_euler_multiple_body,
        "rk4": runge_kutta_four_multiple_body,
        "midpoint": midpoint_method_multiple_body,
    }
    if numerical_methods not in methods:
        raise ValueError("The numerical method is not specified correctly")
    integration_method = methods[numerical_methods]
    # Create directory if it doesn't exist
    model_dir = f"models/{test_case}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        #log(f"Epoch: {epoch}")
        for i in range(0, training_size - step_delay):
            optimizer.zero_grad()
            inputs = body_tensor[i, :, :].clone().to(device).requires_grad_(True)
            target = body_tensor[i + step_delay - 1, :, :].clone().to(device)
            #target = body_tensor[i:i + step_delay, :, :].clone().to(device)
            Integrated_pred = integration_method(inputs, neural_network_force_function_PNODE, step_delay, dt,if_final_state=True, model=model)
            #print("target shape is "+str(target.shape))
            #Integrated_pred = integration_method(inputs, neural_network_force_function_PNODE, step_delay, dt,if_final_state=False, model=model)
            loss1 = criterion(Integrated_pred, target)
            #energy_loss=criterion(total_energy_sms(Integrated_pred),total_energy_sms(inputs))
            #loss = criterion((Integrated_pred-inputs)/dt, (target-inputs)/dt)#+criterion(Integrated_pred, target)
            loss=loss1
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("loss is "+str(loss.item()))
        #losses.append(epoch_loss / training_size)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            log(f"Best loss: {best_loss}")
            model_path = f'{model_dir}/best_model_{numerical_methods}_dt{dt}_epoch{epoch}.pth'
            torch.save(model.state_dict(), model_path)
        scheduler.step()
        log(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / training_size:.10f}')
    return model

def train_trajectory_LSTM(test_case,model,body_tensor,training_size,step_delay=1,num_epochs=450,dt=0.01,device='cuda'):
    #This function is used to train the PNODE model
    #model is the PNODE model we want to train
    #body_tensor is the training data we want to use with shape (num_data,num_bodys,2)
    #training_size is the number of data we want to use for training, training_size<=num_data
    #num_epochs is the number of epochs we want to use for training
    #dt is the time step we use for the training data
    print("start training the LSTM model")
    print("body_tensor shape is "+str(body_tensor.shape))
    print("model is ",model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    best_loss = float("inf")
    num_body=body_tensor.shape[1]
    num_step=body_tensor.shape[0]
    #body_tensor=body_tensor.view(num_step,num_body*2)
    body_tensor=body_tensor.reshape(num_step,num_body*2)
    #normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    #convert the tensor to numpy array and do the normalization
    body_tensor=scaler.fit_transform(body_tensor.cpu().detach().numpy())
    time_step = dt
    X,Y=[],[]
    for i in range(training_size-step_delay):
        X.append(body_tensor[i:i+step_delay,:])
        Y.append(body_tensor[i+step_delay,:])
        #print("X is "+str(X),"y is "+str(y))
    X=np.array(X)
    Y=np.array(Y)
    print(X.shape,Y.shape)
    X=torch.tensor(X,dtype=torch.float32,device=device)
    Y=torch.tensor(Y,dtype=torch.float32,device=device)
    #randomly shuffle the data
    perm = torch.randperm(X.shape[0])
    X=X[perm]
    Y=Y[perm]
    for epoch in range(num_epochs):
        epoch_loss = 0
        print("epoch is "+str(epoch))
        for i in range(0,training_size - step_delay):
            optimizer.zero_grad()
            #print(epoch,i)
            #torch.autograd.set_detect_anomaly(True)
            #optimizer.zero_grad()
            #print("i is "+str(i))
            inputs = X[i, :, :].clone().to(device).requires_grad_(True)
            target = Y[i, :].clone().to(device)
            outputs = model(inputs)
            loss1 = criterion(outputs, target)
            epoch_loss += loss1
            loss1.backward()
            optimizer.step()
            #epoch_loss =epoch_loss+ loss1.item()#+loss2.item()
        if epoch_loss.item() < best_loss:
            best_loss = epoch_loss.item()
            print("best loss is "+str(best_loss))
            #Save the best model for the test case under the test case folder with the numerical methods, time step and epoch number
            if not os.path.exists("models/"+test_case):
                os.makedirs("models/"+test_case)
            #torch.save(model, 'models/'+test_case+'/LSTM_best_model'+'dt'+str(dt)+'epoch'+str(epoch)+'.pth')
            #torch.save(model, 'models/'+test_case+'/best_model'+numerical_methods+'.pth')
        scheduler.step()
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / training_size:.10f}')
    return scaler,model  # Return the trained model
def Infer_LSTM(test_case,body_tensor,scaler,model,step_delay=1,num_steps=1000,dt=0.01,device='cuda'):
    #This function is to use the trained LSTM model to predict the trajectory
    #model is the trained LSTM model
    #step_delay is the step delay we use for training
    #num_steps is the number of steps we want to predict
    #dt is the time step we use for the training data
    #body_tensor is the initial condition we want to use for prediction,it's a tensor with shape (step_delay,num_bodys,2)
    #reshape the body_tensor to (step_delay,num_bodys*2)
    num_body=body_tensor.shape[1]
    #body_tensor=body_tensor.view(step_delay,num_body*2)
    body_tensor=body_tensor.reshape(step_delay,num_body*2)
    #convert to numpy and normalize the data
    body_tensor=scaler.transform(body_tensor.cpu().detach().numpy())
    #convert to tensor
    body_tensor=torch.tensor(body_tensor,dtype=torch.float32,device=device)
    print("body_tensor shape is "+str(body_tensor.shape))
    print("model is ",model)
    print("start testing the LSTM model")
    #the logic of the LSTM model is to predict the next step based on the previous step,
    #so we need to use the previous step to predict the next step
    #we use the body_tensor which contains the first step_delay steps to predict the next one step
    #Then we use the second step to the second step_delay+1 steps to predict the next one step
    #until we predict the num_steps steps
    #we use the body_tensor_pred to store the predicted trajectory
    #Start prediction iteratively:
    body_tensor_pred=torch.zeros(num_steps+step_delay,num_body*2).to(device)
    body_tensor_pred[0:step_delay,:]=body_tensor
    for i in range(num_steps):
        inputs = body_tensor_pred[i:i+step_delay, :]
        #print("inputs shape is "+str(inputs.shape))
        outputs = model(inputs)
        #print("outputs shape is "+str(outputs.shape))
        #print("outputs is "+str(outputs))
        body_tensor_pred[i+step_delay,:]=outputs
    #reshape the body_tensor to (num_steps,num_bodys,2)
    #inverse the normalization
    body_tensor_pred=scaler.inverse_transform(body_tensor_pred.cpu().detach().numpy())
    #convert to tensor
    body_tensor_pred=torch.tensor(body_tensor_pred,dtype=torch.float32,device=device)
    body_tensor_pred = body_tensor_pred.view(num_steps + step_delay, num_body, 2)
    return body_tensor_pred

#This is the training function for the FCNN model, the second baseline model
def train_FCNN(test_case,model,aug_t_tensor,aug_body_tensor,training_size,num_epochs=450,dt=0.01,device='cuda'):
    #This function is used to train the FCNN model
    #model is the FCNN model we want to train
    #aug_body_tensor is the training data we want to use with shape (num_data,num_bodys,2)
    #training_size is the number of data we want to use for training, training_size<=num_data
    #num_epochs is the number of epochs we want to use for training
    #dt is the time step we use for the training data
    print("start training the FCNN model")
    print("aug_body_tensor shape is "+str(aug_body_tensor.shape))
    print("model is ",model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    best_loss = float("inf")
    num_body=aug_body_tensor.shape[1]
    num_step=aug_body_tensor.shape[0]
    for epoch in range(num_epochs):
        epoch_loss = 0
        print("epoch is "+str(epoch))
        for i in range(0,training_size):
            optimizer.zero_grad()
            #print(epoch,i)
            #torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            #print("i is "+str(i))
            inputs = aug_t_tensor[i].clone().to(device).requires_grad_(True)
            target = aug_body_tensor[i, :, :].clone().to(device)
            #reshape the target to (num_bodys*2)
            target=target.view(num_body*2)
            outputs = model(inputs)
            loss1 = criterion(outputs, target)
            epoch_loss += loss1
            loss1.backward()
            optimizer.step()
            #epoch_loss =epoch_loss+ loss1.item()#+loss2.item()
        if epoch_loss.item() < best_loss:
            best_loss = epoch_loss.item()
            print("best loss is "+str(best_loss))
            #Save the best model for the test case under the test case folder with the numerical methods, time step and epoch number
            if not os.path.exists("models/"+test_case):
                os.makedirs("models/"+test_case)
            torch.save(model, 'models/'+test_case+'/FCNN_best_model'+'dt'+str(dt)+'epoch'+str(epoch)+'.pth')
            #torch.save(model, 'models/'+test_case+'/best_model'+numerical_methods+'.pth')
        scheduler.step()
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / training_size:.10f}')
    return model  # Return the trained model

def test_FCNN(test_case,model,aug_t_tensor,num_body,num_step,dt=0.01,device='cuda'):
    #This function is to use the trained FCNN model to predict the trajectory
    #model is the trained FCNN model
    #num_step is the number of steps we want to predict
    #dt is the time step we use for the training data
    #aug_t_tensor is the time tensor we want to use for prediction,it's a tensor with shape (num_step)
    #Start prediction iteratively:
    aug_body_tensor_pred=torch.zeros(num_step,num_body*2).to(device)
    for i in range(num_step):
        inputs = aug_t_tensor[i, :]
        #print("inputs shape is "+str(inputs.shape))
        outputs = model(inputs)
        #print("outputs shape is "+str(outputs.shape))
        #print("outputs is "+str(outputs))
        aug_body_tensor_pred[i,:]=outputs
    #reshape the body_tensor to (num_steps,num_bodys,2)
    aug_body_tensor_pred = aug_body_tensor_pred.view(num_step, num_body, 2)
    return aug_body_tensor_pred

def train_trajectory_PNODE_symplectic(test_case,numerical_methods,model, body_tensor, training_size,step_delay=2, num_epochs=450,dt=0.01,device='cuda'):
    #This function is used to train the PNODE model for the symplectic integrator, with energy conservation
    #test_case is a string that indicates which test case we are using, the case needs to be energy conservative, else please use train_PNODE
    #numerical_methods is a string that indicates which numerical methods we want to use
    #model is the PNODE model we want to train
    #body_tensor is the training data we want to use with shape (num_data,num_bodys,2), the first dimension is the time dimension,
    #the second dimension is the body dimension, the third dimension is the generalized coordinate and momentum dimension
    #training_size is the number of data we want to use for training, training_size<=num_data
    #num_epochs is the number of epochs we want to use for training
    #dt is the time step we use for the training data
    print("start training the PNODE_symplectic model")
    print("body_tensor shape is "+str(body_tensor.shape))
    num_bodys=body_tensor.shape[1]
    print("model is ",model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_loss = float("inf")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(training_size - 1):
            torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            #print("i is "+str(i))
            inputs = body_tensor[i, :, :].clone().to(device).requires_grad_(True)
            target = body_tensor[i + 1, :, :].clone().to(device)
            #inputs = torch.tensor(body_tensor[i, :, :], dtype=torch.float32, requires_grad=True).to(device)
            #acceleration_pred = model(inputs)
            #print("acceleration is",acceleration_pred)
            #target = torch.tensor(body_tensor[i + 1, :, :], dtype=torch.float32, requires_grad=True).to(device)
            #print("acceleration_pred shape is "+str(acceleration_pred.shape),"target shape is "+str(target.shape),"inputs shape is "+str(inputs.shape))
            #choose the numerical methods to integrate the system
            if numerical_methods=="sep_sv":
                Integrated_pred = sep_stormer_verlet_multiple_body(inputs,neural_network_force_function_PNODE_symplectic_dH_dq,neural_network_force_function_PNODE_symplectic_dH_dp,2,dt,if_final_state=True,model=model)
            elif numerical_methods=="yoshida4":
                Integrated_pred = yoshida4_multiple_body(inputs,neural_network_force_function_PNODE_symplectic_dH_dq,neural_network_force_function_PNODE_symplectic_dH_dp,2,dt,if_final_state=True,model=model)
            elif numerical_methods=="fukushima6":
                Integrated_pred = fukushima6_multiple_body(inputs,neural_network_force_function_PNODE_symplectic_dH_dq,neural_network_force_function_PNODE_symplectic_dH_dp,2,dt,if_final_state=True,model=model)
            #raise an error if the numerical methods is not specified correctly
            else:
                raise ValueError("The numerical methods is not specified correctly")
            #we extract the generalized coordinate and generalized momentum from the data, and calculate the loss separately
            loss = criterion(Integrated_pred, target)
            loss.backward()
            """
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(name, param.grad.norm())
                else:
                    print(name, "No gradient")
            """
            optimizer.step()
            epoch_loss += loss.item()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("best loss is "+str(best_loss))
            #Save the best model for the test case under the test case folder with the numerical methods, time step and epoch number
            if not os.path.exists("models/"+test_case):
                os.makedirs("models/"+test_case)
            torch.save(model, 'models/'+test_case+'/best_model'+numerical_methods+'dt'+str(dt)+'epoch'+str(epoch)+'.pth')
            #torch.save(model, 'models/'+test_case+'/best_model'+numerical_methods+'.pth')
        scheduler.step()
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss): {epoch_loss / training_size:.10f}')
    return model  # Return the trained model
#This is the function to test the trained model:
#Input: model is the trained model we want to test
#       body is the initial condition we want to use for testing
#       num_steps is the number of steps we want to use for testing
#       dt is the time step we want to use for testing
#Output: The testing result, with shape (num_steps,num_bodys,2)
def test_PNODE(numerical_methods,model,body,num_steps,dt,device='cuda'):
    print("start testing the PNODE model")
    print("body shape is "+str(body.shape))
    print("model is ",model)
    print("numerical_methods is "+numerical_methods)
    #reshape the body_tensor to (num_data,num_bodys*2)
    num_body=body.shape[0]
    body_tensor = torch.tensor(body, dtype=torch.float32, device=device)
    if numerical_methods=="fe":
        testing_result = forward_euler_multiple_body(body_tensor,neural_network_force_function_PNODE,num_steps+1,dt,if_final_state=False,model=model)
    if numerical_methods=="rk4":
        testing_result = runge_kutta_four_multiple_body(body_tensor,neural_network_force_function_PNODE,num_steps+1,dt,if_final_state=False,model=model)
    if numerical_methods=="midpoint":
        testing_result = midpoint_method_multiple_body(body_tensor,neural_network_force_function_PNODE,num_steps+1,dt,if_final_state=False,model=model)
    #If the numerical methods is not specified correctly, raise an error
    #if numerical_methods!="fe" and numerical_methods!="rk4":
    #raise ValueError("The numerical methods is not specified correctly")
    return testing_result
def test_PNODE_symplectic(numerical_methods,model,body,num_steps,dt,device='cuda'):
    print("start testing the PNODE_symplectic model")
    print("body shape is "+str(body.shape))
    print("model is ",model)
    print("numerical_methods is "+numerical_methods)
    body_tensor = torch.tensor(body, dtype=torch.float32, device=device)
    if numerical_methods=="sep_sv":
        testing_result = sep_stormer_verlet_multiple_body(body_tensor,neural_network_force_function_PNODE_symplectic_dH_dq,neural_network_force_function_PNODE_symplectic_dH_dp,num_steps,dt,if_final_state=False,model=model)
    elif numerical_methods=="yoshida4":
        testing_result = yoshida4_multiple_body(body_tensor,neural_network_force_function_PNODE_symplectic_dH_dq,neural_network_force_function_PNODE_symplectic_dH_dp,num_steps,dt,if_final_state=False,model=model)
    elif numerical_methods=="fukushima6":
        testing_result = fukushima6_multiple_body(body_tensor,neural_network_force_function_PNODE_symplectic_dH_dq,neural_network_force_function_PNODE_symplectic_dH_dp,num_steps,dt,if_final_state=False,model=model)
    else:
        raise ValueError("The numerical methods is not specified correctly")
    return testing_result


def train_trajectory_PNODE_symplectic2(test_case, numerical_methods, model, body_tensor, training_size,step_delay=2,
                                      num_epochs=450, dt=0.01, device='cuda', verbose=True):

    def log(message):
        if verbose:
            print(message)
    log("Starting training of the PNODE_symplectic model")
    log(f"body_tensor shape: {body_tensor.shape}")
    log(f"Model: {model}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    best_loss = float("inf")
    # Numerical methods mapping
    methods = {
        "sep_sv": sep_stormer_verlet_multiple_body,
        "yoshida4": yoshida4_multiple_body,
        "fukushima6": fukushima6_multiple_body
    }
    if numerical_methods not in methods:
        raise ValueError("The numerical method is not specified correctly")
    integration_method = methods[numerical_methods]
    # Create directory if it doesn't exist
    model_dir = f"models/{test_case}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        log(f"Epoch: {epoch}")
        for i in range(training_size - step_delay+1):
            optimizer.zero_grad()
            inputs = body_tensor[i, :, :].clone().to(device).requires_grad_(True)
            #target = body_tensor[i + step_delay-1, :, :].clone().to(device)
            target = body_tensor[i:i +step_delay, :, :].clone().to(device)
            #Integrated_pred = integration_method(inputs, neural_network_force_function_PNODE_symplectic_dH_dq,neural_network_force_function_PNODE_symplectic_dH_dp, step_delay, dt,if_final_state=True, model=model)
            Integrated_pred = integration_method(inputs, neural_network_force_function_PNODE_symplectic_dH_dq,
                                                 neural_network_force_function_PNODE_symplectic_dH_dp, step_delay, dt,
                                                 if_final_state=False, model=model)
            loss1 = criterion(Integrated_pred, target)
            #loss2 =criterion((Integrated_pred - inputs) / dt,(target - inputs) / dt)
            loss=loss1#+loss2
            #log(f"Loss1: {loss1.item()}")
            #log(f"Loss2: {loss2.item()}")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / training_size)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            log(f"Best loss: {best_loss}")
            model_path = f'{model_dir}/best_model_{numerical_methods}_dt{dt}_epoch{epoch}.pth'
            torch.save(model.state_dict(), model_path)
        scheduler.step()
        log(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / training_size:.10f}')
    return model


def train_sampling_PNODE(test_case,numerical_methods,model, training_pair_input,training_pair_output, training_size, num_epochs=450,dt=0.01,device='cuda'):
    #This function is used to train the PNODE model
    #model is the PNODE model we want to train
    #body_tensor is the training data we want to use with shape (num_data,num_bodys,2)
    #training_size is the number of data we want to use for training, training_size<=num_data
    #num_epochs is the number of epochs we want to use for training
    #dt is the time step we use for the training data
    print("start training the sampling PNODE model")
    print("body_tensor shape is "+str(training_pair_input.shape))
    print("model is ",model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    best_loss = float("inf")
    time_step = dt
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(training_size -1):
            #torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            #print("i is "+str(i))
            inputs = training_pair_input[i, :, :].clone().to(device).requires_grad_(True)
            target = training_pair_output[i, :, :].clone().to(device)
            if numerical_methods=="fe":
                Integrated_pred = forward_euler_multiple_body(inputs,neural_network_force_function_PNODE,2,dt,if_final_state=True,model=model)
            if numerical_methods=="rk4":
                Integrated_pred = runge_kutta_four_multiple_body(inputs,neural_network_force_function_PNODE,2,dt,if_final_state=True,model=model)
            #If the numerical methods is not specified correctly, raise an error
            if numerical_methods!="fe" and numerical_methods!="rk4":
                raise ValueError("The numerical methods is not specified correctly")
            #extract the displacement and velocity from the data, and calculate the loss separately
            loss1 = criterion((Integrated_pred-inputs)/dt, (target-inputs)/dt)+criterion(Integrated_pred, target)
            #loss2= criterion(Integrated_pred[1,1], target[1,1])#*(target[:,0].max()-target[:,0].min())/(target[:,1].max()-target[:,1].min())
            #print("loss1 is "+str(loss1),"loss2 is "+str(loss2))
            #print("target1 is",target[:,0],"target2 is",target[:,1])
            #loss = criterion(Integrated_pred, target)/dt/dt
            #loss1.backward(retain_graph=True)
            epoch_loss += loss1.item()
            loss1.backward()
            #loss.backward()
            #check_gradient()
        #epoch_loss.backward()
            optimizer.step()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("best loss is "+str(best_loss))
            #Save the best model for the test case under the test case folder with the numerical methods, time step and epoch number
            if not os.path.exists("models/"+test_case):
                os.makedirs("models/"+test_case)
            torch.save(model, 'models/'+test_case+'/best_sampling_model'+numerical_methods+'dt'+str(dt)+'epoch'+str(epoch)+'.pth')
            #torch.save(model, 'models/'+test_case+'/best_model'+numerical_methods+'.pth')
        scheduler.step()
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / training_size:.10f}')
    return model  # Return the trained model
def train_PNODE_batch(model, body_tensor, training_size, num_epochs=450, batch_size=32, dt=0.01, device='cuda'):
    print("start training the PNODE model")
    print("body_tensor shape is "+str(body_tensor.shape))
    print("model is ",model)
    # Create a DataLoader for mini-batch training
    train_dataset = TensorDataset(body_tensor[:training_size-1], body_tensor[1:training_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_loss = float("inf")
    time_step = dt
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs_batch, targets_batch in train_loader:
            torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            inputs_batch = inputs_batch.clone().to(device).requires_grad_(True)
            targets_batch = targets_batch.clone().to(device)
            Integrated_preds = forward_euler_multiple_body(inputs_batch, neural_network_force_function_PNODE, 2, dt, if_final_state=True, model=model)
            loss = criterion(Integrated_preds, targets_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs_batch.size(0)  # Multiply by batch size as loss.item() gives mean loss per input
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model, '1_body_best_model.pth')
        scheduler.step()
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / training_size:.10f}')

    return model
class HNN(nn.Module):
    def __init__(self, input_dim, differentiable_model):
        super(HNN, self).__init__()
        self.input_dim = input_dim
        self.differentiable_model = differentiable_model
        self.M = self.permutation_tensor(input_dim)
    def forward(self, x):
        y = self.differentiable_model(x)
        return y

    def time_derivative(self, x, t=None):
        y = self.forward(x) # traditional forward pass
        J_dy = torch.zeros_like(x) # start out with both components set to 0
        dy = torch.autograd.grad(y.sum(), x, create_graph=True)[0] # gradients
        M = self.permutation_tensor(self.input_dim)
        J_dy = dy @ M

        return J_dy

    def permutation_tensor(self,n):

        M = torch.eye(n)
        M = torch.cat([-M[n//2:], M[:n//2]])

        return M
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim
        super().__init__()
        N = 256
        self.model = nn.Sequential(
          nn.Linear(self.input_dim, N),
          nn.Sigmoid(),
          nn.Linear(N, N),
          nn.Tanh(),
          nn.Linear(N, self.output_dim)
        )
    def forward(self, x):
        '''Forward pass'''
        output = self.model(x)
        return output
def mean_absolute_relative_error(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))
#The second model is the baseline model, the Hamiltonian Neural Network
#The model is a Pytorch module, with the following parameters:
#       num_bodys is the number of bodys we want to simulate
def hnn_integrate(model, x0, t):
    #using solve to integrate the hamiltonian
    x = torch.tensor(x0, requires_grad=True, dtype=torch.float32)
    xs = []
    for t0 in t:
        xs.append(x.detach().numpy())
        x = rk4_step(model.time_derivative, x, t0, h=0.01)  # 0.01 is the dt
    return np.stack(xs)
class LNN(nn.Module):
    def __init__(self, num_bodys=1, layers=3, width=256, d_interest=0,activation='softmax', initializer='xavier'):
        super(LNN, self).__init__()
        self.act = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax()
        }[activation]

        # Construct the neural network layers
        self.d_interest = d_interest
        self.layers = layers
        self.num_bodys = num_bodys
        self.dim = 2 * num_bodys + self.d_interest
        self.width = width

        self.fc1 = nn.Linear(2*self.num_bodys, self.width)
        self.fc2 = nn.Linear(self.width, self.width)
        self.fc3 = nn.Linear(self.width, self.num_bodys)
    def lagrangian(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x
    def forward(self, x):
        #print("x shape is "+str(x.shape))
        n = x.shape[1] // 2
        xv = torch.autograd.Variable(x, requires_grad=True)
        xv_tup = tuple([xi for xi in x])
        tqt = xv[ :,n:]
        jacpar = partial(jacobian, self.lagrangian, create_graph=True)
        hesspar = partial(hessian, self.lagrangian, create_graph=True)
        A = tuple(map(hesspar, xv_tup))
        B = tuple(map(jacpar, xv_tup))
        multi = lambda Ai, Bi, tqti, n: torch.inverse(Ai[n:, n:]) @ (Bi[:n, 0] - Ai[n:, :n] @ tqti)
        multi_par = partial(multi, n=n)
        tqtt_tup = tuple(map(multi_par, A, B, tqt))
        tqtt = torch.cat([tqtti[None] for tqtti in tqtt_tup])
        xt = torch.cat([tqt, tqtt], axis=1)
        xt.retain_grad()
        return xt
    def t_forward(self, t, x):
        return self.forward(x)
def lnn_solve_ode(model, x0, t):
    x0 = x0.clone().cpu().detach().numpy()

    def f(x, t):
        x_tor = torch.tensor(np.expand_dims(x, 0), requires_grad=True).float().to(device)
        return np.squeeze(model(x_tor).clone().cpu().detach().numpy(), axis=0)

    return odeint(f, x0, t)