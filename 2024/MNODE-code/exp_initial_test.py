import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.cuda.amp import autocast, GradScaler
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
start_time = time.time()
torch.manual_seed(42)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mass1 = 100
mass2=10
mass3=1
num_steps = 100
time_step = 0.01
time_end = num_steps * time_step
x1_00 = 1.0
v1_00 = 0.0
x2_00 = 2.0
v2_00 = 0.0
x3_00=3.0
v3_00=0.0

# Train the neural network
num_epochs = 200
training_size=1000
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(64, 3)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        #x = torch.relu(self.fc3(x))
        x = self.fc3(x)
        return x
def actual_force_function(x1, v1, x2, v2, x3, v3, t):
    spring_force_3 = -50 * (x3 - x2) - 2 * (v3 - v2)
    spring_force_2 = -50 * (x2 - x1) - 2 * (v2 - v1) - spring_force_3
    spring_force_1 = -spring_force_2 - 50 * x1 - 2 * v1
    acceleration_1 = spring_force_1 / mass1
    acceleration_2 = spring_force_2 / mass2
    acceleration_3 = spring_force_3 / mass3
    return acceleration_1, acceleration_2, acceleration_3
def neural_network_force_function(state, model):
    inputs = state.to(device)
    f_pred = model(inputs)
    acceleration_pred_1, acceleration_pred_2,acceleration_pred_3 = f_pred[0] / mass1, f_pred[1] / mass2,f_pred[2]/mass3
    return acceleration_pred_1, acceleration_pred_2,acceleration_pred_3
def runge_kutta4(x1_0, v1_0, x2_0, v2_0, x3_0, v3_0, force_function, num_steps, time_step):
    x1 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    v1 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    x2 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    v2 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    x3 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    v3 = torch.zeros(num_steps, dtype=torch.float32, device=device)

    x1[0] = x1_0
    v1[0] = v1_0
    x2[0] = x2_0
    v2[0] = v2_0
    x3[0] = x3_0
    v3[0] = v3_0

    for i in range(num_steps - 1):
        k1_x1 = time_step * v1[i]
        k1_v1 = time_step * force_function(x1[i], v1[i], x2[i], v2[i], x3[i], v3[i], i * time_step)[0]
        k1_x2 = time_step * v2[i]
        k1_v2 = time_step * force_function(x1[i], v1[i], x2[i], v2[i], x3[i], v3[i], i * time_step)[1]
        k1_x3 = time_step * v3[i]
        k1_v3 = time_step * force_function(x1[i], v1[i], x2[i], v2[i], x3[i], v3[i], i * time_step)[2]

        k2_x1 = time_step * (v1[i] + 0.5 * k1_v1)
        k2_v1 = time_step * force_function(x1[i] + 0.5 * k1_x1, v1[i] + 0.5 * k1_v1, x2[i] + 0.5 * k1_x2, v2[i] + 0.5 * k1_v2, x3[i] + 0.5 * k1_x3, v3[i] + 0.5 * k1_v3, (i + 0.5) * time_step)[0]
        k2_x2 = time_step * (v2[i] + 0.5 * k1_v2)
        k2_v2 = time_step * force_function(x1[i] + 0.5 * k1_x1, v1[i] + 0.5 * k1_v1, x2[i] + 0.5 * k1_x2, v2[i] + 0.5 * k1_v2, x3[i] + 0.5 * k1_x3, v3[i] + 0.5 * k1_v3, (i + 0.5) * time_step)[1]
        k2_x3 = time_step * (v3[i] + 0.5 * k1_v3)
        k2_v3 = time_step * force_function(x1[i] + 0.5 * k1_x1, v1[i] + 0.5 * k1_v1, x2[i] + 0.5 * k1_x2, v2[i] + 0.5 * k1_v2, x3[i] + 0.5 * k1_x3, v3[i] + 0.5 * k1_v3, (i + 0.5) * time_step)[2]

        k3_x1 = time_step * (v1[i] + 0.5 * k2_v1)
        k3_v1 = time_step * force_function(x1[i] + 0.5 * k2_x1, v1[i] + 0.5 * k2_v1, x2[i] + 0.5 * k2_x2, v2[i] + 0.5 * k2_v2, x3[i] + 0.5 * k2_x3, v3[i] + 0.5 * k2_v3, (i + 0.5) * time_step)[0]
        k3_x2 = time_step * (v2[i] + 0.5 * k2_v2)
        k3_v2 = time_step * force_function(x1[i] + 0.5 * k2_x1, v1[i] + 0.5 * k2_v1, x2[i] + 0.5 * k2_x2, v2[i] + 0.5 * k2_v2, x3[i] + 0.5 * k2_x3, v3[i] + 0.5 * k2_v3, (i + 0.5) * time_step)[1]
        k3_x3 = time_step * (v3[i] + 0.5 * k2_v3)
        k3_v3 = time_step * force_function(x1[i] + 0.5 * k2_x1, v1[i] + 0.5 * k2_v1, x2[i] + 0.5 * k2_x2, v2[i] + 0.5 * k2_v2, x3[i] + 0.5 * k2_x3, v3[i] + 0.5 * k2_v3, (i + 0.5) * time_step)[2]

        k4_x1 = time_step * (v1[i] + k3_v1)
        k4_v1 = time_step * force_function(x1[i] + k3_x1, v1[i] + k3_v1, x2[i] + k3_x2, v2[i] + k3_v2, x3[i] + k3_x3, v3[i] + k3_v3, (i + 1) * time_step)[0]
        k4_x2 = time_step * (v2[i] + k3_v2)
        k4_v2 = time_step * force_function(x1[i] + k3_x1, v1[i] + k3_v1, x2[i] + k3_x2, v2[i] + k3_v2, x3[i] + k3_x3, v3[i] + k3_v3, (i + 1) * time_step)[1]
        k4_x3 = time_step * (v3[i] + k3_v3)
        k4_v3 = time_step * force_function(x1[i] + k3_x1, v1[i] + k3_v1, x2[i] + k3_x2, v2[i] + k3_v2, x3[i] + k3_x3, v3[i] + k3_v3, (i + 1) * time_step)[2]

        x1[i + 1] = x1[i] + (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1) / 6
        v1[i + 1] = v1[i] + (k1_v1 + 2 * k2_v1 + 2 * k3_v1 + k4_v1) / 6
        x2[i + 1] = x2[i] + (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2) / 6
        v2[i + 1] = v2[i] + (k1_v2 + 2 * k2_v2 + 2 * k3_v2 + k4_v2) / 6
        x3[i + 1] = x3[i] + (k1_x3 + 2 * k2_x3 + 2 * k3_x3 + k4_x3) / 6
        v3[i + 1] = v3[i] + (k1_v3 + 2 * k2_v3 + 2 * k3_v3 + k4_v3) / 6

    return x1, v1, x2, v2, x3, v3
def rk4_torch(x1_0, v1_0, x2_0, v2_0, x3_0, v3_0, force_function, num_steps, time_step):
    x1 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    v1 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    x2 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    v2 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    x3 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    v3 = torch.zeros(num_steps, dtype=torch.float32, device=device)

    x1[0] = x1_0
    v1[0] = v1_0
    x2[0] = x2_0
    v2[0] = v2_0
    x3[0] = x3_0
    v3[0] = v3_0

    for i in range(num_steps - 1):
        state = torch.stack([x1[i], v1[i], x2[i], v2[i], x3[i], v3[i]]).detach().requires_grad_()
        force = force_function(state, model)
        k1_x1 = time_step * v1[i]
        k1_v1 = time_step * force[0]
        k1_x2 = time_step * v2[i]
        k1_v2 = time_step * force[1]
        k1_x3 = time_step * v3[i]
        k1_v3 = time_step * force[2]
        state = torch.stack([x1[i] + 0.5 * k1_x1, v1[i] + 0.5 * k1_v1, x2[i] + 0.5 * k1_x2,
                             v2[i] + 0.5 * k1_v2, x3[i] + 0.5 * k1_x3, v3[i] + 0.5 * k1_v3]).detach().requires_grad_()
        force = force_function(state, model)
        k2_x1 = time_step * (v1[i] + 0.5 * k1_v1)
        k2_v1 = time_step * force[0]
        k2_x2 = time_step * (v2[i] + 0.5 * k1_v2)
        k2_v2 = time_step * force[1]
        k2_x3 = time_step * (v3[i] + 0.5 * k1_v3)
        k2_v3 = time_step * force[2]
        state = torch.stack([x1[i] + 0.5 * k2_x1, v1[i] + 0.5 * k2_v1, x2[i] + 0.5 * k2_x2,
                             v2[i] + 0.5 * k2_v2, x3[i] + 0.5 * k2_x3, v3[i] + 0.5 * k2_v3]).detach().requires_grad_()
        force = force_function(state, model)
        k3_x1 = time_step * (v1[i] + 0.5 * k2_v1)
        k3_v1 = time_step * force[0]
        k3_x2 = time_step * (v2[i] + 0.5 * k2_v2)
        k3_v2 = time_step * force[1]
        k3_x3 = time_step * (v3[i] + 0.5 * k2_v3)
        k3_v3 = time_step * force[2]
        state = torch.stack([x1[i] + k3_x1, v1[i] + k3_v1, x2[i] + k3_x2, v2[i] + k3_v2,
                             x3[i] + k3_x3, v3[i] + k3_v3]).detach().requires_grad_()
        force = force_function(state, model)
        k4_x1 = time_step * (v1[i] + k3_v1)
        k4_v1 = time_step * force[0]
        k4_x2 = time_step * (v2[i] + k3_v2)
        k4_v2 = time_step * force[1]
        k4_x3 = time_step * (v3[i] + k3_v3)
        k4_v3 = time_step * force[2]

        x1[i + 1] = x1[i] + (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1) / 6
        v1[i + 1] = v1[i] + (k1_v1 + 2 * k2_v1 + 2 * k3_v1 + k4_v1) / 6
        x2[i + 1] = x2[i] + (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2) / 6
        v2[i + 1] = v2[i] + (k1_v2 + 2 * k2_v2 + 2 * k3_v2 + k4_v2) / 6
        x3[i + 1] = x3[i] + (k1_x3 + 2 * k2_x3 + 2 * k3_x3 + k4_x3) / 6
        v3[i + 1] = v3[i] + (k1_v3 + 2 * k2_v3 + 2 * k3_v3 + k4_v3) / 6

    return x1[num_steps-1], v1[num_steps-1], x2[num_steps-1], v2[num_steps-1], x3[num_steps-1], v3[num_steps-1]
def rk4_torch2(x1_0, v1_0, x2_0, v2_0, x3_0, v3_0, force_function, num_steps, time_step):
    x1 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    v1 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    x2 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    v2 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    x3 = torch.zeros(num_steps, dtype=torch.float32, device=device)
    v3 = torch.zeros(num_steps, dtype=torch.float32, device=device)

    x1[0] = x1_0
    v1[0] = v1_0
    x2[0] = x2_0
    v2[0] = v2_0
    x3[0] = x3_0
    v3[0] = v3_0

    for i in range(num_steps - 1):
        state = torch.stack([x1[i], v1[i], x2[i], v2[i], x3[i], v3[i]]).detach().requires_grad_()
        force = force_function(state, model)

        k1_x1 = time_step * v1[i]
        k1_v1 = time_step * force[0]
        k1_x2 = time_step * v2[i]
        k1_v2 = time_step * force[1]
        k1_x3 = time_step * v3[i]
        k1_v3 = time_step * force[2]

        state = torch.stack([x1[i] + 0.5 * k1_x1, v1[i] + 0.5 * k1_v1, x2[i] + 0.5 * k1_x2,
                             v2[i] + 0.5 * k1_v2, x3[i] + 0.5 * k1_x3, v3[i] + 0.5 * k1_v3]).detach().requires_grad_()
        force = force_function(state, model)

        k2_x1 = time_step * (v1[i] + 0.5 * k1_v1)
        k2_v1 = time_step * force[0]
        k2_x2 = time_step * (v2[i] + 0.5 * k1_v2)
        k2_v2 = time_step * force[1]
        k2_x3 = time_step * (v3[i] + 0.5 * k1_v3)
        k2_v3 = time_step * force[2]

        state = torch.stack([x1[i] + 0.5 * k2_x1, v1[i] + 0.5 * k2_v1, x2[i] + 0.5 * k2_x2,
                             v2[i] + 0.5 * k2_v2, x3[i] + 0.5 * k2_x3, v3[i] + 0.5 * k2_v3]).detach().requires_grad_()
        force = force_function(state, model)

        k3_x1 = time_step * (v1[i] + 0.5 * k2_v1)
        k3_v1 = time_step * force[0]
        k3_x2 = time_step * (v2[i] + 0.5 * k2_v2)
        k3_v2 = time_step * force[1]
        k3_x3 = time_step * (v3[i] + 0.5 * k2_v3)
        k3_v3 = time_step * force[2]

        state = torch.stack([x1[i] + k3_x1, v1[i] + k3_v1, x2[i] + k3_x2, v2[i] + k3_v2,
                             x3[i] + k3_x3, v3[i] + k3_v3]).detach().requires_grad_()
        force = force_function(state, model)

        k4_x1 = time_step * (v1[i] + k3_v1)
        k4_v1 = time_step * force[0]
        k4_x2 = time_step * (v2[i] + k3_v2)
        k4_v2 = time_step * force[1]
        k4_x3 = time_step * (v3[i] + k3_v3)
        k4_v3 = time_step * force[2]

        x1[i + 1] = x1[i] + (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1) / 6
        v1[i + 1] = v1[i] + (k1_v1 + 2 * k2_v1 + 2 * k3_v1 + k4_v1) / 6
        x2[i + 1] = x2[i] + (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2) / 6
        v2[i + 1] = v2[i] + (k1_v2 + 2 * k2_v2 + 2 * k3_v2 + k4_v2) / 6
        x3[i + 1] = x3[i] + (k1_x3 + 2 * k2_x3 + 2 * k3_x3 + k4_x3) / 6
        v3[i + 1] = v3[i] + (k1_v3 + 2 * k2_v3 + 2 * k3_v3 + k4_v3) / 6

    return x1, v1, x2, v2, x3, v3


#generate 1000 training examples for training

#load the well trained model
model = torch.load('2_body_best_model.pth')
model.eval()
#write a function to generate the actual trajectory and the predicted trajectory for given initial conditions
def generate_trajectory(x1_0, v1_0, x2_0, v2_0, x3_0, v3_0, force_function, num_steps, time_step):
    x1_actual, v1_actual, x2_actual, v2_actual, x3_actual, v3_actual = runge_kutta4(x1_0, v1_0, x2_0, v2_0, x3_0, v3_0, force_function, num_steps, time_step)
    x1_predicted, v1_predicted, x2_predicted, v2_predicted, x3_predicted, v3_predicted = rk4_torch2(x1_0, v1_0, x2_0, v2_0, x3_0, v3_0, neural_network_force_function, num_steps, time_step)
    #convert the tensors to numpy arrays
    x1_predicted=x1_predicted.detach().cpu().numpy()
    v1_predicted=v1_predicted.detach().cpu().numpy()
    x2_predicted=x2_predicted.detach().cpu().numpy()
    v2_predicted=v2_predicted.detach().cpu().numpy()
    x3_predicted=x3_predicted.detach().cpu().numpy()
    v3_predicted=v3_predicted.detach().cpu().numpy()
    x1_actual=x1_actual.detach().cpu().numpy()
    v1_actual=v1_actual.detach().cpu().numpy()
    x2_actual=x2_actual.detach().cpu().numpy()
    v2_actual=v2_actual.detach().cpu().numpy()
    x3_actual=x3_actual.detach().cpu().numpy()
    v3_actual=v3_actual.detach().cpu().numpy()

    #make them together to only return two arrays for actual and predicted
    ground_truth=np.array([x1_actual,v1_actual,x2_actual,v2_actual,x3_actual,v3_actual])
    predicted=np.array([x1_predicted,v1_predicted,x2_predicted,v2_predicted,x3_predicted,v3_predicted])
    return ground_truth,predicted
x1_00 = 1.0
v1_00 = 0.0
x2_00 = 2.0
v2_00 = 0.0
x3_00=3.0
v3_00=0.0
ground_truth1,predicted1=generate_trajectory(x1_00, v1_00, x2_00, v2_00, x3_00, v3_00, actual_force_function, num_steps, time_step)
x1_01=0.5
v1_01=0.0
x2_01=1.5
v2_01=0.0
x3_01=2.5
v3_01=0.0
ground_truth2,predicted2=generate_trajectory(x1_01, v1_01, x2_01, v2_01, x3_01, v3_01, actual_force_function, num_steps, time_step)
x1_02=0.0
v1_02=0.0
x2_02=1.0
v2_02=0.0
x3_02=2.0
v3_02=0.0
ground_truth3,predicted3=generate_trajectory(x1_02, v1_02, x2_02, v2_02, x3_02, v3_02, actual_force_function, num_steps, time_step)
x1_03=1.5
v1_03=0.0
x2_03=2.5
v2_03=0.0
x3_03=3.5
v3_03=0.0
ground_truth4,predicted4=generate_trajectory(x1_03, v1_03, x2_03, v2_03, x3_03, v3_03, actual_force_function, num_steps, time_step)
#visualize the trajectories
subplot_labels = ['(a)', '(b)', '(c)', '(d)']

# Vertical position of the label below the x-axis
vertical_position = -0.17  # Adjust this as needed
plt.subplots(2,2,figsize=(8,8))
plt.subplot(2,2,1)
#visualize the trajectories for the first initial conditions
plt.plot(ground_truth1[0,:],ground_truth1[1,:],label="Ground truth",color="blue")
plt.plot(ground_truth1[2,:],ground_truth1[3,:],color="blue")
plt.plot(ground_truth1[4,:],ground_truth1[5,:],color="blue")
plt.plot(predicted1[0,:],predicted1[1,:],label="MNODE",color="red",linestyle='--')
plt.plot(predicted1[2,:],predicted1[3,:],color="red",linestyle='--')
plt.plot(predicted1[4,:],predicted1[5,:],color="red",linestyle='--')
plt.grid()
plt.legend(frameon=False)
plt.xlabel("$x$")
plt.ylabel("$v$")
plt.text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
plt.subplot(2,2,2)
#visualize the trajectories for the second initial condition
plt.plot(ground_truth2[0,:],ground_truth2[1,:],label="Ground truth",color="blue")
plt.plot(ground_truth2[2,:],ground_truth2[3,:],color="blue")
plt.plot(ground_truth2[4,:],ground_truth2[5,:],color="blue")
plt.plot(predicted2[0,:],predicted2[1,:],label="MNODE",color="red",linestyle='--')
plt.plot(predicted2[2,:],predicted2[3,:],color="red",linestyle='--')
plt.plot(predicted2[4,:],predicted2[5,:],color="red",linestyle='--')
plt.grid()
plt.legend(frameon=False)
plt.xlabel("$x$")
plt.ylabel("$v$")
plt.text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
plt.subplot(2,2,3)
#visualize the trajectories for the third initial condition
plt.plot(ground_truth3[0,:],ground_truth3[1,:],label="Ground truth",color="blue")
plt.plot(ground_truth3[2,:],ground_truth3[3,:],color="blue")
plt.plot(ground_truth3[4,:],ground_truth3[5,:],color="blue")
plt.plot(predicted3[0,:],predicted3[1,:],label="MNODE",color="red",linestyle='--')
plt.plot(predicted3[2,:],predicted3[3,:],color="red",linestyle='--')
plt.plot(predicted3[4,:],predicted3[5,:],color="red",linestyle='--')
plt.grid()
plt.legend(frameon=False)
plt.xlabel("$x$")
plt.ylabel("$v$")
plt.text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
plt.subplot(2,2,4)
#visualize the trajectories for the fourth initial condition
plt.plot(ground_truth4[0,:],ground_truth4[1,:],label="Ground truth",color="blue")
plt.plot(ground_truth4[2,:],ground_truth4[3,:],color="blue")
plt.plot(ground_truth4[4,:],ground_truth4[5,:],color="blue")
plt.plot(predicted4[0,:],predicted4[1,:],label="MNODE",color="red",linestyle='--')
plt.plot(predicted4[2,:],predicted4[3,:],color="red",linestyle='--')
plt.plot(predicted4[4,:],predicted4[5,:],color="red",linestyle='--')
plt.grid()
plt.legend(frameon=False)
plt.xlabel("$x$")
plt.ylabel("$v$")
plt.text(0.5, vertical_position, subplot_labels[3], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig('3b_initial_test.png')
plt.show()