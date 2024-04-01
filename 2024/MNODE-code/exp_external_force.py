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
def generate_training_example():
    a1, b1 = 0, 1.0  # Range for x1_0
    a2, b2 = 0, 2  # Range for v1_0 (if you want a single value, set a and b to that value)
    a3, b3 = -1, 2  # Range for x2_0
    a4, b4 = -3, 0  # Range for v2_0
    a5, b5 = -1, 3  # Range for x3_0
    a6, b6 = -6, 1  # Range for v3_0

    # Generating samples from the uniform distribution
    x1_0 = a1 + (b1 - a1) * torch.rand(1)
    v1_0 = a2 + (b2 - a2) * torch.rand(1)
    x2_0 = a3 + (b3 - a3) * torch.rand(1)
    v2_0 = a4 + (b4 - a4) * torch.rand(1)
    x3_0 = a5 + (b5 - a5) * torch.rand(1)
    v3_0 = a6 + (b6 - a6) * torch.rand(1)
    x1_target, v1_target, x2_target, v2_target, x3_target, v3_target = runge_kutta4(x1_0, v1_0, x2_0, v2_0, x3_0, v3_0, actual_force_function, 2, time_step)

    inputs = torch.tensor([x1_0, v1_0, x2_0, v2_0, x3_0, v3_0], dtype=torch.float32, requires_grad=True).to(device)
    targets = torch.tensor(
        [x1_target[1], v1_target[1], x2_target[1], v2_target[1], x3_target[1], v3_target[1]], dtype=torch.float32
    ).to(device)

    return inputs, targets


#generate 1000 training examples for training


model = Net().to(device)
x1_actual, v1_actual, x2_actual, v2_actual,x3_actual,v3_actual = runge_kutta4(
    x1_00, v1_00, x2_00, v2_00,x3_00,v3_00, actual_force_function, num_steps, time_step
)

criterion = nn.MSELoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

best_loss=float("inf")
start_time = time.time()
for epoch in range(num_epochs):
    epoch_loss = 0
    for _ in range(training_size):
        inputs, targets = generate_training_example()
        optimizer.zero_grad()
        outputs = model(inputs)
        predicted_targets=  torch.stack(rk4_torch(inputs[0],inputs[1], inputs[2], inputs[3],inputs[4],inputs[5],neural_network_force_function, 2, time_step)).to(device)
        loss = criterion(predicted_targets, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model, '2_body_best_model.pth')

    end_time = time.time()
    epoch_time = end_time - start_time
    scheduler.step()

    if (epoch + 1) % 1 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / training_size:.8f}, Training Speed: {epoch_time:.2f} seconds"
        )
        start_time = time.time()

x1_predicted, v1_predicted, x2_predicted, v2_predicted, x3_predicted, v3_predicted = rk4_torch2(x1_00, v1_00, x2_00, v2_00,x3_00,v3_00,neural_network_force_function, num_steps, time_step)

# Plot the actual and predicted trajectories
plt.figure()
plt.plot(x1_actual.detach().cpu().numpy(), label='Actual Position of object 1')
plt.plot(x1_predicted.detach().cpu().numpy(), "r--", label='Predicted Position of object 1')
plt.plot(x2_actual.detach().cpu().numpy(), label='Actual Position of object 2')
plt.plot(x2_predicted.detach().cpu().numpy(), "g--", label='Predicted Position of object 2')
plt.plot(x3_actual.detach().cpu().numpy(), label='Actual Position of object 3')
plt.plot(x3_predicted.detach().cpu().numpy(), "y--", label='Predicted Position of object 3')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()
plt.title('Actual vs. Predicted Position')
plt.savefig('3bactual_vs_predicted_position.png')
plt.figure()
plt.plot(v1_actual.detach().cpu().numpy(), label='Actual Velocity of object 1')
plt.plot(v1_predicted.detach().cpu().numpy(), "r--", label='Predicted Velocity of object 1')
plt.plot(v2_actual.detach().cpu().numpy(), label='Actual Velocity of object 2')
plt.plot(v2_predicted.detach().cpu().numpy(), "g--", label='Predicted Velocity of object 2')
plt.plot(v3_actual.detach().cpu().numpy(), label='Actual Velocity of object 3')
plt.plot(v3_predicted.detach().cpu().numpy(), "y--", label='Predicted Velocity of object 3')
plt.xlabel('Time step')
plt.ylabel('Velocity')
plt.legend()
plt.title('Actual vs. Predicted Velocity')
plt.savefig('3bactual_vs_predicted_velocity.png')
# Calculate the force using the neural network
nn_force1 = np.zeros(num_steps)
nn_force2 = np.zeros(num_steps)
nn_force3 = np.zeros(num_steps)
for i in range(num_steps):
    state=torch.stack([x1_actual[i],v1_actual[i],x2_actual[i],v2_actual[i],x3_actual[i],v3_actual[i]])
    nn_force1[i] = neural_network_force_function(state,model)[0].detach().cpu().numpy()
    nn_force2[i] = neural_network_force_function(state,model)[1].detach().cpu().numpy()
    nn_force3[i] = neural_network_force_function(state, model)[2].detach().cpu().numpy()
# Generate the actual forces
actual_force1 = np.zeros(num_steps)
actual_force2 = np.zeros(num_steps)
actual_force3 = np.zeros(num_steps)
for i in range(num_steps):
    actual_force1[i] = actual_force_function(x1_actual[i], v1_actual[i], x2_actual[i], v2_actual[i], x3_actual[i], v3_actual[i], i*time_step)[0]
    actual_force2[i] = actual_force_function(x1_actual[i], v1_actual[i], x2_actual[i], v2_actual[i], x3_actual[i], v3_actual[i], i*time_step)[1]
    actual_force3[i] = actual_force_function(x1_actual[i], v1_actual[i], x2_actual[i], v2_actual[i], x3_actual[i], v3_actual[i], i * time_step)[2]

plt.figure()
plt.plot(actual_force1, label='Actual force of object 1')
plt.plot(nn_force1, label='Predicted force of object 1')
plt.plot(actual_force2, label='Actual force of object 2')
plt.plot(nn_force2, label='Predicted force of object 2')
plt.plot(actual_force3, label='Actual force of object 3')
plt.plot(nn_force3, label='Predicted force of object 3')
plt.xlabel('Time step')
plt.ylabel('Acceleration')
plt.legend()
plt.title('Actual vs. Predicted Acceleration')
plt.savefig('3bactual_vs_predicted_acceleration.png')
plt.show()