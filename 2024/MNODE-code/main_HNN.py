import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as np
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
from torch.utils.data import TensorDataset, DataLoader
from Model.model import *
from Model.utils import *
from Model.force_fun import *
from Model.Data_generator import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
solve_ivp = scipy.integrate.solve_ivp

config = {
    "test_case": "Single_Mass_Spring_Symplectic",
    "generation_numerical_methods": "analytical",
    "numerical_methods": "rk4",
    "dt": 0.01,
    "num_steps": 3000,
    "training_size": 300,
    "num_epochs": 30000,
    "num_steps_test": 3000,
    "dt_test": 0.01,
    "step_delay": 10,
    "random_seed": 0,
    "if_symplectic": True,
    "model_string": "HNN",
}
def L2_loss(u, v):
  return (u-v).pow(2).mean()
dyn_sys = 'linear_spring_mass'

numSamples = 1
total_steps = config["num_epochs"]
nn_model = MLP(2, 1)
model = HNN(2, differentiable_model=nn_model)
optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# arrange data
data = get_dataset(seed=5, samples=numSamples, test_split=0.1)
x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
print("x shape:",x.shape)
test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
dxdt = torch.Tensor(data['dx'])
test_dxdt = torch.Tensor(data['test_dx'])
t = torch.Tensor(data['t'])
test_t = torch.tensor(data['test_t'])
# vanilla train loop
stats = {'train_loss': [], 'test_loss': []}
for step in range(total_steps + 1):
    # train step
    dxdt_hat = model.time_derivative(x)
    loss = L2_loss(dxdt, dxdt_hat)
    loss.backward();
    optim.step();
    optim.zero_grad()
    # run test data
    test_dxdt_hat = model.time_derivative(test_x)
    test_loss = L2_loss(test_dxdt, test_dxdt_hat)
    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())

train_dxdt_hat = model.time_derivative(x)
train_dist = (dxdt - train_dxdt_hat) ** 2
test_dxdt_hat = model.time_derivative(test_x)
test_dist = (test_dxdt - test_dxdt_hat) ** 2
print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
      .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
              test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))


#we want to integrate the hamiltonian to get the trajectory

#now we want to plot the trajectory
# do a numerical integration of the hamiltonian
t_span = [0, 30]
t_eval = np.linspace(t_span[0], t_span[1], 3000)
x0 = np.array([1,0])
xs = hnn_integrate(model, x0, t_eval)
save_data_np(xs,config["test_case"],config["model_string"],config["training_size"],config["num_steps_test"],config["dt_test"])
# plot the trajectory and compare with the ground truth
plt.figure()
plt.plot(x.detach().numpy()[:, 0], x.detach().numpy()[:, 1], label='Ground truth')
plt.plot(xs[:, 0], xs[:, 1], label='HNN')
plt.legend()
plt.show()


plt.figure()
plt.plot(train_dxdt_hat.detach().numpy()[:int(len(dxdt[:, 0])), 0],train_dxdt_hat.detach().numpy()[:int(len(dxdt[:, 0])), 1])
plt.xlabel('p dot')
plt.ylabel('q dot')
plt.grid()
plt.title('')
plt.show()

plt.figure()
plt.plot(dxdt[:int(len(dxdt[:, 0])), 0], '-b', linewidth=1)
plt.plot(train_dxdt_hat.detach().numpy()[:int(len(dxdt[:, 0])), 0], '-.r', linewidth=1)
plt.xlabel('index of q dot')
plt.ylabel('q dot')
plt.legend(['Real', 'Predicted'])
plt.grid()

plt.figure()
plt.plot(dxdt[:int(len(dxdt[:, 0])), 1], '-b', linewidth=1)
plt.plot(train_dxdt_hat.detach().numpy()[:int(len(dxdt[:, 0])), 1], '-.r', linewidth=1)
plt.xlabel('index of p dot')
plt.ylabel('p dot')
plt.legend(['Real', 'Predicted'])
plt.grid()


plt.figure()
plt.plot(test_dxdt[:,0],'-b',linewidth = 1)
plt.plot(test_dxdt_hat.detach().numpy()[:,0],'-.r',linewidth = 1)
plt.xlabel('index of q dot')
plt.ylabel('q dot')
plt.legend(['Real','Predicted'])
plt.grid()

plt.figure()
plt.plot(test_dxdt[:,1],'-b',linewidth = 1)
plt.plot(test_dxdt_hat.detach().numpy()[:,1],'-.r',linewidth = 1)
plt.xlabel('index of p dot')
plt.ylabel('p dot')
plt.legend(['Real','Predicted'])
plt.grid()
plt.figure()

plt.plot(model.forward(x).detach().numpy())
plt.xlabel('index of data')
plt.ylabel('Hamiltonian value')
plt.legend(['Real', 'Predicted'])
plt.legend(['Real', 'Predicted'])
plt.grid()
plt.title('Hamiltonian')
plt.show()

plt.figure()
plt.plot(x.detach().numpy()[:,0],x.detach().numpy()[:,1],'.r')
plt.xlabel('q')
plt.ylabel('p')
plt.grid()
plt.show()