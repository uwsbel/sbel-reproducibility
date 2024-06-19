import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import partial
from torch.autograd.functional import jacobian, hessian
from torchdiffeq import odeint as tor_odeint
from torchdiffeq import odeint_adjoint as tor_odeintadj
print(torch.version.__version__)

def lagrangian(x, g=10, k=10):
    # defining the lagrangian function
    # x is the input vector of the form [q, qt]
    # q is array of genralise coords, [r, theta]
    # qt is array of genralise coords, [rdt, thetadt]
    # r is length of pendulum
    # theta is angle of pendulum
    # rdt is time derivative of r
    # thetadt is time derivative of theta
    # T is kinetic energy
    # V is potential energy
    q, qt = torch.split(x, 2)
    # T is the kinetic energy and it's calculated as 1/2*m*v^2
    T = 0.5*(qt[0]**2 + (q[0]*qt[1])**2)
    #here qt[0] is rdt so the first term is 1/2*m*rdt^2, the second term is 1/2*m*r^2*thetadt^2
    #here the velocity is divided into two components, one is the velocity of the pendulum bob and the other is the velocity of the pendulum arm
    V = g*q[0]*(1-torch.cos(q[1])) + k*(q[0] - 1)**2
    return T - V
def E(x, g=10, k=10):
    try:
        q, qt = torch.split(x, 2)
        cos = torch.cos
    except:
        q, qt = np.split(x, 2)
        cos = np.cos
    T = 0.5*(qt[0]**2 + (q[0]*qt[1])**2)
    V = g*q[0]*(1-cos(q[1])) + k*(q[0] - 1)**2
    return T + V
def rk4_step(f, x, t, h):
  # one step of runge-kutta integration
  k1 = h * f(x, t)
  k2 = h * f(x + k1/2, t + h/2)
  k3 = h * f(x + k2/2, t + h/2)
  k4 = h * f(x + k3, t + h)
  return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
def get_qdtt(q, qt, g=10, k=10):
    '''
    q is array of genralise coords, [r, theta]
    qt is array of genralise coords, [rdt, thetadt]

    returns time derivative of q.
    '''
    qdtt = np.zeros_like(q)
    qdtt[:, 0] = q[:, 0] * qt[:, 1] ** 2 - g * (1 - np.cos(q[:, 1])) - 2 * k * (q[:, 0] - 1)
    qdtt[:, 1] = (-g * np.sin(q[:, 1]) - 2 * qt[:, 0] * qt[:, 1]) / q[:, 0]
    return qdtt
def get_xt_anal(x, t):
    d = np.zeros_like(x)
    d[:, :2] = x[:, 2:]
    d[:, 2:] = get_qdtt(x[:, :2], x[:, 2:])
    # print(x, d)
    return d
def anal_solve_ode(q0, qt0, t, ):
    # solves the ode using the analytical solution
    x0 = np.append(q0, qt0)
    def f_anal(x, t):
        d = np.zeros_like(x)
        d[:2] = x[2:]
        d[2:] = np.squeeze(get_qdtt(np.expand_dims(x[:2], axis=0), np.expand_dims(x[2:], axis=0)))
        # print(x, d)
        return d
    return odeint(f_anal, x0, t, rtol=1e-10, atol=1e-10)
class LNN(nn.Module):
    def __init__(self):
        super(LNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    def lagrangian(self, x):
        x = F.softmax(self.fc1(x))
        x = F.softmax(self.fc2(x))
        x = self.fc3(x)
        return x
    def forward(self, x):
        #print("x shape is "+str(x.shape))
        n = x.shape[1] // 2
        xv = torch.autograd.Variable(x, requires_grad=True)
        xv_tup = tuple([xi for xi in x])
        tqt = xv[:, n:]
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
def loss(pred, targ):
    return torch.mean((pred - targ) ** 2)
def nn_solve_ode(model, x0, t):
# solves the ode using the neural network
    x0 = x0.detach().numpy()
    def f(x, t):
        x_tor = torch.tensor(np.expand_dims(x, 0), requires_grad=True).float()
        return np.squeeze(model(x_tor).detach().numpy(), axis=0)
    return odeint(f, x0, t)


def get_xt(lagrangian, t, x):
    # this function uses the lagrangian to calculate the time derivative of x
    # x is the input vector of the form [q, qt]
    # q is array of genralise coords, [r, theta]
    # qt is array of genralise coords, [rdt, thetadt]

    n = x.shape[0] // 2

    xv = torch.autograd.Variable(x, requires_grad=True)
    tq, tqt = torch.split(xv, 2, dim=0)
    A = torch.inverse(hessian(lagrangian, xv, create_graph=True)[n:, n:])
    B = jacobian(lagrangian, xv, create_graph=True)[:n]
    C = hessian(lagrangian, xv, create_graph=True)[n:, :n]
    tqtt = A @ (B - C @ tqt)
    xt = torch.cat([tqt, torch.squeeze(tqtt)])
    return xt


def torch_solve_ode(x0, t, lagrangian):
    # solves the ode using the lagrangian with the torchdiffeq package
    f = partial(get_xt, lagrangian)
    return tor_odeint(f, x0, t)
def q2xy(ql):
    '''
    Polar coords to xy
    '''
    try:
        xy = np.zeros_like(ql)
        sin = np.sin
        cos = np.cos
    except:
        xy = torch.zeros_like(ql)
        sin = torch.sin
        cos = torch.cos

    xy[:, 0] = ql[:, 0] * sin(ql[:, 1])
    xy[:, 1] = -ql[:, 0] * cos(ql[:, 1])
    return xy
# first we need to define the initial conditions
# q0 is the initial position of the pendulum
# qt0 is the initial velocity of the pendulum
t = np.arange(0, 5, 0.005)

q0 = np.array([1.1, 0.5])
q0p = np.array([1.1 + 1e-3, 0.5])
qt0 = np.array([0.0, 0.0])
#two paths with slightly different initial conditions
path = anal_solve_ode(q0, qt0, t)
ppath = anal_solve_ode(q0p, qt0, t)
xy = q2xy(path)
pxy = q2xy(ppath)

plt.figure()
plt.title('Pendulum path with perturbation')
plt.plot(xy[:, 0], xy[:, 1], label='path')
plt.plot(pxy[:, 0], pxy[:, 1], label='path perturbed')
plt.legend()



tx0 = torch.cat([torch.tensor(q0), torch.tensor(qt0)])
tt = torch.tensor(t)
tpath = torch_solve_ode(tx0, tt, lagrangian)
txy = q2xy(tpath).detach().numpy()
#compare the two paths by the lagrangian with pytorch and the analytical solution
plt.figure()
plt.plot(xy[:, 0], xy[:, 1], label='analytical')
plt.plot(txy[:, 0], txy[:, 1], label='pytorch')
plt.legend()
plt.title('Pendulum path with pytorch and analytical solution')


plt.figure()
En = [E(x) for x in path]
Ent = [E(x).detach().numpy() for x in tpath]
#compare the energy of the two paths
plt.plot(t, En, label='analytical')
plt.plot(t, Ent,    label='pytorch')
plt.legend()
plt.title('Energy of the pendulum, analytical vs pytorch')


#compare the difference between the two paths
plt.figure()
plt.plot(t, np.sum((tpath.detach().numpy() - path)**2, axis=1))
plt.title('Difference between analytical and pytorch')
plt.show()

#now we want to train the neural network to solve the ode
#we need to define the loss function
#now define the training and test range of t
N = 300
t_train = torch.tensor(np.linspace(0, 3, N)).float()
t_test = torch.tensor(np.linspace(3, 6, N)).float()

tstep = t_train[1].item()

#the training data is the analytical solution, the original position and velocity with the time derivative of the position and velocity
x_train = torch.tensor(anal_solve_ode(q0, qt0, t_train)).float()
xt_train = torch.tensor(get_xt_anal(x_train, t_train)).float()

#the test data is the analytical solution, the original position and velocity with the time derivative of the position and velocity
y_train = torch.tensor(rk4_step(get_xt_anal, x_train, t_train, tstep)).float()
x_test = torch.tensor(anal_solve_ode(q0, qt0, t_test)).float()
xt_test = torch.tensor(get_xt_anal(x_test, t_test)).float()
y_test = torch.tensor(rk4_step(get_xt_anal, x_test, t_test, tstep)).float()
eps = 100

batch_size = 100
model = LNN()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_list = []
for e in range(eps):
    running_loss = 0.
    for i in range(1, N // batch_size):
        optimizer.zero_grad()
        xi = xt_train[(i - 1) * batch_size:i * batch_size]
        xt_pred = model(xi)
        loss_val = loss(xt_pred, xt_train[i])
        loss_val.backward()
        optimizer.step()
        running_loss += loss_val.item()
    print('[%d, %5d] loss: %.10f' %
          (e + 1, i + 1, running_loss / N))
    loss_list.append(running_loss / N)
    running_loss = 0.0

nn_test = nn_solve_ode(model, x_train[0], t_train)
plt.figure()
plt.plot(t_train, x_train[:, 0], label='analytical_train')
plt.plot(t_train, nn_test[:, 0], label='nn_train')
plt.legend()
plt.title('Training data')

#plt.figure()
#plt.plot(t_test, x_test[:, 0], label='analytical_test')
#nn_test2 = nn_solve_ode(model, x_test[0], t_test)
#plt.plot(t_test, nn_test2[:, 0], label='nn_test')
#plt.title('Test data')
#plt.legend()

plt.show()
for i in range(N):
    print((nn_test[i,0], x_train[i,0].clone().cpu().detach().numpy()))