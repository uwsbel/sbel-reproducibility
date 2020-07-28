import torch
import torch.nn as nn
from torch.distributions import Normal
import copy

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def outputSize(in_size, kernel_size, stride, padding):
    conv_size = copy.deepcopy(in_size)
    for i in range(len(kernel_size)):
        conv_size[0] = int((conv_size[0] - kernel_size[i] + 2*(padding[i])) / stride[i]) + 1
        conv_size[1] = int((conv_size[1] - kernel_size[i] + 2*(padding[i])) / stride[i]) + 1

    return(conv_size)

class MultiSensorEarlyFusion(nn.Module):
    def __init__(self, input1_shape, input2_shape, num_outputs, std=-0.5):
        super(MultiSensorEarlyFusion, self).__init__()

        self.input1_shape = input1_shape
        self.input2_shape = input2_shape
        self.num_outputs = num_outputs

        fc_size = outputSize(input1_shape, [8, 4, 3], [4, 2, 1], [0, 0, 0])
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(fc_size[0] * fc_size[1] * 64, num_outputs * 5),

        )
        self.actor_fc0 = nn.Linear(input2_shape, num_outputs * 5)
        self.actor_fc1 = nn.Linear(num_outputs * 10, num_outputs * 20)
        self.actor_fc2 = nn.Linear(num_outputs * 20, num_outputs * 10)
        self.actor_fc3 = nn.Linear(num_outputs * 10, num_outputs * 5)
        self.actor_fc4 = nn.Linear(num_outputs * 5, num_outputs)

        self.critic_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(fc_size[0] * fc_size[1] * 64, num_outputs * 5),

        )
        self.critic_fc0 = nn.Linear(input2_shape, num_outputs * 5)
        self.critic_fc1 = nn.Linear(num_outputs * 10, num_outputs * 20)
        self.critic_fc2 = nn.Linear(num_outputs * 20, num_outputs * 10)
        self.critic_fc3 = nn.Linear(num_outputs * 10, num_outputs * 5)
        self.critic_fc4 = nn.Linear(num_outputs * 5, 1)


        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.apply(init_weights)


    def forward(self, data):
        x0 = ((data[0]-127)/255.).permute(0, 3, 1, 2)
        x1 = self.actor_cnn(x0)
        x2 = nn.functional.relu(self.actor_fc0(data[1]))
        x = torch.cat((x1, x2), dim=1)
        x = nn.functional.relu(self.actor_fc1(x))
        x = nn.functional.relu(self.actor_fc2(x))
        x = nn.functional.relu(self.actor_fc3(x))
        mu = torch.tanh(self.actor_fc4(x))
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        y1 = self.critic_cnn(x0)
        y2 = nn.functional.relu(self.critic_fc0(data[1]))
        y = torch.cat((y1, y2), dim=1)
        y = nn.functional.relu(self.critic_fc1(y))
        y = nn.functional.relu(self.critic_fc2(y))
        y = nn.functional.relu(self.critic_fc3(y))
        value = self.critic_fc4(y)

        return dist, value
