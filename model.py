import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):

    def __init__(self, num_actions, num_atoms):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.fc_critic = nn.Linear(32 * 3 * 3, 256)
        self.fc_actor = nn.Linear(32 * 3 * 3, 256)

        self.num_atoms = num_atoms
        self.num_outputs = num_actions

        self.critic_linear = nn.Linear(256, self.num_atoms)
        self.actor_linear = nn.Linear(256, self.num_outputs)

        self.apply(weights_init)

        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 0.01)
        self.critic_linear.bias.data.fill_(0)

        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.train()

        self.gamma = 0.99
        self.N = num_atoms
        self.Vmax = int((self.N - 1) / 2)
        self.Vmin = -self.Vmax
        self.dz = 1.
        self.z = np.linspace(self.Vmin, self.Vmax, self.N) #, dtype=np.int32)


    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)

        x_critic = F.relu(self.fc_critic(x))
        x_actor = F.relu(self.fc_actor(x))

        return self.critic_linear(x_critic), self.actor_linear(x_actor)


    def get_loss(self, r, probs):
        # r         | numpy array (bsize)
        # probs     | pytorch Variable (bsize+1 x 51)
        # return    | loss ()
        bsize = r.shape[0]
        r_rep = np.tile(np.expand_dims(r, 1), (1, self.N))
        Tz = np.clip(r_rep + self.gamma * self.z, self.Vmin, self.Vmax)
        # Tz | bsize x N
        b = (Tz - self.Vmin) / self.dz # b[batch, i] belongs [0, N-1]
        l, u = np.floor(b).astype(np.int32), np.ceil(b).astype(np.int32)

        m = np.zeros((bsize, self.N), dtype=np.float32)
        probs_np = probs.data.numpy()

        for i in range(self.N):
            m[range(bsize), l[:, i]] += probs_np[1:, i] * (u[:, i] - b[:, i])
            m[range(bsize), u[:, i]] += probs_np[1:, i] * (b[:, i] - l[:, i])

        loss = - torch.sum(Variable(torch.from_numpy(m)) * torch.log(probs[:-1]))
        return loss


    def get_loss_propogate(self, r, probs):
        # r         | numpy array (seq)
        # probs     | pytorch Variable (seq+1 x 51)
        # return    | loss ()
        seq_len = r.shape[0]
        R = self.z # 51
        Tz = np.zeros((seq_len, self.N))
        
        for i in reversed(range(seq_len)):
            R = R * self.gamma + r[i]
            Tz[i] = np.clip(R, self.Vmin, Vmax)

        b = (Tz - self.Vmin) / self.dz
        l, u = np.floor(b).astype(np.int32), np.ceil(b).astype(np.int32)

        m = np.zeros((seq_len, self.N), dtype=np.float32)
        probs_np = probs.data.numpy()[-1] # last probabilities, shape=(51)
        
        for i in range(self.N):
            m[range(seq_len), l[:, i]] += probs_np[i] * (u[:, i] - b[:, i])
            m[range(seq_len), u[:, i]] += probs_np[i] * (b[:, i] - l[:, i])

        loss = - torch.sum(Variable(torch.from_numpy(m)) * torch.log(probs[:-1]))
        return loss


    def get_v(self, probs, batch=True):
        # probs     | pytorch Variable (bsize x 51)
        # return    | numpy array (bsize) or scalar
        V = self.z * probs.data.numpy()
        V = np.sum(V, axis=1) # bsize

        if not batch:
            return np.squeeze(V)
        return V
