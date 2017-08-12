import math
import os
import sys
import itertools

import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms

import matplotlib.pyplot as plt 
import numpy as np 

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def is_dead(info):
    dead = False
    if is_dead.current_life > info['ale.lives']:
        dead = True
    is_dead.current_life = info['ale.lives']
    return dead

is_dead.current_life = 0

def train(rank, args, shared_model, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.action_space.n, args.num_atoms)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = np.concatenate([state] * 4, axis=0)
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    for ep_counter in itertools.count(1):
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        
        values = []
        log_probs = []
        atoms_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            atoms_logit, logit = model(Variable( state.unsqueeze(0) ))
            atoms_prob = F.softmax(atoms_logit)
            value = model.get_v(atoms_prob, batch=False)
            atoms_probs.append(atoms_prob)

            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            action_np = action.numpy()[0][0]
            state_new, reward, done, info = env.step(action_np)
            dead = is_dead(info)
            state = np.append(state.numpy()[1:,:,:], state_new, axis=0)
            done = done or episode_length >= args.max_episode_length
            
            reward = max(min(reward, 1), -1)
            episode_length += 1

            if done:
                episode_length = 0
                state = env.reset()
                env.seed(args.seed + rank + (args.num_processes+1)*ep_counter)
                state = np.concatenate([state] * 4, axis=0)
            elif dead:
                state = np.concatenate([state_new] * 4, axis=0)

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done or dead:
                break

        value_last = 0
        atoms_prob_last = Variable(torch.zeros(1, model.N))
        atoms_prob_last[0, int( (model.N-1)/2 )] = 1
        if not done and not dead:
            atoms_logit, _ = model(Variable( state.unsqueeze(0) ))
            atoms_prob_last = F.softmax(atoms_logit)
            value_last = model.get_v(atoms_prob_last, batch=False)

        R = value_last
        values.append(value_last)
        atoms_probs.append(atoms_prob_last)
        policy_loss = 0
        for i in reversed(range(len(rewards))):
            # todo, change to values[1:] * gamma + rewards - values[:-1]
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]

            policy_loss = policy_loss - \
                log_probs[i] * advantage - 0.01 * entropies[i]

        policy_loss = policy_loss.squeeze()
        value_loss = model.get_loss_propogate(np.array(rewards), torch.cat(atoms_probs))        
        
        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
