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
import time
from collections import deque

import numpy as np 
import matplotlib.pyplot as plt 

def is_dead(info):
    dead = False
    if is_dead.current_life > info['ale.lives']:
        dead = True
    is_dead.current_life = info['ale.lives']
    return dead

is_dead.current_life = 0

def test(rank, args, shared_model):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    if not os.path.exists('model-a3c-aux'):
        os.makedirs('model-a3c-aux')
    path = 'model-a3c-aux/model-{}.pth'.format(args.model_name)
    print('saving directory is', path)

    model = ActorCritic(env.action_space.n, args.num_atoms, args.gamma)  
    model.eval()

    state = env.reset()
    state = np.concatenate([state] * 4, axis=0)
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    action_stat = [0] * model.num_outputs

    start_time = time.time()
    episode_length = 0

    for ep_counter in itertools.count(1):
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            
            torch.save(shared_model.state_dict(), path)
            print('saved model')

        atoms_logit, logit = model(Variable(state.unsqueeze(0), volatile=True))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()

        action_np = action[0, 0]
        action_stat[action_np] += 1

        state_new, reward, done, info = env.step(action_np)
        dead = is_dead(info)
        
        if args.testing: 
            atoms_prob = F.softmax(atoms_logit)
            value = model.get_v(atoms_prob, batch=False)
            atoms_prob = atoms_prob.squeeze().data.numpy()

            print('episode', episode_length, 'normal action', action_np, 'lives', info['ale.lives'], 'value', value)
            env.render()

            if ep_counter % 100 == 0:
                plt.plot(model.z, atoms_prob)
                plt.title('average v is {}'.format(value))
                plt.show()
        state = np.append(state.numpy()[1:,:,:], state_new, axis=0)
        done = done or episode_length >= args.max_episode_length

        reward_sum += reward
        episode_length += 1

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            print("actions stats real {}".format(action_stat[:model.num_outputs]))

            reward_sum = 0
            episode_length = 0
            state = env.reset()
            env.seed(args.seed + rank + (args.num_processes+1)*ep_counter)
            state = np.concatenate([state] * 4, axis=0)
            action_stat = [0] * model.num_outputs
            if not args.testing: time.sleep(60)

        state = torch.from_numpy(state)
