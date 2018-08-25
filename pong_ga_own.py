#!/usr/bin/env python3
import sys
import gym
import ptan
#import roboschool
import collections
import copy
import time
import numpy as np
import argparse
import logging

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter


POPULATION_SIZE = 600
PARENTS_COUNT = 10
WORKERS_COUNT = 6
# POPULATION_SIZE = 4
# PARENTS_COUNT = 2
# WORKERS_COUNT = 2


NOISE_STD = 0.01
SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
MAX_SEED = 2**32 - 1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('debug.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


def evaluate(env, net, device="cpu"):
    obs = env.reset()
    reward = 0.0
    steps = 0
    net_model = Net(env.observation_space.shape, env.action_space.n).to(device)
    net_model.load_state_dict(net)

    #print("in evalue,net_model", net_model)
    while True:
        obs_v = torch.FloatTensor([np.array(obs, copy=False)]).to(device)
        act_prob = net_model(obs_v).to(device)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env.step(acts.data.cpu().numpy()[0])
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


def mutate_net(parent, child_seed, copy_net=True):
    new_net = copy.deepcopy(parent) if copy_net else net
    np.random.seed(child_seed)
    #for p in net:#.parameters():
    for key, value in new_net.items():
        #logger.debug("current_process: %s,p[]:%s", mp.current_process(), parents[parent])
        #print("key,value,value.data", key, value, type(value))
        noise_t = torch.from_numpy(np.random.normal(size=value.data.size()).astype(np.float32))
        #temp = NOISE_STD*noise_t
        value.data = NOISE_STD*noise_t

        # if(value.data.is_cuda):
        #      temp = temp.cuda()
        # value.data += temp
    return new_net


def build_net(env, seeds, device="cpu"):
    torch.manual_seed(seeds)
    net = Net(env.observation_space.shape, env.action_space.n)
    return net


def worker_func(parents, output_queue,  device="cpu"):
    env = make_env()
    #logger.debug("current_process: %s,parents[0]:%s", mp.current_process(), parents[0])
    guide = False
    temp = parents[0]

    while True:
        child = []
        #logger.debug("len of parents:%s", len(parents))
        logger.debug("current_process: %s,parents[0][0]:%s", mp.current_process(), parents[0]['fc.2.bias'])

        # if guide:
        #     assert temp == parents[0]
        #     guide = not guide

        for _ in range(SEEDS_PER_WORKER):
            parent = np.random.randint(PARENTS_COUNT)
            child_seed = np.random.randint(MAX_SEED)
            child_net = mutate_net(parents[parent], child_seed)#.to(device)
            reward, steps = evaluate(env, child_net, device)
            child.append((child_net, reward, steps))
            #logger.debug("current_process: %s,parents:%s", mp.current_process(), parents)

        child.sort(key=lambda p: p[1], reverse=True)

        for j in range(PARENTS_COUNT):
            output_queue.put(child[j])


if __name__ == "__main__":
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment="-pong-ga")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = "cuda:1" if args.cuda else "cpu"

    env = make_env()
    manager = mp.Manager()
    parents = manager.list()

    #create PARENTS_COUNT parents
    for i in range(PARENTS_COUNT):
        seed = np.random.randint(MAX_SEED)
        net = build_net(env, seed).to(device)
        net.share_memory()
        parents.append(net.state_dict())
        #parents[seed] = net

    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    time_start = time.time()


    for _ in range(WORKERS_COUNT):
        w = mp.Process(target=worker_func, args=(parents, output_queue, device))
        w.start()

    gen_idx = 0
    while True:
        t_start = time.time()
        batch_steps = 0
        children = []

        logger.debug("before, current_process: %s,parents[0][0]:%s", mp.current_process(), parents[0]['fc.2.bias'])

        while len(children) < PARENTS_COUNT * WORKERS_COUNT:
            out_item = output_queue.get()
            children.append(out_item)
            batch_steps += out_item[2]

        children.sort(key=lambda p: p[1], reverse=True)

        rewards = [p[1] for p in children[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        writer.add_scalar("batch_steps", batch_steps, gen_idx)
        writer.add_scalar("gen_seconds", time.time() - t_start, gen_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, gen_idx)
        total_time = (time.time() - time_start) / 60
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s, total_running_time=%.2f/m" % (
            gen_idx, reward_mean, reward_max, reward_std, speed, total_time))

        for i in range(PARENTS_COUNT):
            parents[i] = children[i][0]
        #logger.debug("current_process: %s,parents[0]:%s", mp.current_process(), parents[0])
        logger.debug("after, current_process: %s,parents[0][0]:%s", mp.current_process(), parents[0]['fc.2.bias'])


        gen_idx += 1
