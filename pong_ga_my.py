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


# POPULATION_SIZE = 600
# PARENTS_COUNT = 10
# WORKERS_COUNT = 6
POPULATION_SIZE = 4
PARENTS_COUNT = 2
WORKERS_COUNT = 2


NOISE_STD = 0.01
SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
MAX_SEED = 2**32 - 1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
fh = logging.FileHandler('debug.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
ch.setFormatter(formatter)
logger.addHandler(ch)


class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
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


def evaluate(env, net, device="cpu"):
    obs = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([np.array(obs, copy=False)]).to(device)
        act_prob = net(obs_v)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env.step(acts.data.cpu().numpy()[0])
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


def mutate_net(net, seed, device="cpu", copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_t = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32))
        temp = NOISE_STD*noise_t
        #p.data += NOISE_STD * noise_t
        #print(temp.is_cuda,p.data.is_cuda)
        if(p.data.is_cuda):
            temp = temp.cuda()
        p.data += temp
    return new_net


def build_net(env, seeds, device="cpu"):
    torch.manual_seed(seeds[0])
    net = Net(env.observation_space.shape, env.action_space.n)
    for seed in seeds[1:]:
        assert False
        net = mutate_net(net, seed, device, copy_net=False)#.to(device)
    return net


OutputItem = collections.namedtuple('OutputItem', field_names=['seeds', 'net', 'reward', 'steps'])


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


def worker_func(input_queue, output_queue, top_parent_cache, device="cpu"):
    env = make_env()

    while True:
        # parents:(parent_seed,net,child_seed)
        parents = input_queue.get()
        population = []

        if parents is None:
            break

        #logger.debug("current_process: %s,parents:%s", mp.current_process(), parents)
        #logger.debug("current_process: %s,top_parent_cache:%s", mp.current_process(), top_parent_cache)

        for net_seeds in parents:
            if len(net_seeds) > 1:
                #logger.debug("current_process: %s,net_seeds[:-1]:%s,top_parent_cache: %s", mp.current_process(),
                             #net_seeds[0], top_parent_cache)
                logger.debug("current_process:inside1,%s", mp.current_process())
                print("come here1")
                net = net_seeds[1]
                if net is not None:
                    net = mutate_net(net, net_seeds[-1], device).to(device)
                else:
                    assert False
                    #net = build_net(env, net_seeds, device).to(device)
            else:
                net = build_net(env, net_seeds, device).to(device)
                logger.debug("current_process:inside2,%s", mp.current_process())
                print("come here2")


            reward, steps = evaluate(env, net, device)
            population.append((net, net_seeds, reward, steps))

        #logger.debug("before, current_process: %s,seeds:%s", mp.current_process(), population)
        population.sort(key=lambda p: p[2], reverse=True)
        #logger.debug("output queue put, current_process: %s,population:%s", mp.current_process(), population[:][1])

        for i in range(PARENTS_COUNT):
            #top_parent_cache[population[i][1][-1]] = population[i][0].state_dict()
            output_queue.put(OutputItem(seeds=population[i][1], net=population[i][0], reward=population[i][2],
                                        steps=population[i][3]))
            logger.debug("current_process:inside3,%s", mp.current_process())
            print("come here3")

        #logger.debug("after output queue put, current_process: %s,population:%s", mp.current_process(), population)

#top_parent_cache={}


if __name__ == "__main__":
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment="-pong-ga")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = "cuda:1" if args.cuda else "cpu"

    manager = mp.Manager()
    top_parent_cache = manager.dict()

    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    time_start = time.time()
    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue, top_parent_cache, device))
        w.start()
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put(seeds)

    gen_idx = 0
    elite = None
    while True:
        t_start = time.time()
        batch_steps = 0
        population = []
        while len(population) < PARENTS_COUNT * WORKERS_COUNT:
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.net, out_item.reward))
            batch_steps += out_item.steps
        if elite is not None:
            population.append(elite)

        #top_parent_cache = {}
        #logger.debug("before current_process: %s,top_parent_cache:%s", mp.current_process(), top_parent_cache)
        #logger.debug("current_process: %s,seeds:%s", mp.current_process(), population)
        population.sort(key=lambda p: p[2], reverse=True)
        #logger.debug("current_process: %s,seeds:%s", mp.current_process(), population)

        # for i in range(PARENTS_COUNT):
        #     top_parent_cache[population[i][1][-1]] = population[i][0]

        #logger.debug("after current_process: %s,top_parent_cache:%s", mp.current_process(), top_parent_cache)

        rewards = [p[2] for p in population[:PARENTS_COUNT]]
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
        total_time = (time.time() - time_start)/60
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s, total_running_time=%.2f/m" % (
            gen_idx, reward_mean, reward_max, reward_std, speed, total_time))

        elite = population[0]
        #print(mp.current_process(), "population:", population)
        for worker_queue in input_queues:
            seeds = []
            for _ in range(SEEDS_PER_WORKER):
                parent = np.random.randint(PARENTS_COUNT)
                next_seed = np.random.randint(MAX_SEED)
                seeds.append(tuple([population[parent][0][-1], population[parent][1], next_seed]))
            worker_queue.put(seeds)
        gen_idx += 1

        #time.sleep(1)
        #top_parent_cache = {}

