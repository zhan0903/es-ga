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


NOISE_STD = 0.01
POPULATION_SIZE = 4#2000
PARENTS_COUNT = 2
WORKERS_COUNT = 2
SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
MAX_SEED = 2**32 - 1


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


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


def mutate_net(net, seed, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    i = 0
    #logger.debug("current_process, %s", mp.current_process())
    #logger.debug("current_process, %s, new_new's weight:%s\n", mp.current_process(), new_net.conv[2].weight)
    for p in new_net.parameters():
        #logger.debug("p.data's size, type %s, %s", p.size(), type(p.data))
        noise_t = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
        i += 1
    #logger.debug("current_process, %s, sum of p:%s", mp.current_process(), i)
    return new_net


def build_net(env, seeds):
    torch.manual_seed(seeds[0])
    net = Net(env.observation_space.shape, env.action_space.n)
    logger.debug("in build_net,current_process: %s,seeds:%s",mp.current_process(),seeds)
    for seed in seeds[1:]:
        print(mp.current_process(), seed)
        net = mutate_net(net, seed, copy_net=False)
    return net


OutputItem = collections.namedtuple('OutputItem', field_names=['seeds', 'reward', 'steps'])


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


def worker_func(input_queue, output_queue, device="cpu"):
    env = make_env()#gym.make("RoboschoolHalfCheetah-v1")
    cache = {}

    while True:
        parents = input_queue.get()
        print(mp.current_process(), parents)
        if parents is None:
            break
        new_cache = {}
        for net_seeds in parents:
            if len(net_seeds) > 1:
                net = cache.get(net_seeds[:-1])
                if net is not None:
                    net = mutate_net(net, net_seeds[-1])
                else:
                    net = build_net(env, net_seeds).to(device)
            else:
                net = build_net(env, net_seeds).to(device)
            # store{(seed,):net}
            new_cache[net_seeds] = net
            reward, steps = evaluate(env, net, device)
            output_queue.put(OutputItem(seeds=net_seeds, reward=reward, steps=steps))
        cache = new_cache
        #print("len of cache", len(cache))
        #print(cache)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment="-pong-ga")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"


    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue, device))
        w.start()
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put(seeds)

    gen_idx = 0
    elite = None
    while True:
        t_start = time.time()
        batch_steps = 0
        population = []
        while len(population) < SEEDS_PER_WORKER * WORKERS_COUNT:
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.reward))
            batch_steps += out_item.steps
        if elite is not None:
            population.append(elite)
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
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
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s" % (
            gen_idx, reward_mean, reward_max, reward_std, speed))

        elite = population[0]
        print(mp.current_process(), "population:", population)
        for worker_queue in input_queues:
            seeds = []
            for i in range(SEEDS_PER_WORKER):
                parent = np.random.randint(PARENTS_COUNT)
                next_seed = np.random.randint(MAX_SEED)
                #print("population[parent][0],next_seed", population[parent][0],next_seed)
                seeds.append(tuple(list(population[parent][0]) + [next_seed]))
                #print(i, mp.current_process(), "seeds:", seeds)
            worker_queue.put(seeds)
            #print(mp.current_process(), seeds)
        gen_idx += 1
    pass
