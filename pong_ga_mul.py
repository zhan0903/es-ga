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
logging.basicConfig(level=logging.DEBUG)
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


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


def evaluate(env, net, device="cpu"):
    obs = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([np.array(obs, copy=False)]).to(device)
        act_prob = net(obs_v).to(device)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env.step(acts.data.cpu().numpy()[0])
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


def mutate_net(env, net, seed, copy_net=True):
    new_net = Net(env.observation_space.shape, env.action_space.n)
    new_net.load_state_dict(net)
    #new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))#.cuda()
        p.data += NOISE_STD * noise_t
    return new_net


OutputItem = collections.namedtuple('OutputItem', field_names=['child_net', 'reward', 'steps'])


def build_net(env, seeds, device="cpu"):
    torch.manual_seed(seeds)
    net = Net(env.observation_space.shape, env.action_space.n)
    return net


def rand_pick(seq, probabilities):
    x = np.random.uniform(0 ,1)
    cum_prob = 0.0
    for item, item_pro in zip(seq, probabilities):
        cum_prob += item_pro
        if x < cum_prob:
            break
    return item


def worker_func(input_queue, output_queue, device_w="cpu"):
    new_env = make_env()
    parent_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    pro_list = [0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    #parent_list = [0, 1]
    #pro_list = [0.6, 0.4]

    while True:
        parents_w = input_queue.get()
        batch_steps_w = 0
        child = []
        logger.debug("in worker_func, current_process: {0},parents[0][0]:{1},len of parents:{2}".
                     format(mp.current_process(), parents_w[0]['fc.2.bias'], len(parents_w)))
        for _ in range(SEEDS_PER_WORKER):
            #random = np.random.uniform()
            parent = rand_pick(parent_list, pro_list)
            #parent = np.random.randint(PARENTS_COUNT)
            child_seed = np.random.randint(MAX_SEED)
            child_net = mutate_net(new_env, parents_w[parent], child_seed).to(device_w)
            reward, steps = evaluate(new_env, child_net, device_w)
            batch_steps_w += steps
            child.append((child_net.state_dict(), reward, steps))
        child.sort(key=lambda p: p[1], reverse=True)
        #logger.debug("middle, current_process: {0},child[0][1]:{1},child[0][2]:{2},len of "
        #             "child:{3}".format(mp.current_process(), child[0][1], child[0][2], len(child)))
        for i in range(PARENTS_COUNT):
            #output_queue.put(child[i])
            output_queue.put(OutputItem(child_net=child[i][0], reward=child[i][1], steps=batch_steps_w))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment="-pong-ga")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = "cuda:1" if args.cuda else "cpu"

    env = make_env()
    #create PARENTS_COUNT parents
    parents = []
    for i in range(PARENTS_COUNT):
        seed = np.random.randint(MAX_SEED)
        net = build_net(env, seed).to(device)
        parents.append((net.state_dict()))

    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    time_start = time.time()

    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue, device))
        w.start()
        input_queue.put(parents)

    logger.debug("Before++++, current_process: {0},parents[0]['fc.2.bias']:{1},new_parents[1]['fc.2.bias']:{2}, "
                 "len of parents:{3}, type of parents:{4}".format(mp.current_process(), parents[0]['fc.2.bias'],
                                                                  parents[1]['fc.2.bias'], len(parents), type(parents)))
    gen_idx = 0
    elite = None
    while True:
        t_start = time.time()
        batch_steps = 0
        top_children = []

        while len(top_children) < WORKERS_COUNT * PARENTS_COUNT:
            out_item = output_queue.get()
            top_children.append((out_item.child_net, out_item.reward))
            batch_steps += out_item.steps
        if elite is not None:
            top_children.append(elite)

        rewards = [p[1] for p in top_children]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        writer.add_scalar("batch_steps", batch_steps, gen_idx)
        writer.add_scalar("gen_seconds", time.time() - t_start, gen_idx)
        speed = batch_steps / ((time.time() - t_start)*PARENTS_COUNT)
        writer.add_scalar("speed", speed, gen_idx)
        total_time = (time.time() - time_start) / 60
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s, total_running_time=%.2f/m" % (
            gen_idx, reward_mean, reward_max, reward_std, speed, total_time))

        # next_parents = []
        # for i in range(PARENTS_COUNT):
        #     next_parents[i] = copy.deepcopy(top_children[:PARENTS_COUNT])
        #     new_parents.append(children[i][0])#[i] = copy.deepcopy(children[i][0])

        next_parents = []
        elite = top_children[0]

        for i in range(PARENTS_COUNT):
            next_parents.append(copy.deepcopy(top_children[i][0]))

        #next_parents = copy.deepcopy(top_children[:PARENTS_COUNT][0])

        #logger.debug("type of next_parents[0],children[0], len of next_parents,top_children", type(next_parents[0]),
        #            len(next_parents), type(top_children[0]), len(top_children))

        for worker_queue in input_queues:
            worker_queue.put(next_parents)
        logger.debug("After----, current_process: {0},new_parents[0]['fc.2.bias']:{1},new_parents[1]['fc.2.bias']:{2}, "
                     "len of new_parents:{3}, type of new_parents:{4}".
                     format(mp.current_process(), next_parents[0]['fc.2.bias'], next_parents[1]['fc.2.bias'],
                            len(next_parents), type(next_parents)))

        gen_idx += 1
