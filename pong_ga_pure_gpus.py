#!/usr/bin/env python3
import sys
import gym
import ptan
import gym.spaces
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter


# POPULATION_SIZE = 1000#600
# PARENTS_COUNT = 20
# WORKERS_COUNT = 20#6
POPULATION_SIZE = 8
PARENTS_COUNT = 4
WORKERS_COUNT = 2

#NOISE_STD = 0.01
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


def mutate_net(env, net, seed, noise_std, copy_net=True):
    new_net = Net(env.observation_space.shape, env.action_space.n)
    new_net.load_state_dict(net)
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))#.cuda()
        p.data += noise_std * noise_t
    return new_net


OutputItem = collections.namedtuple('OutputItem', field_names=['child_net', 'reward', 'steps'])


def build_net(env, seeds, device="cpu"):
    torch.manual_seed(seeds)
    net_new = Net(env.observation_space.shape, env.action_space.n)
    # if torch.cuda.device_count() > 1:
    #     logger.debug("let's use {0} gpus.".format(torch.cuda.device_count()))
    #     net_new = torch.nn.DataParallel(net_new)
    net_new.to(device)

    #net_new = Net(env.observation_space.shape, env.action_space.n)
    return net_new


# class Dataset(Dataset):
#     def __init__(self, size, length):
#         self.len = length
#         self.data = torch.randn(length, size)
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#     def __len__(self):
#         return self.len


def worker_func(input_queue, output_queue, device_w="cpu"):
    new_env = make_env()
    parent_list = []
    for i in range(PARENTS_COUNT):
        parent_list.append(i)
    #parent_list = [0, 1]
    logger.debug("in work_func,current_process: {0},parent_list:{1}".format(mp.current_process(), parent_list))

    while True:
        get_item = input_queue.get()
        parents_w = get_item[0]
        pro_list = get_item[1]
        noise_step_w = get_item[2]

        batch_steps_w = 0
        child = []
        logger.debug("in worker_func, current_process: {0},parents[0][0]:{1},len of parents:{2},pro_list:{3}, noise_step_w:{4}".
                     format(mp.current_process(), parents_w[0]['fc.2.bias'], len(parents_w), pro_list, noise_step_w))
        for _ in range(SEEDS_PER_WORKER):
            #solve pro do not sum to 1
            pro_list = np.array(pro_list)
            pro_list = pro_list/sum(pro_list)
            parent = np.random.choice(parent_list, p=pro_list)
            child_seed = np.random.randint(MAX_SEED)
            child_net = mutate_net(new_env, parents_w[parent], child_seed, noise_step_w).to(device_w)
            reward, steps = evaluate(new_env, child_net, device_w)
            batch_steps_w += steps
            child.append((child_net.state_dict(), reward, steps))
        child.sort(key=lambda p: p[1], reverse=True)
        logger.debug("middle, current_process: {0},child[0][1]:{1}, len of "
                     "child:{2}".format(mp.current_process(), child[0][0]['fc.2.bias'], len(child)))
        for i in range(PARENTS_COUNT):
            output_queue.put(OutputItem(child_net=child[i][0], reward=child[i][1], steps=batch_steps_w))

# def cac_noise_std(gen_idx, reward_max):
#     if reward_max == before_reward_max:
#         count = count+1
#     if count == 3:
#         noise_step = noise_step/2
#         count = 0
#     before_reward_max = reward_max
#     return noise_step
#


if __name__ == "__main__":
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment="-pong-ga-pure-gpus")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    #device = "cuda" if args.cuda else "cpu"
    device0 = "cuda:0" if args.cuda else "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = "cuda:0" if args.cuda else "cpu"#torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    logger.debug("gpu number:{0}".format(torch.cuda.device_count()))

    if torch.cuda.device_count() > 1:
        device1 = "cuda:1" if args.cuda else "cpu"

    env = make_env()
    noise_step = 0.06
    #create PARENTS_COUNT parents
    parents = []
    for i in range(PARENTS_COUNT):
        seed = np.random.randint(MAX_SEED)
        net = build_net(env, seed).to(device0)
        parents.append(net.state_dict())

    logger.debug("Before++++, current_process: {0},parents[0]:{1}".format(mp.current_process(), parents[0]['fc.2.bias']))

    input_queues = []
    count = 0
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    probability = []
    time_start = time.time()
    for _ in range(PARENTS_COUNT):
        probability.append(1 / PARENTS_COUNT)

    for j in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        if j >= (WORKERS_COUNT/2-1):
            w = mp.Process(target=worker_func, args=(input_queue, output_queue, device1))
        else:
            w = mp.Process(target=worker_func, args=(input_queue, output_queue, device0))
        w.start()
        input_queue.put((parents, probability, noise_step))

    gen_idx = 0
    elite = None
    reward_max_temp = 0
    count = 0

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

        top_children.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in top_children[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        writer.add_scalar("batch_steps", batch_steps, gen_idx)
        writer.add_scalar("gen_seconds", time.time() - t_start, gen_idx)
        writer.add_scalar("noise_step", noise_step, gen_idx)

        speed = batch_steps / ((time.time() - t_start)*PARENTS_COUNT)
        writer.add_scalar("speed", speed, gen_idx)
        total_time = (time.time() - time_start) / 60
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s, noise_step=%f, total_running_time=%.2f/m" % (
            gen_idx, reward_mean, reward_max, reward_std, speed, noise_step, total_time))

        if reward_mean == 21:
            exit(0)

        value_d = []
        for i in range(PARENTS_COUNT):
            value_d.append(top_children[i][1]+21)
        logger.debug("value_d:{0}".format(value_d))
        probability = F.softmax(torch.tensor(value_d), dim=0)
        logger.debug("probability:{0}".format(probability))

        next_parents = []
        elite = copy.deepcopy(top_children[0])

        for i in range(PARENTS_COUNT):
            #deep copy solve the invalid device bug
            next_parents.append(copy.deepcopy(top_children[i][0]))

        if reward_max == reward_max_temp:
            count = count+1
        if count >= 3:
            noise_step = noise_step/2
        if count >= 5:
            m = torch.distributions.normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
            noise_step = m.sample()
            count = 0

        for worker_queue in input_queues:
            worker_queue.put((next_parents, probability, noise_step))
        logger.debug("After----, current_process: {0},new_parents[0]['fc.2.bias']:{1},new_parents[1]['fc.2.bias']:{2}, "
                     "len of new_parents:{3}, type of new_parents:{4}".
                     format(mp.current_process(), next_parents[0]['fc.2.bias'], next_parents[1]['fc.2.bias'],
                            len(next_parents), type(next_parents)))
        gen_idx += 1
        reward_max_temp = reward_max

