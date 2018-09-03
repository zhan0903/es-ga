#!/usr/bin/env python3
import sys
#import gym
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
import multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

# test
# PARENTS_COUNT = 20
# WORKERS_COUNT = 32
# POPULATION_PER_WORKER = 100

# debug
PARENTS_COUNT = 2
WORKERS_COUNT = 4
POPULATION_PER_WORKER = 10

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


def mutate_net(net, seed, noise_std, device):
    new_net = copy.deepcopy(net).to(device)
    #new_net.share_memory()
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32)).to(device)
        p.data += noise_std * noise_t
    return new_net


# out_item = (reward_max_p, speed_p)
OutputItem = collections.namedtuple('OutputItem', field_names=['top_children', 'speed_p'])


def build_net(env_b, seeds, device="cpu"):
    torch.manual_seed(seeds)
    net_new = Net(env_b.observation_space.shape, env_b.action_space.n)
    net_new.to(device)
    return net_new


def worker_func(input_queue_w, output_queue_w, scale_step_w, device_w="cpu"):
    new_env = make_env()
    parent_list = []
    # this is necessary
    if device_w != "cpu":
        device_w_id = int(device_w[-1])
        torch.cuda.set_device(device_w_id)

    # # create PARENTS_COUNT's top_children
    # top_children = []
    # for _ in range(PARENTS_COUNT):
    #     seed = np.random.randint(MAX_SEED)
    #     child_net = build_net(new_env, seed).to(device_w)
    #     child_net.share_memory()
    #     top_children.append(child_net)

    for m in range(PARENTS_COUNT):
        parent_list.append(m)
    # pro_list = []
    # for _ in range(PARENTS_COUNT):
    #     pro_list.append(1 / PARENTS_COUNT)

    elite = None
    while True:
        t_start = time.time()
        batch_steps_w = 0
        child = []
        get_item = input_queue_w.get()
        parents_w = get_item[0]
        pro_list = get_item[1]


        # value_d = []
        # for l in range(PARENTS_COUNT):
        #     value_d.append(parents_w[l][1])
        # pro_list = F.softmax(torch.tensor(value_d), dim=0)

        #print("parent_w:{0}".format(parents_w[0]))
        noise_step = np.random.normal(scale=scale_step_w)
        logger.debug("Before, current_process: {0}, parents:{1}".format(mp.current_process(), parents_w[0].state_dict()['fc.2.bias']))
        for _ in range(POPULATION_PER_WORKER):
            # solve pro do not sum to 1
            pro_list = np.array(pro_list)
            pro_list = pro_list/sum(pro_list)
            parent = np.random.choice(parent_list, p=pro_list)
            child_seed = np.random.randint(MAX_SEED)
            child_net = mutate_net(parents_w[parent], child_seed, noise_step, device_w)
            reward, steps = evaluate(new_env, child_net, device_w)
            batch_steps_w += steps
            child.append((child_net, reward))
        if elite is not None:
            child.append(elite)
        child.sort(key=lambda p: p[1], reverse=True)
        elite = copy.deepcopy(child[0])
        speed_p = batch_steps_w / (time.time() - t_start)
        top_children_w = []
        # out_item = (reward_max_p, speed_p)
        for k in range(PARENTS_COUNT):
            top_children_w.append((child[k][0].cpu(), child[k][1]))
        logger.debug("After, current_process: {0}, top_children_w[0]:{1},child[0]:{2}".format(mp.current_process(),
                     top_children_w[0][0].state_dict()['fc.2.bias'], child[0][0].state_dict()['fc.2.bias']))
        logger.debug("current_process: {0},len of child:{1}, pro_list:{2},reward_max_p:{3}".
                     format(mp.current_process(), len(child), pro_list, child[0][1]))
        output_queue_w.put(OutputItem(top_children_w, speed_p=speed_p))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment="-pong-ga-multi-population")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    devices = []

    gpu_number = torch.cuda.device_count()
    logger.debug("gpu number:{0}".format(torch.cuda.device_count()))
    if gpu_number >= 1 and args.cuda:
        for i in range(gpu_number):
            devices.append("cuda:{0}".format(i))

    env = make_env()
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    time_start = time.time()
    share_parents = []
    input_queues = []

    # create PARENTS_COUNT parents to share
    for _ in range(PARENTS_COUNT):
        share_parent = Net(env.observation_space.shape, env.action_space.n)
        #share_parent.share_memory()
        share_parents.append(share_parent)

    value_d = []
    for l in range(PARENTS_COUNT):
        value_d.append(1/PARENTS_COUNT)
    pro = F.softmax(torch.tensor(value_d), dim=0)

    for j in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        scale_step = (j+1)*0.001
        if gpu_number >= 1 and args.cuda:
            device_id = j % gpu_number
            logger.debug("device_id:{0}, worker id:{1}".format(device_id, j))
            w = mp.Process(target=worker_func, args=(input_queue, output_queue, scale_step, devices[device_id]))
        else:
            w = mp.Process(target=worker_func, args=(input_queue, output_queue, scale_step, "cpu"))
        w.start()
        input_queue.put((share_parents, pro))

    gen_idx = 0
    while True:
        top_children = []
        speed = 0
        # out_item = (reward_max_p, speed_p)
        while len(top_children) < WORKERS_COUNT:
            out_item = output_queue.get()
            top_children.extend(out_item.top_children)
            speed += out_item.speed_p

        top_children.sort(key=lambda p: p[1], reverse=True)
        top_rewards = [p[1] for p in top_children[:PARENTS_COUNT]]
        reward_mean = np.mean(top_rewards)
        reward_max = np.max(top_rewards)
        reward_std = np.std(top_rewards)
        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        writer.add_scalar("speed", speed, gen_idx)
        total_time = (time.time() - time_start) / 60
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s, total_running_time=%.2f/m" % (
            gen_idx, reward_mean, reward_max, reward_std, speed, total_time))

        if reward_mean == 21:
            exit(0)
        next_parents = []
        #for i in range(PARENTS_COUNT):
        #    share_parents[i] = copy.deepcopy(top_children[i][0])
        for i in range(PARENTS_COUNT):
            next_parents.append(copy.deepcopy(top_children[i][0]))
        logger.debug("After, current_process: {0}, len of next_parents:{1},next_parents[0]:{2}".
                     format(mp.current_process(), len(next_parents), next_parents[0].state_dict()['fc.2.bias']))
        value_d = []
        for l in range(PARENTS_COUNT):
            value_d.append(top_children[l][1])
        pro = F.softmax(torch.tensor(value_d), dim=0)

        for worker_queue in input_queues:
            worker_queue.put((next_parents, pro))
        gen_idx += 1

