#!/usr/bin/env python3
import sys
# import gym
import ptan
import gym.spaces
# import roboschool
import collections
import copy
import time
import numpy as np
import argparse
import logging
import pickle

import torch
import torch.nn as nn
import multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

# test-2 gpus
#PARENTS_COUNT = 10
# workers_number = 20
# POPULATION_PER_WORKER = 100
# debug
# workers_number = 2
# POPULATION_PER_WORKER = 5

# # test-8 gpus
# PARENTS_COUNT = 10
# WORKERS_COUNT = 24
# POPULATION_PER_WORKER = 100

# # debug
# PARENTS_COUNT = 2
# WORKERS_COUNT = 2
# POPULATION_PER_WORKER = 10
#

MAX_SEED = 2**32 - 1

# logger = logging.getLogger(__name__)
# fh = logging.FileHandler('debug.log')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# logger.addHandler(fh)


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


def evaluate(env_e, net, device="cpu"):
    obs = env_e.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([np.array(obs, copy=False)]).to(device)
        act_prob = net(obs_v).to(device)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env_e.step(acts.data.cpu().numpy()[0])
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


def mutate_net(env_m, p_net, seed, noise_std, device):
    #new_net = copy.deepcopy(load_state_dict(net)).to(device)
    #new_net.share_memory()
    new_net_m = Net(env_m.observation_space.shape, env_m.action_space.n).to(device)
    new_net_m.load_state_dict(p_net)
    np.random.seed(seed)
    for p in new_net_m.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32)).to(device)
        p.data += noise_std * noise_t
    return new_net_m


# out_item = (reward_max_p, speed_p)
OutputItem = collections.namedtuple('OutputItem', field_names=['top_children_p', 'speed_p'])


# def build_net(env_b, seeds, device="cpu"):
#     torch.manual_seed(seeds)
#     net_new = Net(env_b.observation_space.shape, env_b.action_space.n)
#     net_new.to(device)
#     return net_new


def worker_func(input_w):  # pro, scale_step_w, device_w="cpu"):
    t_start = time.time()
    # parent_list = []
    # pro_list = input_w[0]
    scale_step_w = input_w[0]
    device_w = input_w[1]
    env_w = input_w[2]
    population_per_worker_w = input_w[3]

    # this is necessary
    if device_w != "cpu":
        device_w_id = int(device_w[-1])
        torch.cuda.set_device(device_w_id)

    # for m in range(PARENTS_COUNT):
    #     parent_list.append(m)

    # elite = None
    batch_steps_w = 0
    child = []
    with open(r"my_trainer_objects.pkl", "rb") as input_file:
        parents_w = pickle.load(input_file)

    noise_step = np.random.normal(scale=scale_step_w)
    for _ in range(population_per_worker_w):
        parent = np.random.randint(0, len(parents_w))
        child_seed = np.random.randint(MAX_SEED)
        child_net = mutate_net(env_w, parents_w[parent], child_seed, noise_step, device_w)
        reward, steps = evaluate(env_w, child_net, device_w)
        batch_steps_w += steps
        child.append((child_net.state_dict(), reward))
    # if elite:
    #     child.extend(elite)
    #     elite = []
    # if elite is not None:
    #     child.append(elite)
    child.sort(key=lambda p: p[1], reverse=True)
    # for k in range(ELITE_NUMBER):
    #     elite.append(copy.deepcopy(child[k]))
    # elite = copy.deepcopy(child[0])
    speed_p = batch_steps_w / (time.time() - t_start)
    #out_item = (child[0], speed_p)
    top_children_w = []
    # out_item = (reward_max_p, speed_p)
    for k in range(2):
        top_children_w.append(child[k])
       # top_children_w.append((child[k][0].state_dict(), child[k][1])) # cpu()
    # reward_max_w = top_children_w[0][1]
    # if reward_max_w != -21:
    # return OutputItem(top_children_w, speed_p=speed_p)
    # logger.debug("After, current_process: {0}, top_children_w[0]:{1},child[0]:{2},reward_max:{3}".
    #              format(mp.current_process(), top_children_w[0][0]['fc.2.bias'],
    #                     child[0][0].state_dict()['fc.2.bias'], top_children_w[0][1]))
    #return out_item
    return OutputItem(top_children_p=top_children_w, speed_p=speed_p)
    # output_queue_w.put(OutputItem(top_children_w, speed_p=speed_p))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment="-pong-ga-multi-population-exp")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug")
    args = parser.parse_args()
    devices = []

    logger = logging.getLogger(__name__)
    if args.debug:
        logger.setLevel(level=logging.DEBUG)
        workers_number = 2
        population_per_worker = 5
    else:
        logger.setLevel(level=logging.INFO)
        workers_number = 20
        population_per_worker = 100
    fh = logging.FileHandler('debug.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    gpu_number = torch.cuda.device_count()
    if gpu_number >= 1 and args.cuda:
        for i in range(gpu_number):
            devices.append("cuda:{0}".format(i))

    env = make_env()
    output_queue = mp.SimpleQueue()
    time_start = time.time()
    share_parents = []
    input_queues = []

    # create PARENTS_COUNT parents to share
    for _ in range(workers_number+1):
        seed = np.random.randint(MAX_SEED)
        torch.manual_seed(seed)
        share_parent = Net(env.observation_space.shape, env.action_space.n)#.cuda()#.cpu()
        share_parents.append(share_parent.state_dict())

    with open(r"my_trainer_objects.pkl", "wb") as output_file:
        pickle.dump(share_parents, output_file, True)

    #logger.debug("parent[0]['fc.2.bias']:{}".format(share_parents[0]))
    # value_d = []
    # for l in range(PARENTS_COUNT):
    #     value_d.append(1/PARENTS_COUNT)
    # pro = F.softmax(torch.tensor(value_d), dim=0)

    #workers_number = 20 # mp.cpu_count()  # 10
    #p_input = []
    init_scale = 1
    speed = 0
    gen_idx = 0
    reward_max_last = None
    elite = None
    Increase = False

    while True:
        p_input = []
        for u in range(workers_number):
            scale_step = (u + 1) * (init_scale / workers_number)
            if gpu_number == 0:
                device = "cpu"
            else:
                device_id = u % gpu_number
                device = devices[device_id]
            p_input.append((scale_step, device, env, population_per_worker))

        pool = mp.Pool(workers_number)  # mp.cpu_count()
        # logger.debug("cpu_count():{0}".format(mp.cpu_count()))
        result = pool.map(worker_func, p_input)
        pool.close()
        pool.join()
        logger.debug("len of result:{0}, type of result[0]:{1}".format(len(result), type(result[0])))

        top_children = []
        for item in result:
            top_children.extend(item.top_children_p)
            speed += item.speed_p

        if elite is not None:
            top_children.append(elite)
            logger.debug("elite:{}".format(elite[0]['fc.2.bias']))

        logger.debug("len of top_children:{0}, top_children[0]:{1}".
                     format(len(top_children), top_children[0][0]['fc.2.bias']))
        top_children.sort(key=lambda p: p[1], reverse=True)
        elite = copy.deepcopy(top_children[0])
        top_rewards = [p[1] for p in top_children]
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

        next_parents = []
        # top_children[i][0]
        # logger.debug("len of top_children:{0}".format(len(top_children)))
        # assert len(top_children) == 24
        for i in range(len(top_children)):
            new_net = Net(env.observation_space.shape, env.action_space.n)  # .to(device)
            new_net.load_state_dict(top_children[i][0])
            next_parents.append(new_net.cpu().state_dict())
            # next_parents.append(top_children[i][0].cpu())
        # value_d = []
        # for l in range(PARENTS_COUNT):
        #     value_d.append(top_children[l][1])
        # pro = F.softmax(torch.tensor(value_d), dim=0)
        # elite = copy.deepcopy(top_children[0])
        logger.debug("Main, next_parents[0]:{0}, init_scale:{1}, len(next_parents:{2})".
                     format(next_parents[0]['fc.2.bias'], init_scale, len(next_parents)))

        with open(r"my_trainer_objects.pkl", "wb") as output_file:
            pickle.dump(next_parents, output_file, True)

        if reward_max == reward_max_last:
            if init_scale == 0.0625:
                Increase = True
            if init_scale == 1:
                Increase = False
            if Increase:
                init_scale = init_scale * 2
            else:
                init_scale = init_scale / 2

        reward_max_last = reward_max
        gen_idx += 1
        speed = 0

    # gen_idx = 0
    # logger.debug("share_parent[0]['fc.2.bias']:{0}".format(share_parents[0]['fc.2.bias']))
    # while True:
    #     top_children = []
    #     speed = 0
    #     # out_item = (reward_max_p, speed_p)
    #     while len(top_children) < (WORKERS_COUNT*PARENTS_COUNT):
    #         out_item = output_queue.get()
    #         top_children.extend(out_item.top_children)
    #         speed += out_item.speed_p
    #
    #     top_children.sort(key=lambda p: p[1], reverse=True)
    #     top_rewards = [p[1] for p in top_children[:PARENTS_COUNT]]
    #     reward_mean = np.mean(top_rewards)
    #     reward_max = np.max(top_rewards)
    #     reward_std = np.std(top_rewards)
    #     writer.add_scalar("reward_mean", reward_mean, gen_idx)
    #     writer.add_scalar("reward_std", reward_std, gen_idx)
    #     writer.add_scalar("reward_max", reward_max, gen_idx)
    #     writer.add_scalar("speed", speed, gen_idx)
    #     total_time = (time.time() - time_start) / 60
    #     print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s, total_running_time=%.2f/m" % (
    #         gen_idx, reward_mean, reward_max, reward_std, speed, total_time))
    #
    #     if reward_mean == 21:
    #         exit(0)
    #     next_parents = []
    #     #top_children[i][0]
    #     logger.debug("len of top_children:{0}".format(len(top_children)))
    #     # assert len(top_children) == 24
    #     for i in range(PARENTS_COUNT):
    #         new_net = Net(env.observation_space.shape, env.action_space.n)#.to(device)
    #         new_net.load_state_dict(top_children[i][0])
    #         next_parents.append(new_net.cpu().state_dict())
    #         #next_parents.append(top_children[i][0].cpu())
    #     logger.debug("Main, next_parents[0]:{0}".format(next_parents[0]['fc.2.bias']))
    #     value_d = []
    #     for l in range(PARENTS_COUNT):
    #         value_d.append(top_children[l][1])
    #     pro = F.softmax(torch.tensor(value_d), dim=0)
    #
    #     with open(r"my_trainer_objects.pkl", "wb") as output_file:
    #         pickle.dump(next_parents, output_file, True)
    #
    #     for worker_queue in input_queues:
    #         worker_queue.put(pro)
    #     gen_idx += 1

        # test pool-version branch
