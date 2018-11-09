#!/usr/bin/env python3
import sys
import gym
import ptan
import collections
import copy
import time
import numpy as np
import argparse
import logging
import pickle
import json

import torch
import torch.nn as nn
import multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import wraps

from tensorboardX import SummaryWriter
from contextlib import contextmanager


MAX_SEED = 2**32 - 1
# logger = logging.getLogger(__name__)
# fh = logging.FileHandler('debug.log')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# logger.addHandler(fh)


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper


class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
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


def make_env(game):
    return ptan.common.wrappers.wrap_dqn(gym.make(game))


def evaluate(env_e, net, device="cpu", evaluate_episodes=1):
    steps = 0
    rewards = []
    for _ in range(evaluate_episodes):
        obs = env_e.reset()
        reward = 0.0
        while True:
            obs_v = torch.FloatTensor([np.array(obs, copy=False)]).to(device)
            act_prob = net(obs_v).to(device)
            for p in net.modules():
                if isinstance(p, nn.Linear):
                    bias_data = p.bias.data

            print("bias_data mean in evaluate:", bias_data.mean())
            print("act_prob mean in evaluate:", act_prob.mean())

            assert bias_data.mean() == act_prob.mean()

            # print("act_prob in evaluate:", act_prob)
            # print("act_prob mean in evaluate:", act_prob.mean())
            acts = act_prob.max(dim=1)[1]
            obs, r, done, _ = env_e.step(acts.data.cpu().numpy()[0])
            reward += r
            steps += 4
            if done:
                rewards.append(reward)
                break
    return np.mean(rewards), steps


def mutate_net(env_m, p_net, seed, scale_step, device):
    new_net_m = Net(env_m.observation_space.shape, env_m.action_space.n).to(device)
    new_net_m.load_state_dict(p_net)
    np.random.seed(seed)
    for p in new_net_m.parameters():
        # np.random.seed(seed)
        noise_t = torch.tensor(np.random.normal(scale=scale_step, size=p.data.size()).astype(np.float32)).to(device)
        p.data += 0.002 * noise_t # noise_std - > 0.002
    return new_net_m


# out_item = (reward_max_p, speed_p)
OutputItem = collections.namedtuple('OutputItem', field_names=['top_children_p', 'frames'])


def worker_func(input_w):  # pro, scale_step_w, device_w="cpu"):
    scale_step_w = input_w[0]
    device_w = input_w[1]
    population_per_worker_w = input_w[2]
    game = input_w[3]
    env_w = make_env(game)

    # this is necessary
    if device_w != "cpu":
        device_w_id = int(device_w[-1])
        torch.cuda.set_device(device_w_id)

    batch_steps_w = 0
    child = []

    with open(r"my_trainer_objects.pkl", "rb") as input_file:
        parents_w = pickle.load(input_file)

    # noise_step = np.random.normal(scale=scale_step_w)
    for _ in range(population_per_worker_w):
        # noise_step = np.random.normal(scale=scale_step_w)
        parent = np.random.randint(0, len(parents_w))
        child_seed = np.random.randint(MAX_SEED)
        child_net = mutate_net(env_w, parents_w[parent], child_seed, scale_step_w, device_w)
        reward, steps = evaluate(env_w, child_net, device_w)
        batch_steps_w += steps
        child.append((child_net.state_dict(), reward))
    child.sort(key=lambda p: p[1], reverse=True)
    frames = batch_steps_w
    top_children_w = []
    for k in range(len(parents_w)):
        top_children_w.append(child[k])

    return OutputItem(top_children_p=top_children_w, frames=frames)


@contextmanager
def timeblock(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('{} : {}'.format(label, end - start))


def evolve(game, exp, logger):
    env = make_env(game)
    species_number = exp["species_number"]
    population_per_worker = exp["population_per_worker"]
    parents_number = exp["parents_number"]
    init_scale = exp["init_scale"]
    if exp["frames"][-1] == "B":
        frames = 1000000000*int(exp["frames"][:-1])
    elif exp["frames"][-1] == "M":
        frames = 1000000*int(exp["frames"][:-1])
    else:
        frames = int(exp["frames"])

    logger.info("frames:{}".format(frames))
    devices = []
    evolve_result = {}
    writer = SummaryWriter(comment="-pong-ga-multi-species")

    frames_per_g = 0
    gen_idx = 0
    reward_max_last = None
    elite = None
    all_frames = 0
    # best_agent = None
    best_average_reward = float('-inf')

    gpu_number = torch.cuda.device_count()
    if gpu_number >= 1:
        for i in range(gpu_number):
            devices.append("cuda:{0}".format(i))

    time_start = time.time()
    share_parents = []

    # create PARENTS_COUNT parents to share
    for _ in range(parents_number):
        seed = np.random.randint(MAX_SEED)
        torch.manual_seed(seed)
        share_parent = Net(env.observation_space.shape, env.action_space.n)
        # print("model.parameter.numbers:{}".format(sum([p.numel() for p in share_parent.parameters()])))
        share_parents.append(share_parent.state_dict())

    with open(r"my_trainer_objects.pkl", "wb") as output_file:
        pickle.dump(share_parents, output_file, True)

    while all_frames < frames:
        p_input = []
        scale_steps = []
        t_start = time.time()

        # for m in range(species_number):
        #     scale_step = (m + 1) * (init_scale / species_number)
        #     scale_steps.append(scale_step)

        for u in range(species_number):
            # scale_idx = np.random.randint(0, species_number)
            # scale_step = scale_steps[scale_idx]
            scale_step = (u+1)*0.3
            logger.debug("in evolve:scale_step:{}".format(scale_step))
            if gpu_number == 0:
                device = "cpu"
            else:
                device_id = u % gpu_number
                device = devices[device_id]
            p_input.append((scale_step, device, population_per_worker, game))

        pool = mp.Pool(species_number)
        result = pool.map(worker_func, p_input)
        pool.close()
        pool.join()

        top_children = []
        for item in result:
            top_children.extend(item.top_children_p)
            frames_per_g += item.frames

        all_frames = all_frames + frames_per_g
        speed = frames_per_g / (time.time() - t_start)

        if elite is not None:
            top_children.append(elite)
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

        logger.info("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s, "
                    "init_scale=%.2f, total_running_time=%.2f/m" % (
                     gen_idx, reward_mean, reward_max, reward_std, speed, init_scale, total_time))

        next_parents = []
        for i in range(parents_number):
            new_net = Net(env.observation_space.shape, env.action_space.n)
            new_net.load_state_dict(top_children[i][0])
            # p_reward, steps = evaluate(env, new_net, device=device, evaluate_episodes=10)
            # all_frames = all_frames+steps
            # elite_c.append((new_net.cpu().state_dict(), p_reward))
            next_parents.append(new_net.cpu().state_dict())
        # elite_c.sort(key=lambda p: p[1], reverse=True)
        # elite = copy.deepcopy(elite_c[0])
        # for j in range(parents_number):
        #     next_parents.append(elite_c[j][0])

        # evaluate top 5 best parents for 10 times
        for i in range(5):
            test_best_net = Net(env.observation_space.shape, env.action_space.n)
            test_best_net.load_state_dict(next_parents[i])
            reward, steps = evaluate(env, test_best_net, evaluate_episodes=100)
            all_frames += steps
            # writer.add_scalar("best_agent %s" % i, reward, all_frames)
            if reward > best_average_reward:
                # best_agent = copy.deepcopy(next_parents[i])
                best_average_reward = reward

        writer.add_scalar("best_agent_score", best_average_reward, all_frames)
        logger.info("best agent average value:{}".format(best_average_reward))
        logger.info("top_children[0] reward:{0}, top_children[1] reward:{1}".format(top_children[0][1],
                                                                                     top_children[1][1]))
        logger.info("next_parents[0]:{0},next_parents[1]:{1}".format(next_parents[0]['fc.2.bias'],
                                                                      next_parents[1]['fc.2.bias']))
        logger.info("all_frames:{}".format(all_frames))
        with open(r"my_trainer_objects.pkl", "wb") as output_file:
            pickle.dump(next_parents, output_file, True)

        if reward_max == reward_max_last:
            if round(init_scale, 1) > 0.1:
                logging.debug("init_scale:{}".format(init_scale))
                init_scale = 0.9 * init_scale

        reward_max_last = reward_max
        gen_idx += 1
        frames_per_g = 0

    total_time = (time.time() - time_start) / 60
    evolve_result["gen_idx"] = gen_idx
    evolve_result["reward_mean"] = reward_mean
    evolve_result["reward_max"] = reward_max
    evolve_result["reward_std"] = reward_std
    evolve_result["speed"] = speed
    evolve_result["total_time"] = total_time
    evolve_result["all_frames"] = all_frames / 1000

    return evolve_result


def main(**exp):
    mp.set_start_method('spawn')
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler('./logger.out')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if exp["debug"]:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    logger.info("{}".format(str(json.dumps(exp, indent=4, sort_keys=True))))
    games = exp["games"].split(',')
    logger.info("games:{}".format(games))

    for game in games:
        g_result = evolve(game, exp, logger)
        logger.info("game=%s,gen_idx=%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s, "
                    "total_running_time=%.2f/m, all_frames=%.2fk" %
                    (game, g_result["gen_idx"], g_result["reward_mean"], g_result["reward_max"],
                     g_result["reward_std"], g_result["speed"], g_result["total_time"], g_result["all_frames"]))


if __name__ == "__main__":
    with open(sys.argv[-1], 'r') as f:
        exp = json.loads(f.read())
    main(**exp)
