import copy, os, logging, shutil
from collections import deque

import numpy as np
from skimage.color import rgb2grey
from skimage.transform import resize
import tensorflow as tf

from scipy.misc import imsave
from matplotlib import pyplot as plt
import statistics

from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.common.vec_env import VecEnv
#from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.a2c.a2c import learn, Model
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from baselines.a2c.utils import fc, conv, conv_to_fc, sample

import gym
from gym import spaces
from gym.envs import map_sim

class GymVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        print('---------------------------------------------')
        print('Environment "mapSim-v1" Successfully Initialized')
        self.remotes = [0]*len(env_fns)
        env = self.envs[0]
        self.action_space = env.action_space
        img_shape = (84, 84, 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=img_shape)
        self.ts = np.zeros(len(self.envs), dtype='int')

    def step(self, action_n, ind=0):
        obs = []
        rews = []
        dones = []
        infos = []
        imgs = []
        # print('action_n = ' + str(action_n))
        for (a,env) in zip(action_n, self.envs):
            ob, rew, done, info = env.step(action_n[ind], ind) # MAY NOT BE CORRECT
            # plt.imshow(ob)
            # plt.draw()
            # plt.pause(0.000001)
            # Need to fix below. What is the difference between obs and imgs?
            obs.append(ob)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
            imgs.append(ob)
        self.ts += 1
        for (i, done) in enumerate(dones):
            # print ('Debug')
            # print('dones ' + str(dones))
            # print('envs' + str(self.envs))
            # print()
            if done:
                # print (i)
                # obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return np.array(imgs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = []
        for env in self.envs:
            env.env, img = env.reset()
            results.append(img)
        #results = [env.reset() for env in self.envs]
        return env.env, np.array(results)

    @property
    def num_envs(self):
        return len(self.envs)

def policy_fn_name(policy_name):
    if policy_name == 'cnn':
        policy_fn = CnnPolicy
    elif policy_name == 'lstm':
        policy_fn = LstmPolicy
    elif policy_name == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy_name == 'mlp':
        policy_fn = MlpPolicy
    return policy_fn

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu, continuous_actions=False, numAgents=2):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed+rank)
            env.ID = rank
            # print("logger dir: ", logger.get_dir())
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            if env_id == 'Pendulum-v0':
                if continuous_actions:
                    env.action_space.n = env.action_space.shape[0]
                else:
                    env.action_space.n = 10
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    env = GymVecEnv([make_env(idx) for idx in range(num_cpu)])
    policy_fn = policy_fn_name(policy)
    learn(policy_fn, env, seed, nsteps=30, nstack=1, total_timesteps=int(num_timesteps * 1.1), lr=7e-4, lrschedule=lrschedule, continuous_actions=continuous_actions, numAgents=numAgents, continueTraining=False, debug=False)

def test(env_id, policy_name, seed, nstack=1):
    iters = 100
    rwd = []
    percent_exp = []
    env = gym.make(env_id)
    env.seed(seed)
    print("logger dir: ", logger.get_dir())
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()))
    if env_id == 'Pendulum-v0':
        if continuous_actions:
            env.action_space.n = env.action_space.shape[0]
        else:
            env.action_space.n = 10
    gym.logger.setLevel(logging.WARN)
    # img_shape = (84, 84, 3)
    img_shape = (84, 84, 3)
    ob_space = spaces.Box(low=0, high=255, shape=img_shape)
    ac_space = env.action_space

    # def get_img(env):
    #     ax, img = env.get_img()
    #    return ax, img

    # def process_img(img):
    #     img = rgb2grey(copy.deepcopy(img))
    #    img = resize(img, img_shape)
    #    return img

    policy_fn = policy_fn_name(policy_name)

    nsteps=5
    total_timesteps=int(80e6)
    vf_coef=0.5
    ent_coef=0.01
    max_grad_norm=0.5
    lr=7e-4
    lrschedule='linear'
    epsilon=1e-5
    alpha=0.99
    continuous_actions=False
    debug=False
    # if i == 0:
    model = Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, nenvs=1, nsteps=nsteps, nstack=nstack, num_procs=1, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, continuous_actions=continuous_actions, debug=debug)

    m_name = 'test_model_Mar7_1mil.pkl'
    model.load(m_name)

    env.env, img = env.reset()
    print('---------------------------------------------')
    print("Initializing Test for: ", m_name)
    print('---------------------------------------------')
    for i in range(1, iters+1):
        if i % 10 == 0:
            print('-----------------------------------')
            print('Iteration: ', i)
            avg_rwd = sum(rwd)/i
            avg_pct_exp = sum(percent_exp)/i
            med_pct_exp = statistics.median(percent_exp)
            print('Average Reward: ', avg_rwd)
            print('Average Percent Explored: ', avg_pct_exp, '%')
            print('Median Percent Explored: ', med_pct_exp)
            print('-----------------------------------')
        frames_dir = 'exp_frames' + str(i+100)
        if os.path.exists(frames_dir):
            # raise ValueError('Frames directory already exists.')
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir)
        # ax, img = get_img(env)
        img_hist = deque([img for _ in range(4)], maxlen=nstack)
        action = 0
        total_rewards = 0
        nstack = 2
        for tidx in range(1000):
            # if tidx % nstack == 0:
            if tidx > 0:
                input_imgs = np.expand_dims(np.squeeze(np.stack(img_hist, -1)), 0)
                # print(np.shape(input_imgs))
                # plt.imshow(input_imgs[0, :, :, 0])
                # plt.imshow(input_imgs[0, :, :, 1])
                # plt.draw()
                # plt.pause(0.000001)
                if input_imgs.shape == (1, 84, 84, 3):
                    actions, values, states = model.step_model.step(input_imgs)
                else:
                    actions, values, states = model.step_model.step(input_imgs[:, :, :, :, 0])
                # actions, values, states = model.step_model.step(input_imgs)
                action = actions[0]
                value = values[0]
                # print('Value: ', value, '   Action: ', action)

            img, reward, done, _ = env.step(action)
            total_rewards += reward
            # img = get_img(env)
            img_hist.append(img)
            imsave(os.path.join(frames_dir, 'frame_{:04d}.png'.format(tidx)), resize(img, (img_shape[0], img_shape[1], 3)))
            # print(tidx, '\tAction: ', action, '\tValues: ', value, '\tRewards: ', reward, '\tTotal rewards: ', total_rewards)#, flush=True)
            if done:
                # print('Faultered at tidx: ', tidx)
                rwd.append(total_rewards)
                percent_exp.append(env.env.percent_explored)
                env.env, img = env.reset()
                break

    print('-----------------------------------')
    print('Iteration: ', iters)
    avg_rwd = sum(rwd)/iters
    avg_pct_exp = sum(percent_exp)/iters
    med_pct_exp = statistics.median(percent_exp)
    print('Average Reward: ', avg_rwd)
    print('Average Percent Explored: ', avg_pct_exp, '%')
    print('Median Percent Explored: ', med_pct_exp)
    print('-----------------------------------')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='mapSim-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-timesteps', type=int, default=int(10e7))
    parser.add_argument('-c', '--continuous_actions', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--interactive', default=False, action='store_true')
    # parser.add_argument('--multiAgent', default=False, action='store_true')
    args = parser.parse_args()
    logger.configure()

    '''
    if args.env == 'Pendulum-v0':
        continuous_actions = True
    else:
    '''
    continuous_actions = False
    numAgents = 2

    if args.test:
        test(args.env, args.policy, args.seed, nstack=1)
    else:
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
              policy=args.policy, lrschedule=args.lrschedule, num_cpu=4, continuous_actions=continuous_actions, numAgents=numAgents)
