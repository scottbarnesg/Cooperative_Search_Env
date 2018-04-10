import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import pickle

# import maddpg.common.tf_util as U
# sfrom maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

# baselines libraries
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.common.vec_env import VecEnv
#from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.a2c.a2c import learn, Model
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
from baselines.a2c.utils import fc, conv, conv_to_fc, sample


def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    # print(world.agents)
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu, continuous_actions=False, numAgents=2, benchmark=False):
    # Create environment
    env = make_env(env_id, benchmark)
    # print('action space: ', env.action_space)
    # env = GymVecEnv([make_env(idx) for idx in range(num_cpu)])
    policy_fn = policy_fn_name(policy)
    learn(policy_fn, env, seed, nsteps=30, nstack=1, total_timesteps=int(num_timesteps * 1.1), lr=1e-5, lrschedule=lrschedule, continuous_actions=continuous_actions, numAgents=numAgents, continueTraining=False, debug=False, particleEnv=True, model_name='partEnv_model_')


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='simple_reference')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='mlp')
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
              policy=args.policy, lrschedule=args.lrschedule, num_cpu=1, continuous_actions=continuous_actions, numAgents=numAgents)
