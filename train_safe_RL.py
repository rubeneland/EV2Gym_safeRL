# this file is used to evalaute the performance of the ev2gym environment with safe RL algorithms from the fsrl library

#from GF.action_wrapper import ThreeStep_Action_DiscreteActionSpace, mask_fn, Fully_Discrete

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.reward import profit_maximization

from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

import gymnasium as gym
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import yaml
import random

from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads
from tianshou.env import DummyVectorEnv
from fsrl.agent import PPOLagAgent, CPOAgent
from fsrl.utils import TensorboardLogger

config_file = "V2GProfit_base.yaml"
reward_function = ProfitMax_TrPenalty_UserIncentives
state_function = V2G_profit_max

gym.envs.register(id='evs-v0', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                      kwargs={'config_file': config_file,
                              'verbose': False,
                              'save_plots': False,
                              'generate_rnd_game': True,
                              'reward_function': reward_function,
                              'state_function': state_function,
                            #   'cost_function': profit_maximization,
                              })

task = "evs-v0"
# init logger
logger = TensorboardLogger("logs", log_txt=True, name=task)
# init the PPO Lag agent with default parameters
agent = CPOAgent(gym.make(task), logger)
# init the env
training_num, testing_num = 1, 1
train_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(testing_num)])

agent.learn(train_envs, test_envs, epoch=100)