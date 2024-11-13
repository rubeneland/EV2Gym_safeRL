#from GF.action_wrapper import ThreeStep_Action_DiscreteActionSpace, mask_fn, Fully_Discrete

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.reward import profit_maximization

from cost_functions import transformer_overload_usrpenalty_cost, ProfitMax_TrPenalty_UserIncentives_safety

from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

import gymnasium as gym
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import yaml
import random
import torch

from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads
from tianshou.env import DummyVectorEnv
from fsrl.agent import PPOLagAgent, CPOAgent
from fsrl.utils import TensorboardLogger

config_file = "V2GProfit_base.yaml"
reward_function = ProfitMax_TrPenalty_UserIncentives_safety #ProfitMax_TrPenalty_UserIncentives
state_function = V2G_profit_max
cost_function = transformer_overload_usrpenalty_cost

run_name =  'min_c_8_tr_100_usr_100_also_reward'
group_name = 'CPO'

# run = wandb.init(project='ev2gym',
#                      sync_tensorboard=True,
#                      group=group_name,
#                      name=run_name,
#                      save_code=True,
#                      )

gym.envs.register(id='evs-v0', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                      kwargs={'config_file': config_file,
                              'verbose': False,
                              'save_plots': False,
                              'generate_rnd_game': True,
                              'reward_function': reward_function,
                              'state_function': state_function,
                              'cost_function': cost_function,
                              })

task = "evs-v0"

save_path = f"./saved_models/{group_name}/{run_name}"

os.makedirs(f"./saved_models/{group_name}", exist_ok=True)
os.makedirs(save_path, exist_ok=True)

device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# init logger
logger = TensorboardLogger("logs", log_txt=True, name=task)
# CPO agent
agent = CPOAgent(gym.make(task), logger, cost_limit = 2, device=device)

# init the env
training_num, testing_num = 10, 10
train_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(testing_num)])

agent.learn(train_envs, test_envs, epoch=200)

torch.save(agent.policy.state_dict(), f"{save_path}/policy.pth")

print(f"Finished training CPO, saving model at {save_path}")   