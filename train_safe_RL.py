#from GF.action_wrapper import ThreeStep_Action_DiscreteActionSpace, mask_fn, Fully_Discrete

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.reward import profit_maximization

from cost_functions import tr_overload_usrpenalty_cost, usrpenalty_cost, ProfitMax_TrPenalty_UserIncentives_safety

from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

import gymnasium as gym
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import yaml
import random
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

import pprint
from dataclasses import asdict

from tianshou.data import VectorReplayBuffer
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb


from fsrl.data import FastCollector
from fsrl.agent import SACLagAgent, PPOLagAgent, CPOAgent, CVPOAgent
from fsrl.policy import CVPO
from fsrl.trainer import OffpolicyTrainer
from fsrl.utils import TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic
from fsrl.utils.net.continuous import DoubleCritic, SingleCritic

from dataclasses import dataclass
from typing import Tuple

from gymnasium import Wrapper

os.environ["WANDB_MODE"] = "offline"

class SpecMaxStepsWrapper(Wrapper):
    def __init__(self, env, spec_max_steps):
        """
        Initialize the wrapper.

        Parameters:
        - env: The environment to wrap.
        - spec_max_steps: The maximum number of steps allowed in the environment.
        """
        super(SpecMaxStepsWrapper, self).__init__(env)
        self.spec.max_episode_steps = spec_max_steps

# Environment configuration
config_file = "V2GProfit_base.yaml"
# config_file = "V2GProfit_loads.yaml"

if config_file == "V2GProfit_base.yaml":
        state_function = V2G_profit_max
        cost_function = usrpenalty_cost

if config_file == "V2GProfit_loads.yaml":
        state_function = V2G_profit_max_loads
        cost_function = tr_overload_usrpenalty_cost

reward_function = ProfitMax_TrPenalty_UserIncentives_safety

# Register the custom environment
gym.envs.register(
id='fsrl-v0',
entry_point='ev2gym.models.ev2gym_env:EV2Gym',
kwargs={
        'config_file': config_file,
        'verbose': False,
        'save_plots': False,
        'generate_rnd_game': True,
        'reward_function': reward_function,
        'state_function': state_function,
        'cost_function': cost_function,
},
)

task = "fsrl-v0"
env = gym.make(task)
env.spec.max_episode_steps = env.env.env.simulation_length

def train_cpo(args):

        seed: int = args.seed
        cost_limit: int = args.cost_limit
        epoch: int = args.epoch
        training_num: int = args.train_num
        testing_num: int = args.test_num
        step_per_epoch: int = 3000
        gamma: float = 0.99
        thread: int = 1
        buffer_size: int = 200000

        run_name= f'cpo_exp1_1_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        # run_name =  f'CPO_5spawn_20cs_cost_lim_{cost_limit}_epochs_{epoch}_usr_1000_train_envs_8_test_envs_8_run{random.randint(0, 1000)}'
        group_name = 'EXP1_5'                   

        wandb.init(project='experiment_1_5',
                        sync_tensorboard=True,
                        group=group_name,
                        name=run_name,
                        save_code=True,
                        mode = "offline",
                        )

        task = "fsrl-v0"

        # init logger
        logger = WandbLogger(log_dir="fsrl_logs/EXP1_5/cpo", log_txt=True, group=group_name, name=run_name)

        env = gym.make(task)

        # CPO agent
        agent = CPOAgent(env=env, logger=logger, cost_limit=cost_limit, seed=seed, thread=thread, gamma=gamma)

        train_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
        test_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(testing_num)])

        agent.learn(train_envs=train_envs, buffer_size = buffer_size, test_envs=test_envs, epoch=epoch, testing_num=testing_num,
                    episode_per_collect=training_num, step_per_epoch=step_per_epoch, save_interval=1)

def train_cvpo(args):

        # general task params
        task: str = "fsrl-v0"
        cost_limit: int = args.cost_limit
        device: str = "cpu"
        thread: int = 1  # was 4, if use "cpu" to train
        # seed: int = random.randint(0, 10000)
        seed: int = args.seed
        # seed: int = 10
        # CVPO arguments
        estep_iter_num: int = 1
        estep_kl: float = 0.02
        estep_dual_max: float = 20
        estep_dual_lr: float = 0.02
        sample_act_num: int = 16
        mstep_iter_num: int = 1
        mstep_kl_mu: float = 0.005
        mstep_kl_std: float = 0.0005
        mstep_dual_max: float = 0.5
        mstep_dual_lr: float = 0.1
        actor_lr: float = 5e-4
        critic_lr: float = 1e-3
        gamma: float = 0.99 # was 0.97
        n_step: int = 2
        tau: float = 0.05
        hidden_sizes: Tuple[int, ...] = (128, 128)
        double_critic: bool = False
        conditioned_sigma: bool = True
        unbounded: bool = False
        last_layer_scale: bool = False
        # collecting params
        epoch: int = args.epoch
        episode_per_collect: int = args.train_num
        step_per_epoch: int = 3000
        update_per_step: float = 0.2
        buffer_size: int = 200000
        worker: str = "ShmemVectorEnv"
        training_num: int = args.train_num
        testing_num: int = args.test_num
        # general train params
        batch_size: int = 256
        reward_threshold: float = 10000  # for early stop purpose
        save_interval: int = 1
        deterministic_eval: bool = True
        action_scaling: bool = True
        action_bound_method: str = "clip"
        resume: bool = False  # TODO
        save_ckpt: bool = True  # set this to True to save the policy model
        verbose: bool = False
        render: bool = False

        # Use 1 task in example.sh! More tasks will create more runs...

        group_name: str = "EXP1_5"
        # run_name= f'cvpo_v45_step_epoch_9k_buff_200k_5spawn_10cs_90kw_loads_0_6_PV_0_1_seed{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        # run_name= f'cvpo_v67_6_h28_20_usr_5spawn_10cs_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        # run_name= f'cvpo_step_epoch_3k_30chargers_buff_400k_exp1_3_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        run_name= f'cvpo_const_eff_0_9_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'

        wandb.init(project='experiment_1_5',
                        sync_tensorboard=True,
                        group=group_name,
                        name=run_name,
                        save_code=True,
                        )
        # init logger
        logger = WandbLogger(log_dir="fsrl_logs/EXP1_5", log_txt=True, group=group_name, name=run_name)

        env = gym.make(task)
        # env.spec.max_episode_steps = env.env.env.simulation_length

        sim_length = env.env.env.simulation_length

        agent = CVPOAgent(
                env=SpecMaxStepsWrapper(gym.make(task), sim_length),
                logger=logger,
                cost_limit=cost_limit,
                device=device,
                thread=thread,
                seed=seed,
                estep_iter_num=estep_iter_num,
                estep_kl=estep_kl,
                estep_dual_max=estep_dual_max,
                estep_dual_lr=estep_dual_lr,
                sample_act_num=sample_act_num,
                mstep_iter_num=mstep_iter_num,
                mstep_kl_mu=mstep_kl_mu,
                mstep_kl_std=mstep_kl_std,
                mstep_dual_max=mstep_dual_max,
                mstep_dual_lr=mstep_dual_lr,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                n_step=n_step,
                tau=tau,
                hidden_sizes=hidden_sizes,
                double_critic=double_critic,
                conditioned_sigma=conditioned_sigma,
                unbounded=unbounded,
                last_layer_scale=last_layer_scale,
                deterministic_eval=deterministic_eval,
                action_scaling=action_scaling,
                action_bound_method=action_bound_method,
                lr_scheduler=None
        )

        # training_num = min(training_num, episode_per_collect)
        worker = eval(worker)
        train_envs = worker([lambda: SpecMaxStepsWrapper(gym.make(task), sim_length) for _ in range(training_num)])
        test_envs = worker([lambda: SpecMaxStepsWrapper(gym.make(task), sim_length) for _ in range(testing_num)])

        # start training
        agent.learn(
                train_envs=train_envs,
                test_envs=test_envs,
                epoch=epoch,
                episode_per_collect=episode_per_collect,
                step_per_epoch=step_per_epoch,
                update_per_step=update_per_step,
                buffer_size=buffer_size,
                testing_num=testing_num,
                batch_size=batch_size,
                reward_threshold=reward_threshold,  # for early stop purpose
                save_interval=save_interval,
                resume=resume,
                save_ckpt=save_ckpt,  # set this to True to save the policy model,
                verbose=verbose,
        )

def train_ppol(args):
        # general task params
        task: str = "fsrl-v0"
        cost_limit: int = args.cost_limit
        device: str = "cpu"
        thread: int = 1  # if use "cpu" to train
        # seed: int = random.randint(0, 1000)
        seed: int = args.seed
        # algorithm params
        lr: float = 5e-4
        hidden_sizes: Tuple[int, ...] = (128, 128)
        unbounded: bool = False
        last_layer_scale: bool = False
        # PPO specific arguments
        target_kl: float = 0.05
        vf_coef: float = 0.25
        max_grad_norm: float = 0.5
        gae_lambda: float = 0.95
        eps_clip: float = 0.2
        dual_clip: float = None
        value_clip: bool = False  # no need
        norm_adv: bool = True  # good for improving training stability
        recompute_adv: bool = False
        # Lagrangian specific arguments
        use_lagrangian: bool = True
        lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1)
        rescaling: bool = True
        # Base policy common arguments
        gamma: float = 0.99
        max_batchsize: int = 100000
        rew_norm: bool = False  # no need, it will slow down training and decrease final perf
        deterministic_eval: bool = True
        action_scaling: bool = True
        action_bound_method: str = "clip"
        # collecting params
        epoch: int = args.epoch
        episode_per_collect: int = args.train_num
        step_per_epoch: int = 3000
        repeat_per_collect: int = 10  # was 4! Increasing this can improve efficiency, but less stability
        buffer_size: int = 200000
        worker: str = "ShmemVectorEnv"
        training_num: int = args.train_num
        testing_num: int = args.test_num
        # general params
        batch_size: int = 256
        reward_threshold: float = 10000  # for early stop purpose
        save_interval: int = 1
        resume: bool = False  # TODO
        save_ckpt: bool = True  # set this to True to save the policy model
        verbose: bool = True
        render: bool = False

        # logger params
        group_name: str = "EXP1_5"
        # run_name= f'PPOL_h20_1powerlimit_sacl_100_v2g_cost_40_loads_PV_no_DR_5spawn_10cs_90kw_seed_{seed}_cost_lim_{int(cost_limit)}_usr_-3_100_tr_30_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        run_name= f'ppol_10repeatpercollect_exp1_1_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'

        wandb.init(project='experiment_1_5',
                        sync_tensorboard=True,
                        group=group_name,
                        name=run_name,
                        save_code=True,
                        mode = "offline",
                        )

        # init logger
        logger = WandbLogger(log_dir="fsrl_logs/EXP1_5/ppol", log_txt=True, group=group_name, name=run_name)

        env = gym.make(task)
        # env.spec.max_episode_steps = env.env.env.simulation_length

        # sim_length = env.env.env.simulation_length

        agent = PPOLagAgent(
        env=env,
        logger=logger,
        device=device,
        thread=thread,
        seed=seed,
        lr=lr,
        hidden_sizes=hidden_sizes,
        unbounded=unbounded,
        last_layer_scale=last_layer_scale,
        target_kl=target_kl,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        gae_lambda=gae_lambda,
        eps_clip=eps_clip,
        dual_clip=dual_clip,
        value_clip=value_clip,
        advantage_normalization=norm_adv,
        recompute_advantage=recompute_adv,
        use_lagrangian=use_lagrangian,
        lagrangian_pid=lagrangian_pid,
        cost_limit=cost_limit,
        rescaling=rescaling,
        gamma=gamma,
        max_batchsize=max_batchsize,
        reward_normalization=rew_norm,
        deterministic_eval=deterministic_eval,
        action_scaling=action_scaling,
        action_bound_method=action_bound_method,
    )

        worker = eval(worker)
        train_envs = worker([lambda: gym.make(task) for _ in range(training_num)])
        test_envs = worker([lambda: gym.make(task) for _ in range(testing_num)])

        # start training
            # start training
        agent.learn(
                train_envs=train_envs,
                test_envs=test_envs,
                epoch=epoch,
                episode_per_collect=episode_per_collect,
                step_per_epoch=step_per_epoch,
                repeat_per_collect=repeat_per_collect,
                buffer_size=buffer_size,
                testing_num=testing_num,
                batch_size=batch_size,
                reward_threshold=reward_threshold,
                save_interval=save_interval,
                resume=resume,
                save_ckpt=save_ckpt,
                verbose=verbose,
        )


def train_sacl(args):
        task: str = "fsrl-v0"
        cost_limit: int = args.cost_limit
        device: str = "cpu"
        thread: int = 1  # if use "cpu" to train
        seed: int = args.seed
        # algorithm params
        actor_lr: float = 5e-4
        critic_lr: float = 1e-3
        hidden_sizes: Tuple[int, ...] = (128, 128)
        auto_alpha: bool = True
        alpha_lr: float = 3e-4
        alpha: float = 0.005
        tau: float = 0.05
        n_step: int = 2
        conditioned_sigma: bool = True
        unbounded: bool = False
        last_layer_scale: bool = False
        # Lagrangian specific arguments
        use_lagrangian: bool = True
        lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1)
        rescaling: bool = True
        # Base policy common arguments
        gamma: float = 0.99
        deterministic_eval: bool = True
        action_scaling: bool = True
        action_bound_method: str = "clip"
        # collecting params
        epoch: int = args.epoch
        episode_per_collect: int = args.train_num
        step_per_epoch: int = 3000
        update_per_step: float = 0.2
        buffer_size: int = 400000
        worker: str = "ShmemVectorEnv"
        training_num: int = args.train_num
        testing_num: int = args.test_num
        # general train params
        batch_size: int = 256
        reward_threshold: float = 10000  # for early stop purpose
        save_interval: int = 1
        resume: bool = False  # TODO
        save_ckpt: bool = True  # set this to True to save the policy model
        verbose: bool = True
        render: bool = False

        # logger params
        group_name: str = "EXP1_3"
        # run_name= f'sacl_v4_h20_no_v2g_cost_5spawn_10cs_90kw_cost_lim_{int(cost_limit)}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        # run_name= f'sacl_exp1_1_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        run_name= f'sacl_step_epoch_3k_30chargers_buff_400k_exp1_3_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        

        wandb.init(project='experiment_1_3',
                        sync_tensorboard=True,
                        group=group_name,
                        name=run_name,
                        save_code=True,
                        )

        # init logger
        logger = WandbLogger(log_dir="fsrl_logs/EXP1_3", log_txt=True, group=group_name, name=run_name)

        env = gym.make(task)
        # env.spec.max_episode_steps = env.env.env.simulation_length

        sim_length = env.env.env.simulation_length

        agent = SACLagAgent(
        env=SpecMaxStepsWrapper(gym.make(task), sim_length),
        logger=logger,
        # general task params
        device=device,
        thread=thread,
        seed=seed,
        # algorithm params
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        hidden_sizes=hidden_sizes,
        auto_alpha=auto_alpha,
        alpha_lr=alpha_lr,
        alpha=alpha,
        tau=tau,
        n_step=n_step,
        # Lagrangian specific arguments
        use_lagrangian=use_lagrangian,
        lagrangian_pid=lagrangian_pid,
        cost_limit=cost_limit,
        rescaling=rescaling,
        # Base policy common arguments
        gamma=gamma,
        conditioned_sigma=conditioned_sigma,
        unbounded=unbounded,
        last_layer_scale=last_layer_scale,
        deterministic_eval=deterministic_eval,
        action_scaling=action_scaling,
        action_bound_method=action_bound_method,
        lr_scheduler=None
    )


        # training_num = min(training_num, episode_per_collect)
        worker = eval(worker)
        train_envs = worker([lambda: SpecMaxStepsWrapper(gym.make(task), sim_length) for _ in range(training_num)])
        test_envs = worker([lambda: SpecMaxStepsWrapper(gym.make(task), sim_length) for _ in range(testing_num)])
        

        agent.learn(
        train_envs=train_envs,
        test_envs=test_envs,
        epoch=epoch,
        episode_per_collect=episode_per_collect,
        step_per_epoch=step_per_epoch,
        update_per_step=update_per_step,
        buffer_size=buffer_size,
        testing_num=testing_num,
        batch_size=batch_size,
        reward_threshold=reward_threshold,  # for early stop purpose
        save_interval=save_interval,
        resume=resume,
        save_ckpt=save_ckpt,
        verbose=verbose,
    )


if __name__ == "__main__":
        #create an argument parser to adjust cost limit and define training algorithm
        parser = argparse.ArgumentParser() 
        parser.add_argument("--train", type=str, default="cvpo", help="Training algorithm to use. Options: 'cpo', 'cvpo', 'ppol', 'sacl'")
        parser.add_argument("--cost_limit", type=int, default=2, help="Cost limit for the environment")
        parser.add_argument("--epoch", type=int, default=1000)
        parser.add_argument("--train_num", type=int, default=10, help="Number of training environments")
        parser.add_argument("--test_num", type=int, default=50, help="Number of testing environments")
        parser.add_argument("--config", type=str, default="V2GProfit_base.yaml", help="Config file")
        parser.add_argument("--seed", type=int, default=10)
        args = parser.parse_args()
        if args.train == "cvpo":        
                train_cvpo(args)
        elif args.train == "cpo":
                train_cpo(args)
        elif args.train == "ppol":
                train_ppol(args)
        elif args.train == "sacl":
                train_sacl(args)
        else:   
                print("Invalid training algorithm. Please choose either 'cpo', 'cvpo', 'ppol' or 'sacl'")