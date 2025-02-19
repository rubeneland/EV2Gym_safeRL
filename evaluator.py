# This script reads the replay files and evaluates the performance.

import yaml
import os
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

from ev2gym.utilities.arg_parser import arg_parser
from ev2gym.models import ev2gym_env

from ev2gym.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity, RoundRobin_1transformer_powerlimit

from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle, V2GProfitMaxLoadsOracle

from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO

from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin
from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB

from ev2gym.rl_agent.reward import SquaredTrackingErrorReward
from ev2gym.rl_agent.reward import profit_maximization, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

from ev2gym.visuals.evaluator_plot import plot_total_power, plot_comparable_EV_SoC
from ev2gym.visuals.evaluator_plot import plot_total_power_V2G, plot_actual_power_vs_setpoint
from ev2gym.visuals.evaluator_plot import plot_comparable_EV_SoC_single, plot_prices

from cost_functions import transformer_overload_usrpenalty_cost, ProfitMax_TrPenalty_UserIncentives_safety

from fsrl.agent import CPOAgent as CPO
from fsrl.agent import CVPOAgent as CVPO
from fsrl.agent import SACLagAgent as SACLag
from fsrl.utils import TensorboardLogger

import gymnasium as gym
import torch

from gymnasium import Wrapper


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

# class CPO():
#     '''
#     This class contains the CPO algorithm
#     '''
#     algo_name = "CPO"


def evaluator():

    # config_file = "ev2gym/example_config_files/V2G_MPC2.yaml"
    config_file = "V2GProfit_evaluate.yaml"
    # config_file = "ev2gym/example_config_files/PublicPST.yaml"
    # config_file = "ev2gym/example_config_files/BusinessPST.yaml"
    # config_file = "ev2gym/example_config_files/V2GProfitPlusLoads.yaml"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    timescale = config["timescale"]
    simulation_length = config["simulation_length"]

    n_test_cycles = 100

    scenario = config_file.split("/")[-1].split(".")[0]
    # eval_replay_path = f'./replay/{number_of_charging_stations}cs_{n_transformers}tr_{scenario}/'
    eval_replay_path = f'./replay/exp1/'
    print(f'Looking for replay files in {eval_replay_path}')
    try:
        eval_replay_files = [f for f in os.listdir(
            eval_replay_path) if os.path.isfile(os.path.join(eval_replay_path, f))]

        print(
            f'Found {len(eval_replay_files)} replay files in {eval_replay_path}')
        if n_test_cycles > len(eval_replay_files):
            n_test_cycles = len(eval_replay_files)

        replay_to_print = 1
        replay_to_print = min(replay_to_print, len(eval_replay_files)-1)
        replays_exist = True

    except:
        n_test_cycles = n_test_cycles
        replays_exist = False

    print(f'Number of test cycles: {n_test_cycles}')

    if config_file == "ev2gym/example_config_files/V2GProfitMax.yaml":
        reward_function = profit_maximization
        state_function = V2G_profit_max

    elif config_file == "ev2gym/example_config_files/PublicPST.yaml":
        reward_function = SquaredTrackingErrorReward
        state_function = PublicPST

    elif config_file == "ev2gym/example_config_files/V2G_MPC.yaml":
        reward_function = profit_maximization
        state_function = V2G_profit_max

    elif config_file == "ev2gym/example_config_files/V2GProfitPlusLoads.yaml":
        reward_function = ProfitMax_TrPenalty_UserIncentives
        state_function = V2G_profit_max_loads

    elif config_file == "V2GProfit_base.yaml":
        reward_function = ProfitMax_TrPenalty_UserIncentives_safety
        state_function = V2G_profit_max_loads
        cost_function = transformer_overload_usrpenalty_cost
    elif config_file == "V2GProfit_evaluate.yaml":
        reward_function = ProfitMax_TrPenalty_UserIncentives_safety
        state_function = V2G_profit_max
        cost_function = transformer_overload_usrpenalty_cost
    else:
        raise ValueError('Unknown config file')

    def generate_replay(evaluation_name):
        # replay_path = evaluation_name.split('/')[-1]
        env = ev2gym_env.EV2Gym(
            config_file=config_file,
            generate_rnd_game=True,
            save_replay=True,
            replay_save_path=f"{evaluation_name}/",
        )
        replay_path = evaluation_name.split('/')[-1]
        replay_path = f"./replay{replay_path}/replay_{env.sim_name}.pkl"
        # print(env.sim_starting_date)

        for _ in range(env.simulation_length):
            actions = np.ones(env.cs)

            new_state, reward, done, truncated, _ = env.step(
                actions, visualize=False)  # takes action

            if done:
                break

        return replay_path

    # Algorithms to compare:

    algorithms = [
        ChargeAsFastAsPossible,
        # ChargeAsLateAsPossible,
        # PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO,
        # SAC,
        # TQC,
        # TD3,
        # # ARS,
        # # RecurrentPPO,
        RoundRobin_1transformer_powerlimit,
        # eMPC_V2G,
        # # V2GProfitMaxLoadsOracle,
        V2GProfitMaxOracleGB,
        # V2GProfitMaxOracle,
        # PowerTrackingErrorrMin
        # CPO,
        CVPO
        # SACLag
    ]

    # algorithms = [
    #     # ChargeAsFastAsPossibleToDesiredCapacity,
    #             'OCMF_V2G_10',
    #             # 'OCMF_V2G_20',
    #             'OCMF_V2G_30',
    #             'OCMF_G2V_10',
    #             # # 'OCMF_G2V_20',
    #             'OCMF_G2V_30',
    #             'eMPC_V2G_10',
    #             # # 'eMPC_V2G_20',
    #             'eMPC_V2G_30',
    #             'eMPC_G2V_10',
    #             'eMPC_G2V_30',

    #             #   eMPC_V2G,
    #             #   eMPC_G2V,
    #             ]

    # evaluation_name = f'eval_{number_of_charging_stations}cs_{n_transformers}tr_{scenario}_{len(algorithms)}_algos' +\
    #     f'_{n_test_cycles}_exp_' +\
        # f'{datetime.datetime.now().strftime("%Y_%m_%d_%f")}'
    evaluation_name = f'exp1' + f'{datetime.datetime.now().strftime("%Y_%m_%d_%f")}'

    # make a directory for the evaluation
    save_path = f'./results/{evaluation_name}/'
    os.makedirs(save_path, exist_ok=True)
    os.system(f'cp {config_file} {save_path}')

    if not replays_exist:
        eval_replay_files = [generate_replay(
            eval_replay_path) for _ in range(n_test_cycles)]

    plot_results_dict = {}
    counter = 0
    for algorithm in algorithms:

        print(' +------- Evaluating', algorithm, " -------+")
        for k in range(n_test_cycles):
            print(f' Test cycle {k+1}/{n_test_cycles} -- {algorithm}')
            counter += 1
            h = -1

            # if replays_exist:
            #     replay_path = eval_replay_path + eval_replay_files[k]
            # else:
            #     replay_path = eval_replay_files[k]

            # replay_path = f'./replay/{number_of_charging_stations}cs_{n_transformers}tr_{scenario}/' + eval_replay_files[k].split('/')[-1]
            replay_path = f'./replay/exp1/' + eval_replay_files[k].split('/')[-1]

            if algorithm in [PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO]:
                gym.envs.register(id='evs-v0', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                                  kwargs={'config_file': config_file,
                                          'generate_rnd_game': True,
                                          'state_function': state_function,
                                          'reward_function': reward_function,
                                          'load_from_replay_path': replay_path,
                                          })
                env = gym.make('evs-v0')

                if algorithm == RecurrentPPO:
                    load_path = f'./saved_models/{number_of_charging_stations}cs_{scenario}/' + \
                        f"rppo_{reward_function.__name__}_{state_function.__name__}"

                else:
                    load_path = f'./saved_models/{number_of_charging_stations}cs_{scenario}/' + \
                        f"{algorithm.__name__.lower()}_{reward_function.__name__}_{state_function.__name__}/best_model.zip"

                # initialize the timer
                timer = time.time()

                model = algorithm.load(load_path, env, device=device)
                env = model.get_env()
                state = env.reset()

            elif algorithm == CPO:
                gym.envs.register(id='eval', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                                  kwargs={'config_file': config_file,
                                          'verbose': False,
                                          'save_plots': False,
                                          'generate_rnd_game': True,
                                          'reward_function': reward_function,
                                          'state_function': state_function,
                                          'cost_function': cost_function,
                                          })

                task = "eval"

                load_path = './fsrl_logs/5cs_30kw_7spawn/50_test_envs_no_loads_no_PV_no_DR_CVPO_5spawn_10cs_120kw_cost_lim_90_usr_-4_100_NO_tr_train_envs_4_test_envs_50_run422/checkpoint/model_best.pt'
                # init logger
                logger = TensorboardLogger("logs", log_txt=True, name=task)
                agent = CPO(env, logger)
                # policy = CPO_policy
                model = agent.policy
                state_dict = (torch.load(load_path))
                model.load_state_dict(state_dict['model'])
                model = model.actor
                model.eval()

                env = ev2gym_env.EV2Gym(
                    config_file=config_file,
                    load_from_replay_path=replay_path,
                    generate_rnd_game=True,
                    state_function=state_function,
                    reward_function=reward_function,
                )

                # initialize the timer
                timer = time.time()
                state, _ = env.reset()

            elif algorithm == CVPO:
                gym.envs.register(id='eval', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                                  kwargs={'config_file': config_file,
                                          'verbose': False,
                                          'save_plots': False,
                                          'generate_rnd_game': True,
                                          'reward_function': reward_function,
                                          'state_function': state_function,
                                          'cost_function': cost_function,
                                          })

                task = "eval"

                env = gym.make(task)
                sim_length = env.env.env.simulation_length

                load_path = 'fsrl_logs/exp_1_no_loads_no_pv_10_cs_spawn_5/cvpo_v19/checkpoint/model_best.pt'

                # init logger
                logger = TensorboardLogger("logs", log_txt=True, name=task)
                agent = CVPO(SpecMaxStepsWrapper(env, sim_length), logger)
                model = agent.policy
                state_dict = (torch.load(load_path))
                model.load_state_dict(state_dict['model'])
                model = model.actor
                model.eval()

                env = ev2gym_env.EV2Gym(
                    config_file=config_file,
                    load_from_replay_path=replay_path,
                    generate_rnd_game=True,
                    state_function=state_function,
                    reward_function=reward_function,
                )

                # initialize the timer
                timer = time.time()
                state, _ = env.reset()

            elif algorithm == SACLag:
                gym.envs.register(id='eval', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                                  kwargs={'config_file': config_file,
                                          'verbose': False,
                                          'save_plots': False,
                                          'generate_rnd_game': True,
                                          'reward_function': reward_function,
                                          'state_function': state_function,
                                          'cost_function': cost_function,
                                          })

                task = "eval"

                env = gym.make(task)
                sim_length = env.env.env.simulation_length

                load_path = 'fsrl_logs/exp_1_no_loads_no_pv_10_cs_spawn_5/sacl_v2/checkpoint/model_best.pt'

                # init logger
                logger = TensorboardLogger("logs", log_txt=True, name=task)
                agent = SACLag(SpecMaxStepsWrapper(env, sim_length), logger)
                model = agent.policy
                state_dict = (torch.load(load_path))
                model.load_state_dict(state_dict['model'])
                model = model.actor
                model.eval()

                env = ev2gym_env.EV2Gym(
                    config_file=config_file,
                    load_from_replay_path=replay_path,
                    generate_rnd_game=True,
                    state_function=state_function,
                    reward_function=reward_function,
                )

                # initialize the timer
                timer = time.time()
                state, _ = env.reset()

            else:
                env = ev2gym_env.EV2Gym(
                    config_file=config_file,
                    load_from_replay_path=replay_path,
                    generate_rnd_game=True,
                    state_function=state_function,
                    reward_function=reward_function,
                )

                # initialize the timer
                timer = time.time()
                state = env.reset()
                try:
                    if type(algorithm) == str:
                        if algorithm.split('_')[0] in ['OCMF', 'eMPC']:
                            h = int(algorithm.split('_')[2])
                            algorithm = algorithm.split(
                                '_')[0] + '_' + algorithm.split('_')[1]
                            print(
                                f'Algorithm: {algorithm} with control horizon {h}')
                            if algorithm == 'OCMF_V2G':
                                model = OCMF_V2G(env=env, control_horizon=h)
                                algorithm = OCMF_V2G
                            elif algorithm == 'OCMF_G2V':
                                model = OCMF_G2V(env=env, control_horizon=h)
                                algorithm = OCMF_G2V
                            elif algorithm == 'eMPC_V2G':
                                model = eMPC_V2G(env=env, control_horizon=h)
                                algorithm = eMPC_V2G
                            elif algorithm == 'eMPC_G2V':
                                model = eMPC_G2V(env=env, control_horizon=h)
                                algorithm = eMPC_G2V

                    else:
                        model = algorithm(env=env,
                                          replay_path=replay_path,
                                          verbose=False)
                except Exception as error:
                    print(error)
                    print(
                        f'Error in {algorithm} with replay {replay_path}')
                    continue

            rewards = []

            for i in range(simulation_length):
                print(f' Step {i+1}/{simulation_length} -- {algorithm}')
                ################# Evaluation ##############################
                if algorithm in [PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO]:
                    action, _ = model.predict(state, deterministic=True)
                    obs, reward, done, _, stats = env.step(action)
                    # if CPO == True:
                    #     action = policy.predict(state, deterministic=True)
                    #     obs, reward, done, stats = env.step(action)
                    if i == simulation_length - 2:
                        saved_env = deepcopy(env.get_attr('env')[0])

                    stats = stats[0]
                    print(stats)
                elif algorithm == CPO or algorithm == CVPO or algorithm == SACLag:

                    # reshape the state to 1 x n_features
                    state = state.reshape(1, -1)

                    action = model.forward(state)
                    action = action[0][0][0].detach().numpy()
                    state, reward, done, _, stats = env.step(action)

                else:
                    actions = model.get_action(env=env)
                    new_state, reward, done, _, stats = env.step(
                        actions, visualize=False)  # takes action
                ############################################################

                rewards.append(reward)

                if done:
                    results_i = pd.DataFrame({'run': k,
                                              'Algorithm': algorithm.__name__,
                                              'control_horizon': h,
                                              'discharge_price_factor': config['discharge_price_factor'],
                                              'total_ev_served': stats['total_ev_served'],
                                              'total_profits': stats['total_profits'],
                                              'total_energy_charged': stats['total_energy_charged'],
                                              'total_energy_discharged': stats['total_energy_discharged'],
                                              'average_user_satisfaction': stats['average_user_satisfaction'],
                                              'power_tracker_violation': stats['power_tracker_violation'],
                                              'tracking_error': stats['tracking_error'],
                                              'min_energy_user_satisfaction': stats['min_energy_user_satisfaction'],
                                              'total_steps_min_v2g_soc_violation': stats['total_steps_min_v2g_soc_violation'],
                                              'energy_tracking_error': stats['energy_tracking_error'],
                                              'energy_user_satisfaction': stats['energy_user_satisfaction'],
                                              'total_transformer_overload': stats['total_transformer_overload'],
                                              'battery_degradation': stats['battery_degradation'],
                                              'battery_degradation_calendar': stats['battery_degradation_calendar'],
                                              'battery_degradation_cycling': stats['battery_degradation_cycling'],
                                              'total_reward': sum(rewards),
                                              'time': time.time() - timer,
                                              # 'time_gb': model.total_exec_time,
                                              }, index=[counter])

                    if counter == 1:
                        results = results_i
                    else:
                        results = pd.concat([results, results_i])

                    if algorithm in [PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO]:
                        env = saved_env

                    plot_results_dict[algorithm.__name__] = deepcopy(env)

                    break

    # save the plot_results_dict to a pickle file
    with open(save_path + 'plot_results_dict.pkl', 'wb') as f:
        pickle.dump(plot_results_dict, f)

    # save the results to a csv file
    results.to_csv(save_path + 'data.csv')

    # Group the results by algorithm and print the average and the standard deviation of the statistics
    results_grouped = results.groupby('Algorithm').agg(['mean', 'std'])

    # savethe latex results in a txt file
    with open(save_path + 'results_grouped.txt', 'w') as f:
        f.write(results_grouped.to_latex())

    # results_grouped.to_csv('results_grouped.csv')
    # print(results_grouped[['tracking_error', 'energy_tracking_error']])
    print(results_grouped[['total_transformer_overload',
          'time', 'total_steps_min_v2g_soc_violation']])
    print(results_grouped[['total_reward',
          'total_profits', 'total_ev_served']])
    print(results_grouped[['total_energy_charged', 'total_energy_discharged']])
    print(results_grouped[['average_user_satisfaction',
          'min_energy_user_satisfaction']])
    # input('Press Enter to continue')

    algorithm_names = []
    for algorithm in algorithms:
        # if class has attribute .name, use it
        if hasattr(algorithm, 'algo_name'):
            algorithm_names.append(algorithm.algo_name)
        else:
            algorithm_names.append(algorithm.__name__)

    # plot_total_power(results_path=save_path + 'plot_results_dict.pkl',
    #                 save_path=save_path,
    #                 algorithm_names=algorithm_names)

    plot_comparable_EV_SoC(results_path=save_path + 'plot_results_dict.pkl',
                           save_path=save_path,
                           algorithm_names=algorithm_names)

    # plot_actual_power_vs_setpoint(results_path=save_path + 'plot_results_dict.pkl',
    #                             save_path=save_path,
    #                             algorithm_names=algorithm_names)

    plot_total_power_V2G(results_path=save_path + 'plot_results_dict.pkl',
                         save_path=save_path,
                         algorithm_names=algorithm_names)

    plot_comparable_EV_SoC_single(results_path=save_path + 'plot_results_dict.pkl',
                                  save_path=save_path,
                                  algorithm_names=algorithm_names)

    plot_prices(results_path=save_path + 'plot_results_dict.pkl',
                save_path=save_path,
                algorithm_names=algorithm_names)


if __name__ == "__main__":
    evaluator()
