"""
This script is used to evaluate the performance of the ev2gym environment.
"""
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin
from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V

from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2, eMPC_G2V_v2

from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle

from ev2gym.baselines.heuristics import RoundRobin, RoundRobin_1transformer_powerlimit, ChargeAsFastAsPossible
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity

from cost_functions import transformer_overload_usrpenalty_cost, ProfitMax_TrPenalty_UserIncentives_safety

import numpy as np
import matplotlib.pyplot as plt
import pkg_resources
import gymnasium as gym


def eval():
    """
    Runs an evaluation of the ev2gym environment.
    """

    verbose = True

    replay_path = "./replay/replay_sim_2024_07_05_106720.pkl"
    replay_path = None

    # config_file = "ev2gym/example_config_files/V2G_MPC2.yaml"
    config_file = "V2GProfit_eval_base.yaml"

    env = EV2Gym(config_file=config_file,
                 load_from_replay_path=replay_path,
                 verbose=False,
                 #  seed=184692,
                 reward_function=ProfitMax_TrPenalty_UserIncentives_safety,
                 cost_function=transformer_overload_usrpenalty_cost,
                 save_replay=True,
                 save_plots=True,
                 )

    new_replay_path = f"replay/replay_{env.sim_name}.pkl"

    state, _ = env.reset()

    ev_profiles = env.EVs_profiles
    max_time_of_stay = max([ev.time_of_departure - ev.time_of_arrival
                            for ev in ev_profiles])
    min_time_of_stay = min([ev.time_of_departure - ev.time_of_arrival
                            for ev in ev_profiles])

    print(f'Number of EVs: {len(ev_profiles)}')
    print(f'Max time of stay: {max_time_of_stay}')
    print(f'Min time of stay: {min_time_of_stay}')

    # exit()
    # agent = OCMF_V2G(env, control_horizon=30, verbose=True)
    # agent = OCMF_G2V(env, control_horizon=25, verbose=True)
    # agent = eMPC_V2G(env, control_horizon=15, verbose=False)
    # agent = PowerTrackingErrorrMin(new_replay_path)
    # agent = eMPC_G2V(env, control_horizon=15, verbose=False)
    # agent = eMPC_V2G_v2(env, control_horizon=10, verbose=False)
    # agent = V2GProfitMaxOracleGB(replay_path=new_replay_path)
    agent = ChargeAsFastAsPossible(verbose=False)
    # agent = ChargeAsFastAsPossibleToDesiredCapacity()
    rewards = []
    print(f'date: {env.sim_date}')
    for t in range(env.simulation_length):
        actions = agent.get_action(env)

        new_state, reward, done, truncated, stats = env.step(
            actions, visualize=False)  # takes action
        rewards.append(reward)

        # print(stats['cost'])

        if done:
            print(stats)
            print(f'total_profits: {stats["total_profits"]}')
            print(f'average_user_satisfaction: {stats["average_user_satisfaction"]}')
            print(f'End of simulation at step {env.current_step}')
            break

    # exit()
    # Solve optimally
    agent = V2GProfitMaxOracleGB(replay_path=new_replay_path)

    env = EV2Gym(config_file=config_file,
                 load_from_replay_path=new_replay_path,
                 verbose=False,
                 save_plots=True,
                 )
    state, _ = env.reset()
    rewards_opt = []

    for t in range(env.simulation_length):
        actions = agent.get_action(env)
        # if verbose:
        #     print(f' OptimalActions: {actions}')

        new_state, reward, done, truncated, stats = env.step(
            actions, visualize=False)  # takes action
        rewards_opt.append(reward)

        if done:
            # print(stats)
            print(f'total_profits: {stats["total_profits"]}')
            print(f'average_user_satisfaction: {stats["average_user_satisfaction"]}')
            print(f'Reward: {reward} \t Done: {done}')

        if done:
            break


if __name__ == "__main__":
    counter = 0
    while True:
        print(f'============================= Counter: {counter}')
        eval()
        counter += 1
        exit()
