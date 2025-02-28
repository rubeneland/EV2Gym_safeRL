'''
This file contains the cost and reward functions for the EV2Gym safety environment.
'''
import math



def transformer_overload_usrpenalty_cost(env, total_costs, user_satisfaction_list, target_SOC, *args):
    """
    Returns the cost of a transformer overload.
    """

    cost = 0

    # for tr in env.transformers:
    #     cost += 0.5 * tr.get_how_overloaded()                  
    
    for score in user_satisfaction_list:  
        cost += 50*math.exp(-3*score) - 50*math.exp(-3)

    # invalid_actions = 0
    # for cs in env.charging_stations:
    #     invalid_actions += cs.invalid_action_punishment
    # cost += invalid_actions * 0.01

        # if score < 0.6:
        #      cost += 45

    # # For every charging station connected to the transformer
    # for cs in env.charging_stations:
    #         # For every EV connected to the charging station
    #         for EV in cs.evs_connected:
    #             # If there is an EV connected
    #             if EV is not None:
    #                 # Add cost for discharging action when SOC below min V2G threshold: 40%
    #                 soc = EV.get_soc()
    #                 if soc <= 0.4 and EV.actual_current < 0:
    #                     cost += (0.4 - soc) * 5


    # cost for charging action when SOC at maximum
    # for ev in self.EVs:
    #     if ev.get_SOC() >= target_SOC and ev.get_action() > 0:
    #         cost += 20

    return cost



def ProfitMax_TrPenalty_UserIncentives_safety(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs     

    # for cs in env.charging_stations:

    #         # For every EV connected to the charging station
    #         for EV in cs.evs_connected:
    #             # If there is an EV connected
    #             if EV is not None:
    #                 # Add reward for discharging action when EV = full and still a long time until departure to stimulate V2G
    #                 soc = EV.get_soc()
    #                 if soc > 0.45 and EV.actual_current < 0 and (EV.time_of_departure - env.current_step) > 24:
    #                     reward += (soc - 0.45) * EV.actual_current * 0.05
        
    return reward