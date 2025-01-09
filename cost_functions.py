'''
This file contains the cost  and reward functions for the EV2Gym safety environment.
'''
import math



def transformer_overload_usrpenalty_cost(env, total_costs, user_satisfaction_list, target_SOC, *args):
    """
    Returns the cost of a transformer overload.
    """
    cost = 0
    # for tr in env.transformers:
    #     cost += 50 * tr.get_how_overloaded()                        
    
    for score in user_satisfaction_list:  
        if score > 0.6:
            cost += 100*math.exp(-3*score) - 100*math.exp(-3)
        else:
            cost += 2*100*math.exp(-3*score) - 2*100*math.exp(-3)

    # # For every charging station connected to the transformer
    # for cs in env.charging_stations:

    #         # For every EV connected to the charging station
    #         for EV in cs.evs_connected:
    #             # If there is an EV connected
    #             if EV is not None:
    #                 # If the EV is charging and the SOC is at maximum
    #                 if EV.current_capacity >= EV.desired_capacity and EV.actual_current > 0:
    #                     cost += 4
    #                 # If the EV is discharged and the SOC is at minimum
    #                 if EV.current_capacity <= EV.min_battery_capacity and EV.actual_current < 0:
    #                     cost += 4


    # cost for charging action when SOC at maximum
    # for ev in self.EVs:
    #     if ev.get_SOC() >= target_SOC and ev.get_action() > 0:
    #         cost += 20

    return cost



def ProfitMax_TrPenalty_UserIncentives_safety(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs

    # for tr in env.transformers:
    #     reward -= 100 * tr.get_how_overloaded()                        
    
    # for score in user_satisfaction_list:        
    #     reward -= 100 * math.exp(-10*score)        
        
    return reward