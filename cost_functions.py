'''
This file contains the cost and reward functions for the EV2Gym safety environment.
'''
import math

def usrpenalty_cost(env, total_costs, user_satisfaction_list, *args):
    """
    Returns the cost of user satisfaction penalty.
    """

    cost = 0
    
    a = 20
    b = -3      
    
    for score in user_satisfaction_list:  
        cost += a*math.exp(b*score) - a*math.exp(b)

    return cost

def V2G_profitmaxV2(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs
    
    verbose = False
    
    if verbose:
        print(f'!!! Costs: {total_costs}')
    
    user_costs = 0
    
    linear = False
    if linear:
        cost_multiplier = 0.1
    else:
        cost_multiplier = 0.05
    
    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is not None:
                min_steps_to_full = (ev.desired_capacity - ev.current_capacity) / \
                    (ev.max_ac_charge_power/(60/env.timescale))
                
                
                departing_step = ev.time_of_departure - env.current_step
                
                cost = 0
                if min_steps_to_full > departing_step:                    
                    min_capacity_at_time = ev.desired_capacity - ((departing_step+1) * ev.max_ac_charge_power/(60/env.timescale))
                    
                    if linear:
                        cost = cost_multiplier*(min_capacity_at_time - ev.current_capacity)
                    else:
                        cost = cost_multiplier*(min_capacity_at_time - ev.current_capacity)**2
                        
                    if verbose:
                        print(f'min_capacity_at_time: {min_capacity_at_time} | {ev.current_capacity} | {ev.desired_capacity} | {min_steps_to_full:.3f} | {departing_step} | cost {(cost):.3f}') 
                    user_costs += - cost
                
                if verbose:
                    print(f'- EV: {ev.current_capacity} | {ev.desired_capacity} | {min_steps_to_full:.3f} | {departing_step} | cost {(cost):.3f}')
                
    for ev in env.departing_evs:
        if ev.desired_capacity > ev.current_capacity:            
            if verbose:
                print(f'!!! EV: {ev.current_capacity} | {ev.desired_capacity} | costs: {-cost_multiplier*(ev.desired_capacity - ev.current_capacity)**2}')
                
            if linear:
                user_costs += -cost_multiplier * (ev.desired_capacity - ev.current_capacity)
            else:
                user_costs += -cost_multiplier * (ev.desired_capacity - ev.current_capacity)**2
            
    if verbose:
        print(f'!!! User Satisfaction Penalty: {user_costs}')

    return (reward + user_costs)


def tr_overload_usrpenalty_cost(env, total_costs, user_satisfaction_list, *args):
    """
    Returns the cost of transformer overload and user satisfaction penalty.
    """

    cost = 0

    for tr in env.transformers:
        cost += 5 * tr.get_how_overloaded()   

    a = 20
    b = -3   
    
    for score in user_satisfaction_list:  
        cost += a*math.exp(b*score) - a*math.exp(b)
        

    # invalid_actions = 0
    # for cs in env.charging_stations:
    #     invalid_actions += cs.invalid_action_punishment
    # cost += invalid_actions * 0.002 # 0.01 


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


    return cost



def ProfitMax_TrPenalty_UserIncentives_safety(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs     


        
    return reward