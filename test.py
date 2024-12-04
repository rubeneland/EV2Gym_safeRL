import torch
checkpoint = torch.load('./logs/min_c_2_usr_1000_train_evs_50/checkpoint/model.pt')
print(checkpoint.keys())  # Check what keys exist

state_dict = torch.load('path_to_model.pt')
if 'model' in state_dict:
    state_dict = state_dict['model']  # Adjust based on actual key structure
policy.load_state_dict(state_dict)