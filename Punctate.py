#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Create a folder to store csv files
import pathlib
pathlib.Path('./Punctate').mkdir()


# In[2]:


# Reward function: Get reward only at the goal state
def R(current_state, end_state, reward_size):
    if current_state == end_state:
        reward = reward_size
    else:
        reward = 0
    return reward

# function for one episode
def punc1(gamma, alpha, state_n, v_state, stay_prob, state_list, action_list, RPE_list, 
          v1_list, step_list, reward_size):
    time_step = 1
    current_state = 0
    timestep_list = []
    not_end = True
    
    while not_end:
        if current_state == state_n:
            not_end = False
            break
        
        else:
            # add V(S1) to the list
            v1_list.append(v_state[0])
            
            # Get reward
            reward = R(current_state, state_n - 1, reward_size)
            
            # Determine the next state
            if current_state == state_n - 1:
                next_state = current_state + 1
                go = 1
            else:
                if rd.random() < stay_prob: # stay
                    next_state = current_state
                    go = 0
                else: # move
                    next_state = current_state + 1
                    go = 1
            
            # calculate RPE and update weights and state values
            if current_state == state_n - 1: # at the goal state
                delta = reward + 0 - v_state[current_state]
            else:
                delta = reward + gamma*v_state[next_state] - v_state[current_state]
            
            # update state value
            v_state[current_state] += alpha * delta
            
            state_list.append(current_state+1)
            if go == 0:
                action_list.append("No-Go")
            else:
                action_list.append("Go")
            RPE_list.append(delta)
            timestep_list.append(time_step)
            step_list.append(time_step)

            # Move to the next state
            current_state = next_state
            
            time_step += 1

    return v_state, state_list, action_list, RPE_list, timestep_list, v1_list, step_list

# function for multi episodes
def punc2(epi_num, gamma, alpha, state_n, v_state, stay_prob, state_list, action_list, 
          RPE_list, v1_list, step_list, reward_size, epi_num_list):
    epi_length = []
    with_punish = False
    
    for k in range(epi_num):
        c_v_state, c_state_list, c_action_list, c_RPE_list, timestep_list, c_v1_list, c_step_list =         punc1(gamma, alpha, state_n, v_state, stay_prob, state_list, action_list, 
              RPE_list, v1_list, step_list, reward_size)
        
        for j in range(len(timestep_list)):
            epi_num_list.append(k+1)
                
        for j in range(len(timestep_list)):
            epi_length.append(k+1)
        
        v_state = c_v_state
        state_list = c_state_list
        action_list = c_action_list
        RPE_list = c_RPE_list
        v1_list = c_v1_list
        step_list = c_step_list
        
    return c_v_state, c_state_list, c_action_list, c_RPE_list, epi_num_list, epi_length, c_v1_list, c_step_list

# function for multi simulations
def punc3(sim_num, epi_num, gamma, alpha, state_n, stay_prob, state_list, action_list, 
          RPE_list, v1_list, step_list, reward_size, epi_num_list, policy):
    sim_num_list = []
    
    for t in range(sim_num):
        v_state = [] # initialize
        if policy == "Resistant":
            for k in range(state_n):
                v_state.append(gamma**(state_n - k - 1)) # v_state = [gamma^n-1, gamma^n-2, ..., gamma, 1]; true value under "Non-resistant" policy
        elif policy == "Non-Resistant":
            for k in range(state_n):
                v_state.append(0.0) # v_state = [0, 0, ..., 0]; initial value
        else:
            print("Please designate a correct policy.")
        
        c_v_state, c_state_list, c_action_list, c_RPE_list, c_epi_num_list, epi_length, c_v1_list, c_step_list =         punc2(epi_num, gamma, alpha, state_n, v_state, stay_prob, state_list, action_list, 
              RPE_list, v1_list, step_list, reward_size, epi_num_list)
        
        for u in range(len(epi_length)):
            sim_num_list.append(t+1)
        
        state_list = c_state_list
        action_list = c_action_list
        RPE_list = c_RPE_list
        v1_list = c_v1_list
        step_list = c_step_list
        epi_num_list = c_epi_num_list
    
    return c_v_state, c_state_list, c_action_list, c_RPE_list, c_epi_num_list, sim_num_list, c_v1_list, c_step_list


# In[3]:


# functions with punishment
# reward function
def R2(current_state, end_state, reward_size, punish_size):
    if current_state == end_state - 1:
        reward = reward_size
    elif current_state == end_state:
        reward = punish_size
    else:
        reward = 0
    return reward

def punc1_2(gamma, alpha, state_n, v_state, stay_prob, state_list, action_list, RPE_list, 
          v1_list, step_list, reward_size, punish_size):
    time_step = 1
    current_state = 0
    timestep_list = []
    not_end = True
    
    while not_end:
        if current_state == state_n:
            not_end = False
            break
        
        else:
            # add V(S1) to the list
            v1_list.append(v_state[0])
            
            # Get reward
            reward = R2(current_state, state_n - 1, reward_size, punish_size)
            
            # Determine the next state
            if current_state == state_n-2 or current_state == state_n-1:
                next_state = current_state + 1
                go = 1
            else:
                if rd.random() < stay_prob: # stay
                    next_state = current_state
                    go = 0
                else: # move
                    next_state = current_state + 1
                    go = 1
            
            # calculate RPE and update weights and state values
            if current_state == state_n - 1: # at the goal state
                delta = reward + 0 - v_state[current_state]
            else:
                delta = reward + gamma*v_state[next_state] - v_state[current_state]
            
            # update state value
            v_state[current_state] += alpha * delta
            
            state_list.append(current_state+1)
            if go == 0:
                action_list.append("No-Go")
            else:
                action_list.append("Go")
            RPE_list.append(delta)
            timestep_list.append(time_step)
            step_list.append(time_step)

            # Move to the next state
            current_state = next_state
            
            time_step += 1

    return v_state, state_list, action_list, RPE_list, timestep_list, v1_list, step_list


# function for multi episodes
def punc2_2(epi_num, gamma, alpha, state_n, v_state, stay_prob, state_list, action_list, 
          RPE_list, v1_list, step_list, reward_size, punish_param, epi_num_list, switch):
    epi_length = []
    
    for k in range(epi_num):
        if k < switch: # without punishment
            punish_size = 0
            
        else: # with punishment
            punish_size = punish_param
            
        c_v_state, c_state_list, c_action_list, c_RPE_list, timestep_list, c_v1_list, c_step_list =         punc1_2(gamma, alpha, state_n, v_state, stay_prob, state_list, action_list, 
                RPE_list, v1_list, step_list, reward_size, punish_size)

        for j in range(len(timestep_list)):
            epi_num_list.append(k+1)

        for j in range(len(timestep_list)):
            epi_length.append(k+1)

        v_state = c_v_state
        state_list = c_state_list
        action_list = c_action_list
        RPE_list = c_RPE_list
        v1_list = c_v1_list
        step_list = c_step_list

    return c_v_state, c_state_list, c_action_list, c_RPE_list, epi_num_list, epi_length, c_v1_list, c_step_list

# function for multi simulations
def punc3_2(sim_num, epi_num, gamma, alpha, state_n, stay_prob, state_list, action_list, 
          RPE_list, v1_list, step_list, reward_size, punish_param, epi_num_list, switch, policy):
    sim_num_list = []
    
    for t in range(sim_num):
        v_state = [] # initialize
        for k in range(state_n - 1):
            v_state.append(gamma**(state_n - k - 2)) # v_state = [gamma^n-1, gamma^n-2, ..., gamma, 1]; true value under "Non-resistant" policy
        v_state.append(0.0)
        
        c_v_state, c_state_list, c_action_list, c_RPE_list, c_epi_num_list, epi_length, c_v1_list, c_step_list =         punc2_2(epi_num, gamma, alpha, state_n, v_state, stay_prob, state_list, action_list, 
              RPE_list, v1_list, step_list, reward_size, punish_param, epi_num_list, switch)
        
        for u in range(len(epi_length)):
            sim_num_list.append(t+1)
        
        state_list = c_state_list
        action_list = c_action_list
        RPE_list = c_RPE_list
        v1_list = c_v1_list
        step_list = c_step_list
        epi_num_list = c_epi_num_list
    
    return c_v_state, c_state_list, c_action_list, c_RPE_list, c_epi_num_list, sim_num_list, c_v1_list, c_step_list


# In[4]:


# Simulation of "Non-Resistant" policy
import numpy as np
import random as rd
import pandas as pd

sim_num = 1
epi_num = 200
gamma = 0.97
alpha = 0.50
state_n = 10
stay_prob = 0.0 # Always "Go"
state_list = []
action_list = []
RPE_list = []
v1_list = []
step_list = []
reward_size = 1
punish_size = 0
epi_num_list = []
policy = "Non-Resistant"

rl = punc3(sim_num, epi_num, gamma, alpha, state_n, stay_prob, state_list, action_list, 
          RPE_list, v1_list, step_list, reward_size, epi_num_list, policy)

result = pd.DataFrame({'Simulation': rl[5], 'Episode': rl[4], 'time_step': rl[7], 'State': rl[1], 'Action': rl[2], 'RPE': rl[3], 'V1': rl[6]})
result.to_csv('./Punctate/NonR_g{:.0f}_s{:.0f}_{:.0f}states.csv'.format(100*gamma, 100*stay_prob, state_n))


# In[5]:


# Simulations with various parameters
import numpy as np
import random as rd
import pandas as pd

seed_list = [22, 76, 50, 57, 30, 55, 33, 54,  0]
index = 0

for gamma in [0.95, 0.97, 0.99]:
    for stay_prob in [0.50, 0.75, 0.90]:
        
        rd.seed(seed_list[index])
        
        # set constant variables
        sim_num = 100
        epi_num = 200
        alpha = 0.50
        state_n = 10
        state_list = []
        action_list = []
        RPE_list = []
        v1_list = []
        step_list = []
        reward_size = 1
        punish_size = 0
        epi_num_list = []
        policy = "Resistant"
        
        # simulation
        rl = punc3(sim_num, epi_num, gamma, alpha, state_n, stay_prob, state_list,
                  action_list, RPE_list, v1_list, step_list, reward_size, epi_num_list, policy)
        
        # create dataframe and convert it to csv
        result = pd.DataFrame({'Simulation': rl[5], 'Episode': rl[4], 'time_step': rl[7], 'State': rl[1],
                              'Action': rl[2], 'RPE': rl[3], 'V1': rl[6]})
        result.to_csv('./Punctate/g{:.0f}_s{:.0f}_{:.0f}states_r{}.csv'.format(100*gamma, 100*stay_prob, state_n, reward_size))
        index += 1


# In[6]:


# Simulation under Non-Resistant with punishment
import numpy as np
import random as rd
import pandas as pd

rd.seed(20210203)

gamma = 0.97
stay_prob = 0.0 # Non-Resistant
sim_num = 1
epi_num = 200
alpha= 0.50
state_n = 11
state_list = []
action_list = []
RPE_list = []
v1_list = []
step_list = []
reward_size = 1.0
punish_param = -2.0
epi_num_list = []
switch = 0 # the agent will be given punishment from the first episode
policy = "Non-Resistant"

rl = punc3_2(sim_num, epi_num, gamma, alpha, state_n, stay_prob, state_list, action_list, RPE_list, 
             v1_list, step_list, reward_size, punish_param, epi_num_list, switch, policy)

# create dataframe and convert it to csv
result = pd.DataFrame({'Simulation': rl[5], 'Episode': rl[4], 'time_step': rl[7], 'State': rl[1], 'Action': rl[2], 'RPE': rl[3], 
                       'V1': rl[6]})
result.to_csv('./Punctate/NonR_punish_g{:.0f}_s{:.0f}_{:.0f}states_r{}_p{}.csv'.format(100*gamma, 100*stay_prob, state_n, reward_size, -punish_param))


# In[ ]:




