## xplane environment for training a Deep Q Learning based RL agent
## Q-Learning Object
## Deep Q Learn AP (Auto Pilot)
## xplane environment for training the RL agent
## xPLANE and Reinforcement Learning - AI/RL based AutoPilot
## Deep Q AP
## Book -> Getting Started with Deep Learning: Programming and Methodologies using Python
## By Ricardo Calix
## www.rcalix.com
## Copyright (c) 2020, Ricardo A. Calix, Ph.D.

###############################################################################

import numpy as np
import time
import os

##############################################################################

import xplane_sim as sim

################################################################################

from QLearn import QLearn
from XPlaneEnv import XPlaneEnv

from DeepQLearn import DeepQLearn

#############################################################################
# y or gamma should be important if care about the future
# epsilon small and preferably decay
## n_states = 9x9x9 = 729   ## find index in cube encoding approach, not binary approach
## 3 vectors of 9 each for pitch, roll, jaw. Equivalent to reading from instruments (six pack)

gb_n_states      =  8   ## 729
gb_n_actions     =  6
gb_gamma         =  0.95
gb_learning_rate =  0.01
gb_epsilon       =  0.15
gb_q_lr          =  1.0 # 1.0 or 0.01 or 0.10
gb_trainBatchSize=  1
gb_memBatchSize  =  1024
gb_memorySize    =  4096



#############################################################################
## initialize agent


#            (n_states, n_actions,     gamma,      q_lr,     epsilon)
#Q = QLearn(gb_n_states, gb_n_actions, gb_gamma,  gb_q_lr,  gb_epsilon ) 

##                 (n_stat,    n_acts,     gamma,    learningRate,     q_lr,    trainBatchSize,      memBatchSize,   memorySize)  
Q = DeepQLearn(gb_n_states, gb_n_actions, gb_gamma, gb_learning_rate, gb_q_lr, gb_trainBatchSize, gb_memBatchSize, gb_memorySize)


#################################################################################
# [lat, long, elev, pitch, roll, true heading/yaw , gear] ##-998 -> NO CHANGE
##  self.starting_position = orig
##  self.destination_position = dest
##  self.actions_binary_n = acts_bin   ## binary possibilities
##  self.end_game_threshold = end_param  ## end_game_distance_threshold

flight_origin = [37.524, -122.06899,  6000, -998, -998, -998, 1] # Palo Alto
flight_destinaion = [37.505, -121.843611, 6000, -998, -998, -998, 1] # Sunol Valley

## (states, orig, dest, actions_bin_n, end_param) ## end_param = 50 feet
env = XPlaneEnv(gb_n_states, flight_origin, flight_destinaion, 6, 50.0)

##########################################################################################

reward_pi_up = 0.0
reward_pi_do = 0.0
reward_ro_ri = 0.0
reward_ro_le = 0.0
reward_ru_ri = 0.0
reward_ru_le = 0.0

def print_summary_results(episode, j, reward, kms_to_go, penalties, state, actions_binary, observation, disp_ctrl, stage, actions_q_vector, env):

    global reward_pi_up
    global reward_pi_do
    global reward_ro_ri
    global reward_ro_le
    global reward_ru_ri
    global reward_ru_le
  
    os.system('clear')

    print(stage)
    
    print("Game # {} and move is {}".format(episode, j)   )
    print("the distance is (kms) ", kms_to_go)
    print("penalties ", penalties)
    print("state= ", state)
    print("                 [p+ p- r+ r- ]")
    print("actions_binary = ", actions_binary)
    print("lat ", observation[0])
    print("long ", observation[1])
    print("altitude ", observation[2])
    
    action_index = np.argmax(actions_binary)
    
    print('**********************')
    print("pitch posi ", observation[3])
    print("pitch ctrl ", disp_ctrl[0])
    if (action_index == 0):
        reward_pi_up = reward
    print("pi_up ", reward_pi_up)
    if (action_index == 1):
          reward_pi_do = reward
    print("pi_do ", reward_pi_do)
    
    print('**********************')
    print("roll posi ", observation[4])
    print("roll ctrl ", disp_ctrl[1])
    if (action_index == 2):
        reward_ro_ri = reward
    print("ro_ri ", reward_ro_ri)
    if (action_index == 3):
        reward_ro_le = reward
    print("ro_le ", reward_ro_le)
    
    print('**********************')
    print("yaw posi ", env.convert_rangeA_to_rangeB(observation[5]))
    print("yaw ctrl ", disp_ctrl[2])
    if (action_index == 4):
        reward_ru_ri = reward
    print("ru_ri ", reward_ru_ri)
    if (action_index == 5):
        reward_ru_le = reward
    print("ru_le ", reward_ru_le)

    print('**********************')
    print("actions_q_vector")
    print(actions_q_vector[0, 0])
    print(actions_q_vector[0, 1])
    print(actions_q_vector[0, 2])
    print(actions_q_vector[0, 3])
    print(actions_q_vector[0, 4])
    print(actions_q_vector[0, 5])

##########################################################################################

def test1(posi):
    sim.send_posi(posi)
    print(   sim.get_posi()   )
    
    for i in range(2000):
        try:
            ctrl = [-1.0, -998, -998, -998, 0, 0] # [pitch, roll, rudder, throttle, gear, flaps]
            sim.send_ctrl(ctrl)
            print(   sim.get_posi()   )
            print(   sim.get_ctrl()    )
        except:
            print("error - probably with xplane comm")
            continue
            
##############################################################################

#test1(starting_position) ## fly at random

##############################################################################

stage = "training ..."

n_epochs = 200
n_moves  = 200

#############################################################################

##observation = sim.reset(env.starting_position)  ## debug only
##observation = sim.get_posi()  ## debug only

for episode in range(n_epochs):
    try:
        j = 0
        observation = sim.reset(env.starting_position)
        done = False
        penalties, reward, errors = 0, 0, 0
        while j < n_moves:
            try:
                state = env.get_state_from_observation(observation)
                
                action, actions_q_vector = Q.select_action(state, episode, n_epochs)
                
                
                observation_, actions_binary, disp_ctrl = sim.update(action, reward)
                
                reward, done, kms_to_go = env.step(action, observation, observation_)
                
                state_ = env.get_state_from_observation(observation_)
                
                Q.learn(state, action, reward, state_)
                if reward < 0.0:
                    penalties = penalties + 1
                print_summary_results(episode, j, reward, kms_to_go, penalties, state, actions_binary, observation_, disp_ctrl, stage, actions_q_vector, env)
                
                observation = observation_
                
                j = j + 1
                if done:
                    break
                     
            except:
                print("the while loop error number ", errors)
                errors = errors + 1
                continue
               
    except:
        print("error - the for loop - probably comm with x-plane")
        continue
                

#############################################################################
##
##   Testing
##
#############################################################################
## After learning the Q Table
## now fly based on what you learned

stage = "testing ..."

n_epochs = 100
n_moves  = 200

#############################################################################

for episode in range(n_epochs):
    try:
        j = 0
        observation = sim.reset(env.starting_position)
        errors, reward = 0, 0
        while j < n_moves:
            try:
                state = env.get_state_from_observation(observation)
                action, actions_q_vector = Q.select_action(state, 90, 100)
                
                observation_, actions_binary, disp_ctrl = sim.update(action, reward)
                reward, done, kms_to_go = env.step(action, observation, observation_)
                
                state_ = env.get_state_from_observation(observation_)

                print_summary_results(episode, j, 0, 0, 0, state, actions_binary, observation_, disp_ctrl, stage, actions_q_vector, env)
                
                observation = observation_
                
                j = j + 1
                     
            except:
                print("the while loop error number ", errors)
                errors = errors + 1
                continue
               
    except:
        print("error - the for loop - probably comm with x-plane")
        continue
                

#############################################################################

print("<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>")


