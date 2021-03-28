## Q-Learning Object
## Deep Q Learn AP (Auto Pilot)
## xplane environment for training the RL agent
## xPLANE and Reinforcement Learning - AI/RL based AutoPilot
## Deep Q AP
## Book -> Getting Started with Deep Learning: Programming and Methodologies using Python
## By Ricardo Calix
## www.rcalix.com
## Copyright (c) 2020, Ricardo A. Calix, Ph.D.

######################################################################

import numpy as np
import time
import math
import random

#######################################################################

## ["pitch+","pitch-","roll+","roll-","rudd+","rudd-"]
## actions = np.zeros(   n_actions , dtype=int  ) # actions = [] ## 6
## states = np.zeros(  n_states , dtype=int  )
## actions is a one_hot encoded vector
## actions_binary = [pi+, pi-, ro+, ro-, ru+, ru-]
## actions = [] ## 6

#######################################################################

class QLearn():

    def __init__(self, n_stat, n_acts, gamm, lr, eps):
        self.n_states = n_stat
        self.n_actions = n_acts
        self.gamma = gamm
        self.learning_rate = lr
        self.epsilon = eps
        self.q_table = np.zeros( [self.n_states, self.n_actions] )
 
    #############################################################################
    
    def select_action(self, state, episode, n_epochs):

        action = 0
        selected_q_row = None
        
        print("time sleep for GPU (too fast)")
        time.sleep(0.02)
        
        more_random = random.randint(5, 6)

        if ((episode/n_epochs) < 0.30 and (more_random % 2 == 0) ):
            selected_q_row = self.q_table[state,:]
            selected_q_row = selected_q_row[np.newaxis, :]  ## go from (6,) to (1,6)
            action = random.randint(0, 5)
        elif (   (episode/n_epochs) < 0.75   ):
            selected_q_row = self.q_table[state,:]
            selected_q_row = selected_q_row[np.newaxis, :]  ## go from (6,) to (1,6)
            selected_q_row = selected_q_row  +  np.random.randn(    1, self.n_actions  ) * (  1.0/(episode+1)  )
            action = np.argmax(  selected_q_row  )
        else:
            selected_q_row = self.q_table[state,:]
            selected_q_row = selected_q_row[np.newaxis, :]  ## go from (6,) to (1,6)
            action = np.argmax(  selected_q_row  )
           

        
        return action, selected_q_row
        
    ###############################################################################

    def learn(self, s, action, reward, s_):
       lr = self.learning_rate
       y = self.gamma
       a = action
       r = reward
       self.q_table[s,a] = self.q_table[s,a] + lr*(r + y * np.max(self.q_table[s_,:]) - self.q_table[s,a] )


##############################################################################################


