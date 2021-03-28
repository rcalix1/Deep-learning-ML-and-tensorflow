#########################################################################
## 2021
## Authors: David Richter and Ricardo A. Calix, Ph.D.
## Paper:
## Deep Q AP for X-Plane 11
##
#########################################################################

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from collections import deque

import numpy as np
import random

#########################################################################

class DeepQLearn():



    def __init__(self, n_stat, n_acts, gamma, learningRate, q_lr, trainBatchSize, memBatchSize, memorySize):

        self.numOfInputs  = n_stat
        self.numOfOutputs = n_acts

        self.memorySize = memorySize
        self.memoryBatchSize = memBatchSize
        self.memoryQueue = deque(maxlen=self.memorySize)
        
        self.learningRate = learningRate
        self.gamma = gamma
        self.q_lr = q_lr
        
        self.trainBatchSize = trainBatchSize
        
        self.model = self.createModel()
        
        
    ######################################################################################

    def createModel(self):
        model = Sequential()
        model.add(    Input(   shape=(self.numOfInputs, )    )     )
        model.add(Dense(int(self.numOfInputs       ),  activation='relu'))
        model.add(Dense(int(self.numOfInputs * 1.0),  activation='relu'))
        model.add(Dense(int(self.numOfInputs * 0.8 ), activation='relu'))
        #model.add(Dense(    self.numOfOutputs,         activation='linear'))  ## in most papers this is linear ...??
        model.add(Dense(    self.numOfOutputs   ))  

        model.compile(loss="mse", optimizer=Adam(lr=self.learningRate))  

        model.summary()
        
        return model

 
        
    ######################################################################
    
    def select_action(self, state, episode, n_epochs):

        action = 0
        
        state_vector = np.identity(self.numOfInputs)[state:state+1]
     
        actions_q_vector = self.model.predict(state_vector)

        more_random = random.randint(5, 6)

        if (  (episode/n_epochs) < 0.30 and more_random % 2 == 0 ):   ## every even is random  
            action = random.randint(0, 5)
            print("random 0")
        elif (  (episode/n_epochs) < 0.75  ):  
            ## this is a vector of size 6 with random normal distribution data
            added_randomness = np.random.randn(  1, self.numOfOutputs   ) * (  1.0/(episode+1)  )    
            actions_q_vector = actions_q_vector + added_randomness
            action = np.argmax(  actions_q_vector  )    
            print("random 1")
        else:
            action = np.argmax(  actions_q_vector  )
            print("random 2")
            
        
        return action, actions_q_vector
        
    ################################################################################
    # update Deep Q Network
    

    def learn(self, state, action, reward, new_state):
        
        #mask = np.zeros((1,6), dtype=np.float32)
        #mask[0, action] = 1.0
        #print(mask)
         
        #[1, n_stat]  [1, n_actions]
        input_state, output_action = self.get_bellman_training_vectors(state, action, reward, new_state)
        #masked_output_action = np.multiply(output_action, mask)
        self.model.fit(input_state, output_action, verbose=0)  ## fits each set (state_vec, action_vec) every move

        concat_inputs_outputs = np.concatenate((input_state, output_action), axis=1)
        #print(concat_inputs_outputs.shape)
        self.memoryQueue.append(    concat_inputs_outputs     )    
  
        

        if len(self.memoryQueue) >= self.memorySize:
            print("Processing miniBatches from memory ...")
            miniBatch_list = random.sample(self.memoryQueue, self.memoryBatchSize)
            np_miniBatch = np.array(   miniBatch_list   )
            np_miniBatch = tf.squeeze(np_miniBatch, axis=(1)  ) 
            input_state_batch   = np_miniBatch[:, :self.numOfInputs]
            output_action_batch = np_miniBatch[:, self.numOfInputs:]
            #print(input_state_batch.shape)
            #print(output_action_batch.shape)
            self.model.fit(input_state_batch, output_action_batch, verbose=0, shuffle=False)  #batch_size=self.trainBatchSize,

    

    #################################################################################
    ##  (state, action, reward, new_state)
    
    def get_bellman_training_vectors(self, state, action, reward, new_state):

        currentState = np.identity(self.numOfInputs)[state:state+1]
        actions_q_vector_old_states = self.model.predict(currentState)
           
        newCurrentState = np.identity(self.numOfInputs)[new_state:new_state+1]
        actions_q_vector_new_states = self.model.predict(newCurrentState)

        maxFutureQ = np.max(  actions_q_vector_new_states )  
            
        bellman_part1 =  actions_q_vector_old_states[0, action]
            
        bellman_part2 = self.q_lr * (reward + self.gamma * maxFutureQ - actions_q_vector_old_states[0, action])
        #bellman_part2 =              (reward + self.gamma * maxFutureQ - actions_q_vector_old_states[0, action])
  
        actions_q_vector_old_states[0, action] = bellman_part1 + bellman_part2
   
        ## tf.math.sigmoid( )

        return currentState, actions_q_vector_old_states
        
    #################################################################################
        
        
     
