import gym
env = gym.make("Taxi-v2").env

env.render()

env.reset()
env.render()

print(   "Action Space {}".format(env.action_space)  )
print(   "State Space {}".format(env.observation_space)  )


state = env.encode(3, 1, 2, 0)

print("State: ", state)

env.s = state

env.render()

print(env.P[328])
 
env.s = 328

epochs = 0

penalties = 0
reward = 0

frames = []

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if reward == -10:
        penalties += 1
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )
    epochs += 1


print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

from IPython.display import clear_output
from time import sleep
import os


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        os.system('clear')
        print(  frame['frame'] )
        print("Timestep: {}".format(i + 1))
        print("State: {}".format(  frame['state']  ))
        print("Action: {}".format(  frame['action']  ) )
        print("Reward: {}".format(  frame['reward']   ))
        sleep(.1)


print_frames(frames)   ##uncomment to view animation

#############################################################
## now add Q table

import numpy as np

q_table = np.zeros([env.observation_space.n, env.action_space.n])

##%%time

import random
from IPython.display import clear_output

alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() #random action
        else:
            action = np.argmax(q_table[state]) # select optimal action in Q table for state
        next_state, reward, done, info = env.step(action)
        old_value =q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max )
        q_table[state, action] = new_value
        
        if reward == -10:
            penalties = penalties + 1
        
        state = next_state
        epochs = epochs + 1

    if i % 100 == 0:
        os.system('clear')
        print("Episode: {}".format(i))

print("Training Finished\n")

print(q_table[328]) ## should predict 1 (north)

#############################################################
## evaluate the agent now that we have learned the Q table after 100,000 tries

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties = penalties + 1
        epochs = epochs + 1

    total_penalties = total_penalties + penalties
    total_epochs = total_epochs + epochs
    
print("Results after {} episodes: ".format(episodes))
print("Average time steps per episode: {} ".format(total_epochs / episodes) )
print("average penalties per episode: {} ".format(total_penalties/episodes) )




#############################################################

print("<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>")
