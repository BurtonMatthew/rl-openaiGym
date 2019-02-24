import collections
import gym
import random
import tensorflow as tf

# hyperParams
minibatchSize = 32
replayMemorySize = 100000 #million in experience replay paper

# data
replayMemory = collections.deque([], replayMemorySize) #dequeue has O(n/64) access, should be list, but this handles dropping old elements for free

# simulate games
env = gym.make('Pong-v0')
observation = env.reset()
for _ in range(10000):
    # pick an action
    action = env.action_space.sample()
    # simulate step, record history
    nextObservation, reward, done, info = env.step(action)
    replayMemory.append((observation, action, reward, nextObservation))
    observation = nextObservation
    if done:
        observation = env.reset()
env.close()

# train
random.sample(replayMemory, minibatchSize)