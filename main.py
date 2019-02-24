import collections
import deepqnetwork
import gym
import random

# hyperParams
minibatchSize = 32
replayMemorySize = 100000 #million in experience replay paper

# runtime vars
env = gym.make('Pong-v0')
replayMemory = collections.deque([], replayMemorySize) #dequeue has O(n/64) access, should be list, but this handles dropping old elements for free
net = deepqnetwork.DeepQNetwork(env.action_space.n)

# simulate games
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