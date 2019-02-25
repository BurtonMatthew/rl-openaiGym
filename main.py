import collections
import deepqnetwork
import gym
import random

from skimage import img_as_ubyte, io
from skimage.transform import resize
from skimage.color import rgb2gray

# To draw images
#io.imshow(ret)
#io.show()

def preProcessFrame(frame):
    return img_as_ubyte(resize(rgb2gray(frame), output_shape=(84, 84), anti_aliasing=True, mode='constant'))

def initEpisode(env, stackSize, noopAction):
    stack = collections.deque([], stackSize)
    stack.append(preProcessFrame(env.reset()))
    for _ in range(stackSize-1):
        stack.append(preProcessFrame(env.step(noopAction)[0]))
    return stack

# hyperParams
minibatchSize = 32
replayMemorySize = 100000 #million in experience replay paper
initExploration = 1.0
finalExploration = 0.1
frameStackSize = 4

# runtime vars
env = gym.make('Pong-v0')
noopAction = env.unwrapped.get_action_meanings().index("NOOP")
replayMemory = collections.deque([], replayMemorySize) #dequeue has O(n/64) access, should be list, but this handles dropping old elements for free
qNetwork = deepqnetwork.DeepQNetwork(env.action_space.n, frameStackSize)

# simulate games
frameStack = initEpisode(env, frameStackSize, noopAction)

for _ in range(10):
    # pick an action
    print(qNetwork.predict(frameStack))
    action = env.action_space.sample()
    # simulate step, record history
    observation, reward, done, info = env.step(action)
    oldStack = frameStack.copy()
    frameStack.append(preProcessFrame(observation))
    replayMemory.append((oldStack, action, reward, frameStack.copy()))
    if done:
        frameStack = initEpisode(env, frameStackSize, noopAction)
env.close()

# train
#random.sample(replayMemory, minibatchSize)