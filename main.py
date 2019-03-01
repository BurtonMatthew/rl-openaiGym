import collections
import deepqnetwork
import gym
import numpy
import random
import ringbuffer

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

def clamp(curVal, minVal, maxVal):
        return max(min(curVal, maxVal), minVal)

def lerp(minVal, maxVal, alpha):
        return (maxVal - minVal) * clamp(alpha, 0, 1) + minVal

# hyperParams
minibatchSize = 32
replayMemorySize = 100000 #million in experience replay paper
initExploration = 1.0
finalExploration = 0.1
frameStackSize = 4
numTrainingFrames = 1000000 #50000000
learningStart = 10000 #50000
learningFrequency = 4
initExploration = 1
finalExploration = .1
finalExplorationFrame = 500000 #1000000

# runtime vars
env = gym.make('Pong-v0')
noopAction = env.unwrapped.get_action_meanings().index("NOOP")
replayMemory = ringbuffer.RingBuffer(replayMemorySize)
qNetwork = deepqnetwork.DeepQNetwork(env.action_space.n, frameStackSize)

# simulate games
frameStack = initEpisode(env, frameStackSize, noopAction)

for frame in range(numTrainingFrames):
    # pick an action
    action = 0
    if numpy.random.ranf() > lerp(initExploration, finalExploration, frame / finalExplorationFrame):
        action = numpy.argmax(qNetwork.predict(frameStack))
    else:
        action = env.action_space.sample()

    # simulate step, record history
    observation, reward, episodeDone, info = env.step(action)
    oldStack = frameStack.copy()
    frameStack.append(preProcessFrame(observation))
    replayMemory.append((oldStack, action, numpy.sign(reward), episodeDone, frameStack.copy()))

    # train from experience
    if frame > learningStart and frame % learningFrequency == 0:
        qNetwork.train(replayMemory.sample(minibatchSize))

    # on finishing an episode, reset simulation
    if episodeDone:
        frameStack = initEpisode(env, frameStackSize, noopAction)
        print(frame)
env.close()

qNetwork.save()
print("Done!")