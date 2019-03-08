import collections
import deepqnetwork
import gym
import numpy
import random
import ringbuffer

import tensorflow as tf

from skimage import img_as_ubyte, io
from skimage.transform import resize
from skimage.color import rgb2gray
from datetime import datetime


# To draw images
#io.imshow(ret)
#io.show()

def preProcessFrame(frame):
    image = img_as_ubyte(resize(rgb2gray(frame), output_shape=(84, 84), mode='constant'))
    return image

def initEpisode(env, stackSize, noopAction):
    stack = collections.deque([], maxlen=stackSize)
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
replayMemorySize = 800000 #million in experience replay paper
initExploration = 1.0
finalExploration = 0.1
frameStackSize = 4
numTrainingFrames = 50000000
learningStart = 50000
learningFrequency = 4
initExploration = 1
finalExploration = .1
finalExplorationFrame = 1000000
updateFrequency = learningFrequency * 10000

# runtime vars
env = gym.make('Breakout-v0')
noopAction = env.unwrapped.get_action_meanings().index("NOOP")
replayMemory = ringbuffer.RingBuffer(replayMemorySize)
qNetworkLearn = deepqnetwork.DeepQNetwork("learn", env.action_space.n, frameStackSize)
qNetworkEval = deepqnetwork.DeepQNetwork("eval", env.action_space.n, frameStackSize)

# simulate games
frameStack = initEpisode(env, frameStackSize, noopAction)
totalReward = 0
startTime = datetime.now()
numEpisodes = 0

playback = True

saver = tf.train.Saver()
with tf.Session() as session:
    if playback:
        saver.restore(sess=session, save_path='./model/breakout-2640000')
    else:
        session.run(tf.global_variables_initializer())

    #print(tf.trainable_variables())
    for frame in range(numTrainingFrames):
        # pick an action
        action = 0
        if playback or numpy.random.ranf() > lerp(initExploration, finalExploration, frame / finalExplorationFrame):
            action = numpy.argmax(qNetworkEval.predict(session, frameStack))
        else:
            action = env.action_space.sample()

        # simulate step, record history
        observation, reward, episodeDone, info = env.step(action)
        oldStack = frameStack.copy()
        frameStack.append(preProcessFrame(observation))
        replayMemory.append((oldStack, action, numpy.sign(reward), episodeDone, frameStack.copy()))
        totalReward += reward

        if playback:
            env.render('human')

        # train from experience
        if not playback and frame > learningStart and frame % learningFrequency == 0:
            qNetworkLearn.train(session, replayMemory.sample(minibatchSize))

        if not playback and frame > learningStart and frame % updateFrequency == 0 :
            #path = saver.save(learnSession, './model/model', global_step=frame)
            #saver.restore(sess=evalSession, save_path=path)
            update_weights = [tf.assign(new, old) for (new, old) in zip(tf.trainable_variables('eval'), tf.trainable_variables('learn'))]
            session.run(update_weights)
            print("Loaded new model")

        # on finishing an episode, reset simulation
        if episodeDone:
            frameStack = initEpisode(env, frameStackSize, noopAction)
            print("Frame: " + str(frame) + " Score: " + str(totalReward) + " Time: " + str((datetime.now() - startTime).total_seconds()))
            startTime = datetime.now()
            totalReward = 0
            numEpisodes += 1

    env.close()

#qNetwork.save()
print("Done!")