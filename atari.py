import argparse
import collections
import common.wrappers
import gym
import numpy as np
from qlearning.qlearning import QLearn, MainNetVariableScope, TargetNetVariableScope, Playback
import qlearning.types as qltypes
import tensorflow as tf
import sys
try:
    import gym_tetris
except:
    pass

from skimage import img_as_ubyte, io
from skimage.transform import resize
from skimage.color import rgb2gray

class Model(qltypes.Model):
    def __init__(self, stackSize, numOutputs, optimizer):
        #prep the inputs
        self.inputs = tf.placeholder(shape=[None,84,84,stackSize], dtype=tf.uint8)
        self.qys = tf.placeholder(shape=[None], dtype=tf.float32)
        self.selectedActions = tf.placeholder(shape=[None], dtype=tf.int32)

        floatInputs = tf.cast(self.inputs, dtype=tf.float32)
        normalizedInputs = tf.math.divide(floatInputs, 255)

        #first convolution layer
        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(8,8), strides=4, padding='valid', activation='relu'
            , input_shape=(84,84,4), kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2), name="conv1").apply(normalizedInputs)

        #second convolution layer
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=2, padding='valid', activation='relu'
            , kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2), name="conv2").apply(conv1)

        #third convolution layer
        conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='valid', activation='relu'
            , kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2), name="conv3").apply(conv2)

        #flatten
        convFlattened = tf.keras.layers.Flatten(name="flatten").apply(conv3)

        #hidden layer
        hidden = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2), name="hidden").apply(convFlattened)

        #output
        self.outputs = tf.keras.layers.Dense(numOutputs, name="outputs").apply(hidden)

        #training
        actions_onehot = tf.one_hot(self.selectedActions, numOutputs, dtype=tf.float32)
        qs = tf.reduce_sum(tf.multiply(self.outputs, actions_onehot), axis=1)
        loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.qys, predictions=qs))
        self.trainFn = optimizer.minimize(loss)

    def getInputs(self):
        return self.inputs
    def getOutputs(self):
        return self.outputs
    def getActions(self):
        return self.selectedActions
    def getYs(self):
        return self.qys
    def getTrain(self):
        return self.trainFn

class FrameProcessor(qltypes.ObservationPreProcessor):
    def __init__(self, stackSize):
        self.newEpisode = True
        self.frameStack = collections.deque([], maxlen=stackSize)

    def resetEnv(self):
        self.newEpisode = True

    def process(self, observation):
        self.frameStack.append(observation)
        if self.newEpisode:
            for _ in range(self.frameStack.maxlen):
                self.frameStack.append(observation)
            self.newEpisode = False
        return self.frameStack.copy()

class FrameStacker(qltypes.ObservationPreFeedProcessor):
    def __init__(self):
        pass
    
    def process(self, preProcessedObservation):
        return np.stack(preProcessedObservation, axis=2)

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='PongDeterministic-v4', help='Define Environment')
parser.add_argument('--saveDir', type=str, default="", help="Path to save checkpoints to")
parser.add_argument('--restorePath', type=str, default="", help="Restore a saved checkpoints to begin training from" )
parser.add_argument('--playback', action='store_true', help="Plays back a restored policy, requires --restorePath")
args = parser.parse_args()

stackSize = 4
env = gym.make(args.env)
env = common.wrappers.ResizeFrame(env, 84, 84)

with tf.variable_scope(MainNetVariableScope):
    mainNet = Model(stackSize
        , env.action_space.n
        , tf.train.AdamOptimizer(learning_rate=0.0000625, epsilon=0.00015))
with tf.variable_scope(TargetNetVariableScope):
    targetNet = Model(stackSize
        , env.action_space.n
        , tf.train.AdamOptimizer(learning_rate=0.0000625, epsilon=0.00015))

if not args.playback:
    QLearn(env = env
        , mainNet = mainNet
        , targetNet = targetNet
        , initExperienceFrames = 50000
        , trainingFrames = 10000000
        , replayMemorySize = 800000
        , initExploration = 1.0
        , finalExploration = 0.01
        , explorationSteps = 1000000
        , learningStart = 30000
        , learningFrequency = 4
        , learningBatchSize = 32
        , observationPreProcessor = FrameProcessor(stackSize)
        , observationPreFeedProcessor = FrameStacker()
        , savePath = args.saveDir
        , restorePath = args.restorePath)
else:
    Playback(env = env
        , mainNet = mainNet
        , restorePath = args.restorePath
        , observationPreProcessor = FrameProcessor(stackSize)
        , observationPreFeedProcessor = FrameStacker()
        , fps = 30)