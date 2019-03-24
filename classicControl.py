import gym
import numpy as np
from qlearning.qlearning import QLearn, MainNetVariableScope, TargetNetVariableScope
import qlearning.types as qltypes
import tensorflow as tf
import sys

class Model(qltypes.Model):
    def __init__(self, numInputs, numOutputs, optimizer):
        self.inputs = tf.placeholder(shape=[None,numInputs], dtype=tf.float32)
        self.qys = tf.placeholder(shape=[None], dtype=tf.float32)
        self.selectedActions = tf.placeholder(shape=[None], dtype=tf.int32)
        hidden1 = tf.keras.layers.Dense(24, activation='relu', name="hidden1").apply(self.inputs)
        hidden2 = tf.keras.layers.Dense(24, activation='relu', name="hidden2").apply(hidden1)
        self.outputs = tf.keras.layers.Dense(numOutputs, activation='linear', name="outputs").apply(hidden2)

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

if len(sys.argv) == 1:
    game = "CartPole-v1"
elif sys.argv[1].lower() == "cartpole":
    game = "CartPole-v1"
elif sys.argv[1].lower() == "mountaincar":
    game = "MountainCar-v0"
elif sys.argv[1].lower() == "acrobot":
    game = "Acrobot-v1"
elif sys.argv[1].lower() == "pendulum":
    game = "Pendulum-v0"
else:
    game = "CartPole-v1"

env = gym.make(game)
with tf.variable_scope(MainNetVariableScope):
    model = Model(env.observation_space.shape[0]
        , env.action_space.n
        , tf.contrib.estimator.clip_gradients_by_norm(tf.train.AdamOptimizer(learning_rate=0.0000625, epsilon=0.00015), 1.0))
with tf.variable_scope(TargetNetVariableScope):
    target = Model(env.observation_space.shape[0]
        , env.action_space.n
        , tf.contrib.estimator.clip_gradients_by_norm(tf.train.AdamOptimizer(learning_rate=0.0000625, epsilon=0.00015), 1.0))

QLearn(env = env
    , mainNet = model
    , targetNet = target
    , targetNetUpdateFrequency = 1000
    , initExperienceFrames = 50000
    , trainingFrames = 1000000
    , replayMemorySize = 1000000
    , initExploration = 1.0
    , finalExploration = 0.1
    , explorationSteps = 100000
    , learningStart = 10000
    , learningFrequency = 4
    , learningBatchSize = 32)