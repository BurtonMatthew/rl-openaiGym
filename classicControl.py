import gym
import numpy as np
from qlearning.qlearning import QLearn
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
        loss = tf.reduce_mean(tf.square(self.qys - qs))
        self.trainFn = optimizer.minimize(loss)

    def predict(self, session, obs):
        return session.run(self.outputs, feed_dict = { self.inputs: np.expand_dims(obs, axis=0) })

    def train(self, session, memories):
        xs = []
        ys = []
        for xObs, _, _, _, yObs in memories:
            xs.append(xObs)
            ys.append(yObs)

        yPredicts = session.run(self.outputs, feed_dict = { self.inputs: np.stack(ys) })

        idx = 0
        qys = []
        actions = []
        for _, action, reward, done, _ in memories:
            if done:
                qys.append(reward)
            else:
                qys.append(reward + .99 * max(yPredicts[idx]))
            actions.append(action)
            idx += 1

        session.run([self.trainFn], feed_dict = {self.inputs: np.stack(xs), self.qys: np.stack(qys), self.selectedActions: np.stack(actions)})

if len(sys.argv) == 1:
    game = "CartPole-v1"
elif sys.argv[1].tolower() == "cartpole":
    game = "CartPole-v1"
elif sys.argv[1].tolower() == "mountaincar":
    game = "MountainCar-v0"
elif sys.argv[1].tolower() == "acrobat":
    game = "Acrobot-v1"
elif sys.argv[1].tolower() == "pendulum":
    game = "Pendulum-v0"
else:
    game = "CartPole-v1"

env = gym.make(game)
model = Model(env.observation_space.shape[0]
    , env.action_space.n
    , tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95))

QLearn(env = env
    , model = model
    , initExperienceFrames = 50000
    , trainingFrames = 5000000
    , replayMemorySize = 1000000
    , initExploration = 1.0
    , finalExploration = 0.1
    , explorationSteps = 1000000
    , learningStart = 10000
    , learningFrequency = 4
    , learningBatchSize = 32)