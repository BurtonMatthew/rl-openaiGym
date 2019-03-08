import gym
from qlearning.qlearning import QLearn
import qlearning.types as qltypes

class Model(qltypes.Model):
    def __init__(self):
        self.loss = 1

    def getLoss(self):
        return self.loss

    def predict(self, session, obs):
        return 0

    def buidTrainFeedDict(self, session):
        pass

QLearn(gym.make('CartPole-v1')  # env
    , Model()                   # model
    , 1                         # optimizer
    , 1000                      # trainingFrames
    , 500000                    # replayMemorySize
    , 1.0                       # initExploration
    , 0.1                       # finalExploration
    , 10000)                    # explorationSteps