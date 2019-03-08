import gym
import numpy as np
import tensorflow as tf
import qlearning.types as qltypes
import ringbuffer



def QLearn(   env : gym.Env
            , model : qltypes.Model
            , optimizer
            , trainingFrames
            , replayMemorySize
            , initExploration
            , finalExploration
            , explorationSteps):
        if not isinstance(env, gym.Env):
            raise RuntimeError()
        if not isinstance(model, qltypes.Model):
            raise RuntimeError()

        epsilons = np.linspace(initExploration, finalExploration, explorationSteps)

        with tf.Session() as session:
            episodeDone = True
            for frame in range(trainingFrames):
                # reset the environment if we finished an episode
                if episodeDone:
                    env.reset()

                # select an action to take this observation
                action = 0
                if np.random.ranf() < epsilons[min(frame, explorationSteps-1)]:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(model.predict(session, 1))

                # step
                observation, reward, episodeDone, info = env.step(action)
                env.render()