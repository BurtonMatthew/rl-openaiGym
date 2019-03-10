import gym
import numpy as np
import tensorflow as tf
import qlearning.types as qltypes
from qlearning.ringbuffer import RingBuffer

def QLearn(   env : gym.Env
            , model : qltypes.Model
            , initExperienceFrames : int
            , trainingFrames : int
            , replayMemorySize : int
            , initExploration : float
            , finalExploration : float
            , explorationSteps : int
            , learningStart : int
            , learningFrequency : int
            , learningBatchSize : int):
        if not isinstance(env, gym.Env):
            raise RuntimeError()
        if not isinstance(model, qltypes.Model):
            raise RuntimeError()

        epsilons = np.linspace(initExploration, finalExploration, explorationSteps)
        memory = RingBuffer(replayMemorySize)

        # run inital experience
        episodeDone = True
        for _ in range(initExperienceFrames):
            if episodeDone:
                prevObservation = env.reset()
            action = env.action_space.sample()
            observation, reward, episodeDone, info = env.step(action)
            memory.append((prevObservation, action, np.sign(reward), episodeDone, observation))
            prevObservation = observation

        # train
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            episodeDone = True
            totalReward = 0
            for frame in range(trainingFrames):
                # reset the environment if we finished an episode
                if episodeDone:
                    print("Frame: " + str(frame) + " Score: " + str(totalReward))
                    totalReward = 0
                    prevObservation = env.reset()

                # select an action to take this observation
                if np.random.ranf() < epsilons[min(frame, explorationSteps-1)]:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(model.predict(session, prevObservation))

                # step
                observation, reward, episodeDone, info = env.step(action)
                memory.append((prevObservation, action, np.sign(reward), episodeDone, observation))
                prevObservation = observation
                totalReward += reward

                if frame > learningStart and frame % learningFrequency == 0:
                    model.train(session, memory.sample(learningBatchSize))

                #env.render()