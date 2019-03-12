import collections
import gym
import numpy as np
import tensorflow as tf
import qlearning.types as qltypes
from qlearning.ringbuffer import RingBuffer

class _PassThroughPreProcessor(qltypes.ObservationPreProcessor):
    def resetEnv(self):
        pass
    def process(self, observation):
        return observation

class _PassThroughFeedProcessor(qltypes.ObservationPreFeedProcessor):
    def process(self, preProcessedObservation):
        return preProcessedObservation

MainNetVariableScope = "main"
TargetNetVariableScope = "target"

def QLearn(   env : gym.Env
            , mainNet : qltypes.Model
            , initExperienceFrames : int
            , trainingFrames : int
            , replayMemorySize : int
            , initExploration : float
            , finalExploration : float
            , explorationSteps : int
            , learningStart : int
            , learningFrequency : int
            , learningBatchSize : int
            , observationPreProcessor : qltypes.ObservationPreProcessor = _PassThroughPreProcessor() 
            , observationPreFeedProcessor : qltypes.ObservationPreFeedProcessor = _PassThroughFeedProcessor()
            , targetNet : qltypes.Model = None
            , targetNetUpdateFrequency : int = 10000):
        if not isinstance(env, gym.Env):
            raise RuntimeError("Environment must be an OpenAI Gym Environment")
        if not isinstance(mainNet, qltypes.Model):
            raise RuntimeError("Main net must be a qltypes model")
        if not isinstance(observationPreProcessor, qltypes.ObservationPreProcessor):
            raise RuntimeError("ObservationPreProcessor must be a qltypes preprocessor")
        if not isinstance(observationPreFeedProcessor, qltypes.ObservationPreFeedProcessor):
            raise RuntimeError("ObservationPreFeedProcessor must be a qltypes preprocessor")
        if targetNet != None:
            if type(mainNet) != type(targetNet):
                raise RuntimeError("Main and target nets must be the same class")
            UseTargetNetwork = True
        else:
            UseTargetNetwork = False

        epsilons = np.linspace(initExploration, finalExploration, explorationSteps)
        memory = RingBuffer(replayMemorySize)
        rewardHistory = collections.deque([], maxlen=100)

        # run inital experience
        episodeDone = True
        for _ in range(initExperienceFrames):
            if episodeDone:
                observationPreProcessor.resetEnv()
                prevObservation = observationPreProcessor.process(env.reset())
            action = env.action_space.sample()
            observation, reward, episodeDone, info = env.step(action)
            observation = observationPreProcessor.process(observation)
            memory.append((prevObservation, action, np.sign(reward), episodeDone, observation))
            prevObservation = observation

        # train
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            # counters
            totalReward = 0
            episodeCount = 0
            learnSteps = 0
            episodeDone = False

            #init starting state
            observationPreProcessor.resetEnv()
            prevObservation = observationPreProcessor.process(env.reset())
            for frame in range(trainingFrames):
                # reset the environment if we finished an episode
                if episodeDone:
                    episodeCount += 1
                    rewardHistory.append(totalReward)
                    if episodeCount % 10 == 0:
                        print("Episode: " + str(episodeCount) + " Frame: " + str(frame) + " ScoreAvg: " + str(np.mean(rewardHistory)))
                    totalReward = 0
                    observationPreProcessor.resetEnv()
                    prevObservation = observationPreProcessor.process(env.reset())

                # select an action to take this observation
                if np.random.ranf() < epsilons[min(frame, explorationSteps-1)]:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(_predict(session, targetNet if UseTargetNetwork else mainNet, observationPreFeedProcessor, prevObservation))

                # step
                observation, reward, episodeDone, info = env.step(action)
                observation = observationPreProcessor.process(observation)
                memory.append((prevObservation, action, np.sign(reward), episodeDone, observation))
                prevObservation = observation
                totalReward += reward

                if frame > learningStart and frame % learningFrequency == 0:
                    _train(session, mainNet, observationPreFeedProcessor, memory.sample(learningBatchSize))
                    learnSteps += 1
                    if UseTargetNetwork and learnSteps % targetNetUpdateFrequency == 0:
                        _updateTargetNet(session)

            episodeDone = True
            while True:
                if episodeDone:
                    observationPreProcessor.resetEnv()
                    prevObservation = observationPreProcessor.process(env.reset())
                action = np.argmax(_predict(session, mainNet, observationPreFeedProcessor, prevObservation))
                observation, reward, episodeDone, info = env.step(action)
                observation = observationPreProcessor.process(observation)
                prevObservation = observation
                env.render()

def _predict(session, model, proc, obs):
    return session.run(model.getOutputs(), feed_dict = { model.getInputs(): np.expand_dims(proc.process(obs), axis=0) })

def _train(session, model, proc, memories):
    xs = []
    ys = []
    for xObs, _, _, _, yObs in memories:
        xs.append(proc.process(xObs))
        ys.append(proc.process(yObs))

    yPredicts = session.run(model.getOutputs(), feed_dict = { model.getInputs(): np.stack(ys) })

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

    session.run([model.getTrain()], feed_dict = {model.getInputs(): np.stack(xs), model.getYs(): np.stack(qys), model.getActions(): np.stack(actions)})

def _updateTargetNet(session):
    update_weights = [tf.assign(new, old) for (new, old) in zip(tf.trainable_variables(TargetNetVariableScope), tf.trainable_variables(MainNetVariableScope))]
    session.run(update_weights)