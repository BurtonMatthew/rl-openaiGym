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
            , targetNet : qltypes.Model
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
            , targetNetUpdateFrequency : int = 10000
            , savePath = ""
            , saveFrequencyEpisodes = 200
            , restoreFromSave = False):
        if not isinstance(env, gym.Env):
            raise RuntimeError("Environment must be an OpenAI Gym Environment")
        if not isinstance(mainNet, qltypes.Model):
            raise RuntimeError("Main net must be a qltypes model")
        if not isinstance(observationPreProcessor, qltypes.ObservationPreProcessor):
            raise RuntimeError("ObservationPreProcessor must be a qltypes preprocessor")
        if not isinstance(observationPreFeedProcessor, qltypes.ObservationPreFeedProcessor):
            raise RuntimeError("ObservationPreFeedProcessor must be a qltypes preprocessor")
        if type(mainNet) != type(targetNet):
            raise RuntimeError("Main and target nets must be the same class")

        epsilons = np.linspace(initExploration, finalExploration, explorationSteps)
        memory = RingBuffer(replayMemorySize)
        rewardHistory = collections.deque([], maxlen=100)
        saver = tf.train.Saver()

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
            if restoreFromSave and savePath != "":
                saver.restore(sess=session, save_path=savePath)
            else:
                session.run(tf.global_variables_initializer())
            _updateTargetNet(session)
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
                    if savePath != "" and episodeCount % 20 == 0:
                        saver.save(session, savePath, global_step=frame)
                    totalReward = 0
                    observationPreProcessor.resetEnv()
                    prevObservation = observationPreProcessor.process(env.reset())

                # select an action to take this observation
                if np.random.ranf() < epsilons[min(frame, explorationSteps-1)]:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(_predict(session, mainNet, observationPreFeedProcessor, prevObservation))

                # step
                observation, reward, episodeDone, info = env.step(action)
                observation = observationPreProcessor.process(observation)
                memory.append((prevObservation, action, np.sign(reward), episodeDone, observation))
                prevObservation = observation
                totalReward += reward

                if frame > learningStart and frame % learningFrequency == 0:
                    _train(session, mainNet, targetNet, observationPreFeedProcessor, memory.sample(learningBatchSize))
                    learnSteps += 1
                    if learnSteps % targetNetUpdateFrequency == 0:
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

def _train(session, mainNet, targetNet, proc, memories):
    xs = []
    ys = []
    for xObs, _, _, _, yObs in memories:
        xs.append(proc.process(xObs))
        ys.append(proc.process(yObs))

    # main network estimates best actions
    bestActions = np.argmax(session.run(mainNet.getOutputs(), feed_dict = { mainNet.getInputs(): np.stack(ys) }), axis=1)
    # target network estimates values
    estVals = session.run(targetNet.getOutputs(), feed_dict = { targetNet.getInputs(): np.stack(ys) })

    idx = 0
    qys = []
    actions = []
    for _, action, reward, done, _ in memories:
        if done:
            qys.append(reward)
        else:
            qys.append(reward + .99 * estVals[idx][bestActions[idx]])
        actions.append(action)
        idx += 1

    session.run([mainNet.getTrain()], feed_dict = {mainNet.getInputs(): np.stack(xs), mainNet.getYs(): np.stack(qys), mainNet.getActions(): np.stack(actions)})

def _updateTargetNet(session):
    update_weights = [tf.assign(new, old) for (new, old) in zip(tf.trainable_variables(TargetNetVariableScope), tf.trainable_variables(MainNetVariableScope))]
    session.run(update_weights)