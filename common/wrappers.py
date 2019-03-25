import gym
import numpy as np
from skimage import img_as_ubyte, io
from skimage.transform import resize
from skimage.color import rgb2gray

#class RecordGifWrapper(gym.ObservationWrapper):
#    def __init__(self, env, saveDir="./gifs/"):
#        super(ObservationWrapper, self).__init__(env)
#        self.dir = saveDir

#class FrameStack(gym.Wrapper):


class ResizeFrame(gym.ObservationWrapper):
    def __init__(self, env, height, width):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, observation):
        return img_as_ubyte(resize(rgb2gray(observation), output_shape=(self.height, self.width), mode='constant'))