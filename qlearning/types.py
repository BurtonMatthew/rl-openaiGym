from abc import ABCMeta, abstractmethod
from typing import NewType

class Model(metaclass=ABCMeta):
    @abstractmethod
    def getInputs(self):
        pass
    @abstractmethod
    def getOutputs(self):
        pass
    @abstractmethod
    def getActions(self):
        pass
    @abstractmethod
    def getYs(self):
        pass
    @abstractmethod
    def getTrain(self):
        pass

class ObservationPreProcessor(metaclass=ABCMeta):
    @abstractmethod
    def resetEnv(self):
        pass
    @abstractmethod
    def process(self, observation):
        pass

class ObservationPreFeedProcessor(metaclass=ABCMeta):
    @abstractmethod
    def process(self, preProcessedObservation):
        pass