from abc import ABCMeta, abstractmethod
from typing import NewType

class ProcessedObservation(metaclass=ABCMeta):
    pass

class Model(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, session, obs : ProcessedObservation):
        pass
    @abstractmethod
    def train(self, session, obs : [ProcessedObservation]):
        pass