from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def fit_and_save(self, train_x, train_y, save_dir):
        pass