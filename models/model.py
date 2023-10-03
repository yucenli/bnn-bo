from abc import ABC, abstractmethod

import botorch


class Model(botorch.models.model.Model, ABC):

    @abstractmethod
    def fit_and_save(self, train_x, train_y, save_dir):
        r"""Fits the model to the provided points.

        Args:
            train_x: A n x d tensor of queried function points
            train_y: A n x o tensor of function values at the queried potins
            save_dir: A string with the directory name corresponding to the model.
                      Can be used to save information or model diagnostics
        """
        pass