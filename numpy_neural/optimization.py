"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""


import numpy as np


from numpy_neural import utils


class LossLayer:

    def __init__(self, output_layer):
        self.output_layer = output_layer
        self.backprop_derivatives = None
        self.loss = None

    def compute_backrop_derivatives(self, y_one_hot):
        raise NotImplementedError


class SoftmaxCrossEntropyWithLogits(LossLayer):

    def __init__(self, output_layer):
        super().__init__(output_layer=output_layer)

    def compute_backprop_derivatives(self, y_one_hot):
        y_pred = utils.softmax(self.output_layer.y)
        self.backprop_derivatives = y_pred - y_one_hot
        self.backprop_derivatives /= self.backprop_derivatives.shape[0]
        self.loss = np.mean(-np.log((y_pred * y_one_hot).sum(axis=1)))
        return self.backprop_derivatives
