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


import pickle
from numpy_neural.layers import ConvLayer, FCLayer, ReLULayer
from numpy_neural.optimization import SoftmaxCrossEntropyWithLogits


class FullyConnectedNeuralNet:

    def __init__(self, prev_layer=None, neuron_counts=[], load_path='', learn_rate=.8, momentum=.009,
                 param_init_mean=0., param_init_var=.001, param_init_type='standard_normal'):
        if load_path == '':
            self.layers = []
            self.neuron_counts = neuron_counts
            for i in range(len(neuron_counts) - 1):
                layer = FCLayer(neuron_counts[i], prev_layer=prev_layer,
                                learn_rate=learn_rate, momentum=momentum,
                                param_init_mean=param_init_mean, param_init_var=param_init_var,
                                param_init_type=param_init_type)
                self.layers.append(layer)
                layer = ReLULayer(self.layers[-1])
                self.layers.append(layer)
                prev_layer = layer
            layer = FCLayer(neuron_counts[-1], prev_layer=prev_layer,
                            learn_rate=learn_rate, momentum=momentum,
                            param_init_mean=param_init_mean, param_init_var=param_init_var,
                            param_init_type=param_init_type)
            self.layers.append(layer)
        else:
            self.load(load_path)
        self.output_layer = self.layers[-1]
        self.loss_module = SoftmaxCrossEntropyWithLogits(self.output_layer)
        self.num_layers = len(self.layers)

    def feed_forward(self, x):
        self.layers[0].forward(x)
        for i in range(1, self.num_layers):
            self.layers[i].forward(self.layers[i - 1].y)
        self.out = self.layers[-1].y
        return self.out

    def compute_gradients(self, y_one_hot):
        self.loss_module.compute_backprop_derivatives(y_one_hot)
        self.layers[-1].compute_gradients(self.loss_module.backprop_derivatives)
        for i in range(self.num_layers - 2, -1, -1):
            self.layers[i].compute_gradients(self.layers[i + 1].backprop_derivatives)

    def update_weights(self):
        for i in range(self.num_layers):
            self.layers[i].update_params()

    def train_step(self, x, y):
        self.feed_forward(x)
        self.compute_gradients(y)
        self.update_weights()
        return self.loss_module.loss

    def save(self, path):
        pickle.dump(self.layers, open(path, 'wb'))

    def load(self, path):
        self.layers = pickle.load(open(path, 'rb'))
        self.output_layer = self.layers[-1]
        self.neuron_counts = [self.layers[0].x.shape[1]]
        for layer in self.layers[1:]:
            self.neuron_counts.append(layer.params.shape[0])
        self.num_layers = len(self.layers)


class ConvolutionalNeuralNet:

    def __init__(self, prev_layer=None, neuron_counts=[], load_path='', learn_rate=.8, momentum=.009,
                 param_init_mean=0., param_init_var=.001, param_init_type='standard_normal'):
        if load_path == '':
            self.layers = []
            self.neuron_counts = neuron_counts
            for i in range(len(neuron_counts) - 1):
                layer = ConvLayer(neuron_counts[i], kernel_size=3, stride=1, prev_layer=prev_layer,
                                  learn_rate=learn_rate, momentum=momentum,
                                  param_init_mean=param_init_mean, param_init_var=param_init_var,
                                  param_init_type=param_init_type)
                self.layers.append(layer)
                layer = ReLULayer(self.layers[-1])
                self.layers.append(layer)
                prev_layer = layer
            layer = FCLayer(neuron_counts[-1], prev_layer=prev_layer,
                            learn_rate=learn_rate, momentum=momentum,
                            param_init_mean=param_init_mean, param_init_var=param_init_var,
                            param_init_type=param_init_type)
            self.layers.append(layer)
        else:
            self.load(load_path)
        self.output_layer = self.layers[-1]
        self.loss_module = SoftmaxCrossEntropyWithLogits(self.output_layer)
        self.num_layers = len(self.layers)

    def feed_forward(self, x):
        self.layers[0].forward(x)
        for i in range(1, self.num_layers):
            self.layers[i].forward(self.layers[i - 1].y)
        self.out = self.layers[-1].y
        return self.out

    def compute_gradients(self, y_one_hot):
        self.loss_module.compute_backprop_derivatives(y_one_hot)
        self.layers[-1].compute_gradients(self.loss_module.backprop_derivatives)
        for i in range(self.num_layers - 2, -1, -1):
            self.layers[i].compute_gradients(self.layers[i + 1].backprop_derivatives)

    def update_weights(self):
        for i in range(self.num_layers):
            self.layers[i].update_params()

    def train_step(self, x, y):
        self.feed_forward(x)
        self.compute_gradients(y)
        self.update_weights()
        return self.loss_module.loss

    def save(self, path):
        pickle.dump(self.layers, open(path, 'wb'))

    def load(self, path):
        self.layers = pickle.load(open(path, 'rb'))
        self.output_layer = self.layers[-1]
        self.neuron_counts = [self.layers[0].x.shape[1]]
        for layer in self.layers[1:]:
            self.neuron_counts.append(layer.params.shape[0])
        self.num_layers = len(self.layers)