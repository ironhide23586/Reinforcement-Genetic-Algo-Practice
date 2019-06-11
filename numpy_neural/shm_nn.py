import numpy as np
import pickle


class Layer:

    def __init__(self, prev_layer=None, is_input=False, trainable=True, stop_grads=False,
                 learn_rate=1e-3, momentum=1e-4):
        self.is_input = is_input
        self.prev_layer = prev_layer
        self.trainable = trainable
        self.forward_pass_done = False

        self.backward_pass_done = False
        self.batch_size_changed = False
        self.params_dict = {}
        self.layer_type = 'GenericLayer'

        self.feature_dims = []

        self.x = None
        self.y = None

        self.stop_grads = stop_grads
        self.batch_size = 0

        self.learn_rate = learn_rate
        self.momentum = momentum

        if trainable:
            self.prev_param_gradients = None  # copy of gradients from previous iteration for momentum
            self.param_gradients = None  # used to update the weights
            self.backprop_gradients = None  # propagated to previous layer


class InputLayer(Layer):

    def __init__(self):
        Layer.__init__(stop_grads=True, trainable=False, is_input=True)
        self.layer_type = 'Input'


class ReLULayer(Layer):

    def __init__(self, prev_layer, stop_grads=False):
        Layer.__init__(prev_layer=prev_layer, stop_grads=stop_grads, trainable=False, is_input=False)
        self.layer_type = 'Activation/ReLU'

    def relu(self, x):
        y = x.copy()
        y[y < 0.] = 0.
        return y

    def d_relu(self, y):
        dy = y.copy()
        dy[dy < 0.] = 0.
        dy[dy > 0.] = 1.
        return dy

    def compute_gradients(self, derivatives):
        grads = self.d_relu(self.y)
        self.backprop_gradients = grads * derivatives
        return self.backprop_gradients

    def forward(self, x):
        self.x = x
        self.y = self.relu(x)


class SigmoidLayer(Layer):

    def __init__(self, prev_layer, stop_grads=False):
        Layer.__init__(prev_layer=prev_layer, stop_grads=stop_grads, trainable=False, is_input=False)
        self.layer_type = 'Activation/Sigmoid'


class ConvLayer(Layer):  # format is NHWC

    def __init__(self, prev_layer, num_filters, kernel_size, stride, padding='VALID',
                 stop_grads=False, trainable=True, biased=True,
                 param_init_type='gaussian', param_init_mean=0., param_init_var=.001):
        Layer.__init__(prev_layer=prev_layer, stop_grads=stop_grads, trainable=trainable, is_input=False)
        self.layer_type = 'Convolution'

        self.param_init_type = param_init_type
        self.param_init_mean = param_init_mean
        self.param_init_var = param_init_var

        self.biased = biased


class BatchNormLayer(Layer):

    def __init__(self, prev_layer, stop_grads=False, trainable=True):
        Layer.__init__(prev_layer=prev_layer, stop_grads=stop_grads, trainable=trainable, is_input=False)
        self.layer_type = 'BatchNormalization'


class FCLayer(Layer):

    # 0th column is bias, params -> output x input+1
    def __init__(self, prev_layer, output_params,
                 learn_rate=.001, momentum=0., activation='relu',
                 param_init_mean=0., param_init_var=.001, param_init_type='gaussian', trainable=True):
        Layer.__init__(prev_layer, trainable=trainable, is_input=False)
        self.layer_type = 'FullyConnected'

        self.num_outs = output_params
        if prev_layer.layer_type == 'FullyConnected':
            input_params = prev_layer.num_outs
        else:
            input_params = prev_layer.y.reshape([])
        if not self.is_input:
            if param_init_type == 'gaussian':
                self.params = np.random.normal(param_init_mean, param_init_var, size=[self.num_outs, input_params + 1])
            elif param_init_type == 'standard_normal':
                self.params = np.random.randn(self.num_outs, input_params + 1) / np.sqrt(input_params)

    # batch_size x input+1 -> input, 0th column padded with 1s
    def forward(self, x_in):
        if not self.forward_pass_done or x.shape[0] != self.batch_size:
            if x_in.shape[0] != self.batch_size:
                self.batch_size_changed = True
            self.batch_size = x_in.shape[0]
            x = x_in.reshape([self.batch_size, -1])
            self.feature_dims = [x.shape[1]]
            self.x = np.zeros([x.shape[0], x.shape[1] + 1])
            self.x[:, 0] = 1.
            self.forward_pass_done = True
        self.x[:, 1:] = x.astype(np.float32)
        self.y = np.dot(self.x, self.params.T)  # batch_size x output
        # self.y_activation = self.activation(self.y)

    def get_params(self):
        self.params_dict['weights'] = self.params[:, 1:]
        self.params_dict['biases'] = self.params[:, 0]
        return self.params_dict

    # def relu(self, x):
    #     y = x.copy()
    #     y[y < 0.] = 0.
    #     return y
    #
    # def d_relu(self, y):
    #     dy = y.copy()
    #     dy[dy < 0.] = 0.
    #     dy[dy > 0.] = 1.
    #     return dy
    #
    # def sigmoid(self, x):
    #     y = 1. / (1. + np.exp(-x))
    #     return y
    #
    # def d_sigmoid(self, y):
    #     dy = y * (1. - y)
    #     return dy
    #
    # def softmax(self, x):
    #     shiftx = x - np.max(x)
    #     exps = np.exp(shiftx)
    #     den = exps.sum(axis=1)
    #     den = np.tile(den, [exps.shape[1], 1]).T
    #     return exps / den
    #
    # def d_cross_entropy_softmax(self, truth): # truth -> one-hot vector
    #     derivative = self.y_activation - truth
    #     derivative /= derivative.shape[0]
    #     return derivative
    #
    # def no_activation(self, x):
    #     return x.copy()
    #
    # def d_no_activation(self, y):
    #     return y.copy()

    # derivatives -> batch_size x output
    def compute_gradients(self, derivatives):
        # if self.is_input == True:
        #     self.batch_size_changed = False
        #     return
        if self.backward_pass_done == False:
            self.param_gradients = np.zeros_like(self.params)
            self.backward_pass_done = True
        if self.backward_pass_done == False or self.batch_size_changed == True:
            self.backprop_gradients = np.zeros_like([self.batch_size, self.feature_dims[0]])
            self.batch_size_changed = False
        # if self.activation_func != 'softmax':
        #     self.activation_derivatives = self.activation_backprop(self.y_activation)
        #     self.gradient_derivatives = derivatives * self.activation_derivatives
        # else:
        #     self.gradient_derivatives = self.activation_backprop(derivatives)
        self.prev_param_gradients = self.param_gradients.copy()
        self.param_gradients = np.dot(self.x.T, derivatives)
        self.param_gradients = self.param_gradients.T  # output x input+1
        self.backprop_gradients = np.dot(derivatives, self.params[:, 1:])  # batch_size x input
        return self.backprop_gradients

    def update_params(self, learn_rate=None, momentum=None):
        if not self.is_input:
            if learn_rate:
                self.learn_rate = learn_rate
            if momentum:
                self.momentum = momentum
            self.params = self.params - self.learn_rate * self.param_gradients \
                          + self.momentum * self.prev_param_gradients


class FullyConnectedNeuralNet:

    def __init__(self, neuron_counts=[], load_path='', learn_rate=.001, momentum=0., 
                 activation='relu', param_init_mean=0.,
                 param_init_var=.001, param_init_type='standard_normal'):
        if load_path == '':
            self.layers = []
            self.neuron_counts = neuron_counts
            self.input_layer = FCLayer(neuron_counts[0], neuron_counts[1], is_input=True, activation='none')
            self.layers.append(self.input_layer)
            self.num_layers = len(neuron_counts)
            for i in range(1, self.num_layers - 1):
                layer = FCLayer(neuron_counts[i - 1], neuron_counts[i], is_input=False,
                                learn_rate=learn_rate, momentum=momentum, activation=activation,
                                param_init_mean=param_init_mean, param_init_var=param_init_var,
                                param_init_type=param_init_type)
                self.layers.append(layer)
            self.output_layer = FCLayer(neuron_counts[-2], neuron_counts[-1], is_input=False,
                                        learn_rate=learn_rate, momentum=momentum, activation='softmax')
            self.layers.append(self.output_layer)
        else:
            self.load(load_path)

    def feed_forward(self, x):
        self.layers[0].forward(x)
        for i in range(1, self.num_layers):
            self.layers[i].forward(self.layers[i - 1].y_activation)
        self.out = self.layers[-1].y_activation
        return self.out

    def get_backprop_gradients(self, y_one_hot):
        self.layers[-1].compute_gradients(y_one_hot)
        for i in range(self.num_layers - 2, 0, -1):
            self.layers[i].compute_gradients(self.layers[i + 1].backprop_gradients)

    def update_weights(self):
        for i in range(self.num_layers):
            self.layers[i].update_params()

    def train_step(self, x, y):
        self.feed_forward(x)
        self.get_backprop_gradients(y)
        self.update_weights()
        log_likelihood = -np.log((self.out * y).sum(axis=1))
        loss = np.mean(log_likelihood)
        return loss

    def save(self, path):
        pickle.dump(self.layers, open(path, 'wb'))

    def load(self, path):
        self.layers = pickle.load(open(path, 'rb'))
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        self.neuron_counts = [self.layers[0].x.shape[1]]
        for layer in self.layers[1:]:
            self.neuron_counts.append(layer.params.shape[0])
        self.num_layers = len(self.layers)

def to_one_hot(labels, classes):
    ret = np.zeros([labels.shape[0], classes]).astype(np.float)
    ret[range(labels.shape[0]), labels] = 1.
    return ret