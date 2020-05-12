import numpy as np
import pickle


class Layer:

    def __init__(self, layer_name='', prev_layer=None, learn_rate=1e-3, momentum=1e-4,
                 trainable=True, stop_grads=False, layer_type='GenericLayer'):
        self.layer_name = layer_name
        self.prev_layer = prev_layer
        self.trainable = trainable
        self.forward_pass_done = False

        self.backward_pass_done = False
        self.params_dict = {}
        self.layer_type=layer_type
        if self.layer_name == '':
            self.layer_name = str(self.layer_type)
        self.output_dims = None
        self.num_output_features = 0

        self.x = None
        self.y = None
        self.params = None

        self.batch_size = 0

        if self.trainable:
            self.prev_param_gradients = None  # copy of gradients from previous iteration for momentum
            self.param_gradients = None  # used to update the weights
            self.backprop_derivatives = None  # propagated to previous layer
            self.learn_rate = learn_rate
            self.momentum = momentum
            self.stop_grads = stop_grads

    def forward(self, x):
        raise NotImplementedError

    def compute_gradients(self, derivatives):
        if self.trainable:
            raise NotImplementedError

    def get_params(self):
        if self.trainable:
            raise NotImplementedError

    def update_params(self, learn_rate=None, momentum=None):
        if self.trainable:
            raise NotImplementedError


class InputLayer(Layer):

    def __init__(self):
        super().__init__(self, stop_grads=True, trainable=False)
        self.layer_type = 'Input'


class ReLULayer(Layer):

    def __init__(self, prev_layer, stop_grads=False):
        super().__init__(self, prev_layer=prev_layer, stop_grads=stop_grads, trainable=False,
                         layer_type='Activation/ReLU')
        self.output_dims = prev_layer.output_dims

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
        self.backprop_derivatives = grads * derivatives
        return self.backprop_derivatives

    def forward(self, x):
        self.x = x
        self.y = self.relu(x)
        self.output_dims = self.y.shape
        self.num_output_features = np.sum(self.y.shape[1:])


class SigmoidLayer(Layer):

    def __init__(self, prev_layer, stop_grads=False):
        super().__init__(prev_layer=prev_layer, stop_grads=stop_grads, trainable=False,
                         layer_type='Activation/Sigmoid')


class ConvLayer(Layer):  # format is NHWC

    def __init__(self, prev_layer, num_filters, kernel_size, stride, padding='VALID',
                 stop_grads=False, trainable=True, biased=True,
                 param_init_type='gaussian', param_init_mean=0., param_init_var=.001):
        super().__init__(prev_layer=prev_layer, stop_grads=stop_grads, trainable=trainable)
        self.layer_type = 'Convolution'

        self.param_init_type = param_init_type
        self.param_init_mean = param_init_mean
        self.param_init_var = param_init_var

        self.biased = biased


class BatchNormLayer(Layer):

    def __init__(self, prev_layer, stop_grads=False, trainable=True):
        super().__init__(prev_layer=prev_layer, stop_grads=stop_grads, trainable=trainable,
                         layer_type='BatchNormalization')


class FCLayer(Layer):

    # 0th column is bias, params -> output x input+1
    def __init__(self, num_output_features, layer_name='', prev_layer=None, learn_rate=.001, momentum=0.,
                 param_init_mean=0., param_init_var=.001, param_init_type='gaussian', trainable=True):
        super().__init__(layer_name=layer_name, prev_layer=prev_layer, learn_rate=learn_rate, momentum=momentum,
                         trainable=trainable, layer_type='FullyConnected')
        self.prev_layer = prev_layer
        self.output_dims = [-1, num_output_features]
        self.num_output_features = num_output_features
        self.param_init_type = param_init_type
        self.param_init_mean = param_init_mean
        self.param_init_var = param_init_var
        if self.prev_layer is not None:
            self.num_input_features = np.sum(self.prev_layer.output_dims[1:])
            self.init_params()

    def init_params(self):
        if self.param_init_type == 'gaussian':
            self.params = np.random.normal(self.param_init_mean, self.param_init_var,
                                           size=[self.num_output_features, self.num_input_features + 1])
        elif self.param_init_type == 'standard_normal':
            self.params = np.random.randn(self.num_output_features, self.num_input_features + 1) \
                          / np.sqrt(self.num_input_features)

    # batch_size x input+1 -> input, 0th column padded with 1s
    def forward(self, x_in):
        self.batch_size = x_in.shape[0]
        x = x_in.reshape([self.batch_size, -1])
        _, self.num_input_features = x.shape
        if self.params is None:
            self.init_params()
        self.x = np.zeros([self.batch_size, self.num_input_features + 1])
        self.x[:, 0] = 1.
        self.forward_pass_done = True
        self.x[:, 1:] = x.astype(np.float32)
        self.y = np.dot(self.x, self.params.T)  # batch_size x output

    def get_params(self):
        self.params_dict['weights'] = self.params[:, 1:]
        self.params_dict['biases'] = self.params[:, 0]
        return self.params_dict

    # derivatives -> batch_size x num_features
    def compute_gradients(self, derivatives):
        if self.param_gradients is None:
            self.prev_param_gradients = np.zeros([self.num_output_features, self.num_input_features + 1])
        else:
            self.prev_param_gradients = self.param_gradients.copy()
        self.param_gradients = np.dot(self.x.T, derivatives).T
        self.backprop_derivatives = np.dot(derivatives, self.params[:, 1:])  # batch_size x input
        return self.backprop_derivatives

    def update_params(self, learn_rate=None, momentum=None):
        if learn_rate:
            self.learn_rate = learn_rate
        if momentum:
            self.momentum = momentum
        update_matrix = (self.learn_rate * self.param_gradients) + (self.momentum * self.prev_param_gradients)
        self.params = self.params - update_matrix
