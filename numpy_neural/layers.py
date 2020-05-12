import numpy as np
import pickle
import itertools

class Layer:

    def __init__(self, layer_name='', prev_layer=None, learn_rate=1e-3, momentum=1e-4,
                 trainable=True, stop_grads=False, layer_type='GenericLayer'):
        self.layer_name = layer_name
        self.prev_layer = prev_layer
        self.trainable = trainable
        self.forward_pass_done = False

        self.backward_pass_done = False
        self.params_dict = {}
        self.layer_type = layer_type
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

    def init_params(self):
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

    def __init__(self, num_output_features, kernel_size, stride, padding='VALID', layer_name='', prev_layer=None,
                 learn_rate=.001, momentum=0., param_init_mean=0., param_init_var=.001, param_init_type='gaussian',
                 trainable=True, batch_size=None):
        super().__init__(layer_name=layer_name, prev_layer=prev_layer, learn_rate=learn_rate, momentum=momentum,
                         trainable=trainable, layer_type='Convolution')
        self.stride = stride
        self.padding = padding
        self.param_init_type = param_init_type
        self.param_init_mean = param_init_mean
        self.param_init_var = param_init_var
        self.num_output_features = num_output_features
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.prev_layer = prev_layer
        self.output_dims = [-1, -1, -1, self.num_output_features]
        if self.prev_layer is not None:
            self.num_input_features = np.sum(self.prev_layer.output_dims[-1])
            self.init_params()

    def init_params(self):
        if self.param_init_type == 'gaussian':
            self.filters = np.random.normal(self.param_init_mean, self.param_init_var,
                                            size=[self.kernel_size, self.kernel_size,
                                                  self.num_input_features, self.num_output_features])
        elif self.param_init_type == 'standard_normal':
            filters = np.random.randn(self.kernel_size, self.kernel_size,
                                      self.num_input_features, self.num_output_features)
        biases = np.zeros(self.num_output_features)
        self.params = [filters, biases]
        self.in2out_mindim = lambda dim_in: ((dim_in - self.kernel_size) // self.stride) + 1
        out2in_mindim = lambda dim_out: self.stride * dim_out
        out2in_maxdim = lambda dim_out: out2in_mindim(dim_out) + self.kernel_size - 1
        get_receptive_field = lambda x, y: self.x[:, out2in_mindim(y): out2in_maxdim(y) + 1,
                                                  out2in_mindim(x): out2in_maxdim(x) + 1, :]
        reshape_rf_for_conv_input = lambda rf: np.rollaxis(np.tile(np.expand_dims(rf, 1), [1, self.num_output_features,
                                                                                           1, 1, 1]),
                                                           1, 5)
        if self.batch_size is not None:
            self.filters_repeated = np.tile(np.expand_dims(filters, 0), [self.batch_size, 1, 1, 1, 1])
            self.biases_repeated = np.tile(np.expand_dims(biases, 0), [self.batch_size, 1])

        get_conv_output_elemwise = lambda x, y: reshape_rf_for_conv_input(get_receptive_field(x, y)) \
                                                * self.filters_repeated
        get_filters_outs = lambda x, y: get_conv_output_elemwise(x, y).reshape([self.batch_size, -1,
                                                                                self.num_output_features]).sum(axis=1)
        self.get_layer_out_rf = lambda xy: get_filters_outs(xy[0], xy[1]) + self.biases_repeated

    def get_params(self):
        self.params_dict['weights'] = self.params[0]
        self.params_dict['biases'] = self.params[1]
        return self.params_dict

    def forward(self, x):
        batch_size, d_in, _, self.num_input_features = x.shape
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.init_params()
        d_out = self.in2out_mindim(d_in)
        all_out_xys = list(itertools.product(np.arange(d_out), np.arange(d_out)))
        y = np.array(list(map(self.get_layer_out_rf, all_out_xys)))
        self.y = np.transpose(np.rollaxis(y, 0, 2).reshape([self.batch_size, d_out, d_out, -1]), [0, 2, 1, 3])


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
