import numpy as np
import pickle

class FCLayer():

    # 0th column is bias, params -> output x input+1
    def __init__(self, input_params, output_params, is_input=False,
                 learn_rate=.001, momentum=0., activation='relu',
                 init_mean=0., init_var=.001, init_type='gaussian'):
        self.is_input = is_input
        if self.is_input == False:
            if init_type == 'gaussian':
                self.params = np.random.normal(init_mean, init_var, size=[output_params, input_params + 1])
            elif init_type == 'standard_normal':
                self.params = np.random.randn(output_params, input_params + 1) / np.sqrt(input_params)
        self.forward_pass_done = False
        self.backward_pass_done = False
        self.batch_size_changed = False
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.batch_size = 0
        self.activation_func = activation
        self.x = np.zeros([1, input_params])
        if activation == 'relu':
            self.activation = self.relu
            self.activation_backprop = self.d_relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_backprop = self.d_sigmoid
        elif activation == 'softmax':
            self.activation = self.softmax
            self.activation_backprop = self.d_cross_entropy_softmax
        elif activation == 'none':
            self.activation = self.no_activation
            self.activation_backprop = self.d_no_activation

    # batch_size x input+1 -> input, 0th column padded with 1s
    def forward(self, x):
        if self.is_input == True:
            self.forward_pass_done = True
            self.x = x.copy()
            self.y_activation = self.x.copy()
            return
        if self.forward_pass_done == False or x.shape[0] != self.batch_size:
            if x.shape[0] != self.batch_size:
                self.batch_size_changed = True
            self.x = np.zeros([x.shape[0], x.shape[1] + 1])
            self.x[:, 0] = 1.
            self.fwd_pass_done = True
            self.features = x.shape[1]
            self.batch_size = x.shape[0]
        self.x[:, 1:] = x.astype(np.float32)
        self.y = np.dot(self.x, self.params.T) # batch_size x output
        self.y_activation = self.activation(self.y)

    def relu(self, x):
        y = x.copy()
        y[y < 0.] = 0.
        return y

    def d_relu(self, y):
        dy = y.copy()
        dy[dy < 0.] = 0.
        dy[dy > 0.] = 1.
        return dy

    def sigmoid(self, x):
        y = 1. / (1. + np.exp(-x))
        return y

    def d_sigmoid(self, y):
        dy = y * (1. - y)
        return dy

    def softmax(self, x):
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        den = exps.sum(axis=1)
        den = np.tile(den, [exps.shape[1], 1]).T
        return exps / den

    def d_cross_entropy_softmax(self, truth): # truth -> one-hot vector
        derivative = self.y_activation - truth
        derivative /= derivative.shape[0]
        return derivative

    def no_activation(self, x):
        return x.copy()

    def d_no_activation(self, y):
        return y.copy()

    # derivatives -> batch_size x output
    def compute_gradients(self, derivatives):
        if self.is_input == True:
            self.batch_size_changed = False
            return
        if self.backward_pass_done == False:
            self.param_gradients = np.zeros_like(self.params)
            self.backward_pass_done = True
        if self.backward_pass_done == False or self.batch_size_changed == True:
            self.backprop_gradients = np.zeros_like([self.batch_size, self.features])
            self.batch_size_changed = False
        if self.activation_func != 'softmax':
            self.activation_derivatives = self.activation_backprop(self.y_activation)
            self.gradient_derivatives = derivatives * self.activation_derivatives
        else:
            self.gradient_derivatives = self.activation_backprop(derivatives)
        self.prev_param_gradients = self.param_gradients.copy()
        self.param_gradients = np.dot(self.x.T, self.gradient_derivatives)
        self.param_gradients = self.param_gradients.T #output x input+1
        self.backprop_gradients = np.dot(self.gradient_derivatives, self.params[:, 1:]) #batch_size x input

    def update_params(self, learn_rate=None, momentum=None):
        if self.is_input == False:
            if learn_rate == None:
                learn_rate = self.learn_rate
            if momentum == None:
                momentum = self.momentum
            self.params = self.params - learn_rate * self.param_gradients + momentum * self.prev_param_gradients


def to_one_hot(labels, classes):
    ret = np.zeros([labels.shape[0], classes]).astype(np.float)
    ret[range(labels.shape[0]), labels] = 1.
    return ret