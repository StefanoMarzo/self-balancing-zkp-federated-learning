import numpy as np
from lib.utils import *
from lib.utils import properties as p
import matplotlib.pyplot as plt


# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation_function):
        self.activation = activation_function.fun
        self.activation_prime = activation_function.fun_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


# activation function and its derivative
class ActivationFunction:
    def __init__(self, fun, prime):
        self.fun = fun
        self.fun_prime = prime
    
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

tanh_act = ActivationFunction(tanh, tanh_prime)
sigm_act = ActivationFunction(sigmoid, sigmoid_prime)


# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;


# In[28]:


class Network:
    def __init__(self, seed=None):
        if seed != None:
            np.random.seed(seed)
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.errors = []
        self.accuracy = []
        self.correct = 0
        self.num_samples_fit = 0
        self.num_correct_fit = 0
        self.num_fit_calls = 0

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        #print(f'x: {x_train}, y: {y_train}, epochs: {epochs}, lr: {learning_rate}')
        # sample dimension first
        samples = len(x_train)
        # training loop
        for i in range(epochs):
            err = 0
            corr = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)
                
                if self.is_prediction_correct(y_train[j], output):
                    self.num_correct_fit += 1

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
                    
                self.num_samples_fit += 1
                    
        self.errors += [err / samples / epochs]                
        self.accuracy += [self.num_correct_fit / self.num_samples_fit]            
            
            
    def is_prediction_correct(self, y_true, y_pred):
        return np.all(np.array(convert_output([np.array([y_pred])])) == np.array(y_true))
            
    def get_weights(self):
        return [self.layers[i].weights for i in range(0, len(self.layers), 2)]
    
    def load_weights(self, W):
        for i in range(0, len(W)):
            self.layers[i * 2].weights = W[i] #One FCLayer every 2 layers
            
    def plot_accuracy_loss(self):
        fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=(5, 5))
        plt.ylim((0,1))
        x = np.arange(len(self.errors))
        #fig.suptitle('')
        axs.title.set_text('Accuracy')
        axs.plot(x, self.accuracy)
        plt.show()
            

def get_standard_nn(seed):
    hidden_layer_size_1 = 96
    hidden_layer_size_2 = 80
    net = Network(seed=p['seed'])
    net.add(FCLayer(p['img_size'] * p['img_size'], hidden_layer_size_1))
    net.add(ActivationLayer(sigm_act))
    #net.add(FCLayer(hidden_layer_size_1, hidden_layer_size_2))
    #net.add(ActivationLayer(sigm_act))
    net.add(FCLayer(hidden_layer_size_1, round(p['age_range']() / p['age_bins'])))
    net.add(ActivationLayer(sigm_act))
    # setup
    net.use(mse, mse_prime)
    return net

def get_3l_nn(inp=2, hid=2, out=1):
    net = Network(seed=p['seed'])
    net.add(FCLayer(inp, hid))
    net.add(ActivationLayer(tanh_act))
    net.add(FCLayer(hid, out))
    net.add(ActivationLayer(tanh_act))
    # setup
    net.use(mse, mse_prime)
    return net