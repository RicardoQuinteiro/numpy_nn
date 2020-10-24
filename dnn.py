import numpy.random as rnd
import numpy as np
from copy import deepcopy
import json

class NN:

    def __init__(self, dims, actf='sigmoid', lossf='mse', lr=0.001):
        nn = self.initNN(dims)
        self.nn = nn[0]
        self.bias = nn[1]
        self.nn_dim = len(self.nn)
        self.activationf_selector(actf)
        self.errorf_selector(lossf)
        self.lr = lr

    def activationf_selector(self, actf):
        if actf == 'sigmoid':
            self.activationf = self.Sigmoid
            self.dactivationf = self.dSigmoid
        elif actf == 'relu':
            self.activationf = self.ReLU
            self.dactivationf = self.dReLU
        else:
            raise ValueError('Activation function code incorrect')

    def errorf_selector(self, errorf):
        if errorf == 'mse':
            self.errorf = self.MSE
            self.derrorf = self.dMSE

    def initNN(self, d):
        matrices_dims = [(d[i],d[i+1]) for i in range(len(d)-1)]
        matrices = [(rnd.random(dim).T - 0.5)/0.5 for dim in matrices_dims]
        biases = [rnd.random((dim,1)) for dim in d[1:]]
        return [matrices, biases]

    def save_weights(self, filename):
        nn_dict = {'nn': [layer.tolist() for layer in self.nn], 'bias': [bias.tolist() for bias in self.bias]}
        with open(filename, 'w') as file:
            json.dump(nn_dict, file)

    def load_weights(self, filename):
        with open(filename, 'r') as json_file:
            dic = json.load(json_file)

        nn = [np.array(l) for l in dic['nn']]
        biases = [np.array(b) for b in dic['bias']]

        if self.nn_dim == len(nn) and self.nn_dim == len(biases):
            for layer, bias, slayer, sbias in zip(nn, biases, self.nn, self.bias):
                if not (layer.shape == slayer.shape and bias.shape == sbias.shape):
                    raise ValueError('Neural Network dimensions do not match')

            self.nn = nn
            self.bias = biases

        else:
            raise ValueError('Neural Network dimensions do not match')


    def forward(self, x):
        r = [x]
        for layer,bias in zip(self.nn, self.bias):
            x = self.activationf(np.dot(layer,x) + bias)
            r += [x]
        return r

    def softmax(self, x):
        x_ = np.exp(x)
        return x_/np.sum(x_)

    def backpropagation(self, x, y):
        out = x[-1]
        dx = [self.derrorf(out, y)]
        db = []
        dM = []
        hidden_x = x[:-1]
        for i in range(self.nn_dim):
            M = self.nn[-(i+1)]
            b = self.bias[-(i+1)]
            layer_x = hidden_x[-(i+1)]
            x_ = np.dot(M, layer_x) + b
            db = [dx[-(i+1)]*self.dactivationf(x_)] + db
            dx = [np.dot(M.T, db[-(i+1)])] + dx
            dM = [np.dot(db[-(i+1)], layer_x.T)] + dM
        return dM, db

    def grad_descent(self, d):
        dNN, db = d
        for nn,bias,dnn,dbias in zip(self.nn, self.bias, dNN, db):
            nn -= self.lr*dnn
            bias -= self.lr*dbias

    def ReLU(self, x):
        return np.maximum(0, x)

    def dReLU(self, x):
        dx = np.ones(x.shape)
        dx[x<=0] = 0
        return dx

    def Sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def dSigmoid(self, x):
        return self.Sigmoid(x)*(1-self.Sigmoid(x))

    def MSE(self, yhat, y):
        return np.sum((yhat-y)**2)/len(yhat)

    def dMSE(self, yhat, y):
        return 2*(yhat-y)/len(yhat)
