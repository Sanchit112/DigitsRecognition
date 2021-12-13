import math
import pandas as pd
import numpy as np
import sys


class MultiLayerPerceptron:
    def __init__(self, batch_size, learning_rate, num_epochs):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Layer 1: weight, bias
        # w: 512 * (28*28)
        # x: (28*28) * 1
        # b: 512 * 1
        # r: 512 * 1
        # Layer 2: weight, bias
        # w: 256 * 512
        # x: 512 * 1
        # b: 256 * 1
        # r: 256 * 1
        # Layer 3: weight, bias
        # w: 10 * 256
        # x: 256 * 1
        # b: 10 * 1
        # r: 10 * 1
        self.params = {
            'W1': np.random.randn(512, 784) * np.sqrt(1.0 / 784),
            'b1': np.random.randn(512, 1) * np.sqrt(1.0 / 748),
            'W2': np.random.randn(256, 512) * np.sqrt(1.0 / 512),
            'b2': np.random.randn(256, 1) * np.sqrt(1.0 / 512),
            'W3': np.random.randn(10, 256) * np.sqrt(1.0 / 256),
            'b3': np.random.randn(10, 1) * np.sqrt(1.0 / 256)
        }

    def start(self):

        if len(sys.argv) > 3:
            trainX = pd.read_csv(sys.argv[1], header=None)
            trainY = pd.read_csv(sys.argv[2], header=None)
            testX = pd.read_csv(sys.argv[3], header=None)
        else:
            trainX = pd.read_csv('train_image.csv', header=None)
            trainY = pd.read_csv('train_label.csv', header=None)
            testX = pd.read_csv('test_image.csv', header=None)

        # Transpose
        trainX = trainX.T
        trainY = trainY.T
        testX = testX.T

        # pd to np
        trainX = trainX.values
        trainY = trainY.values
        testX = testX.values

        # categorical encoding
        onehotY = np.zeros((trainY.size, 10))
        onehotY[np.arange(trainY.size), trainY] = 1
        onehotY = onehotY.T

        return trainX, onehotY, testX

    # e^x/1+e^x
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    # v * s * 1 - s
    def sigmoid_backward(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)

    # 10 labels
    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def get_mini_batches(self, X, y, batch_size):
        n = X.shape[1]
        mini_batches = list()

        for i in range(0, math.floor(n / batch_size)):
            mx = X[:, i * batch_size: (i + 1) * batch_size]
            my = y[:, i * batch_size: (i + 1) * batch_size]
            mini_batch = (mx, my)
            mini_batches.append(mini_batch)

        if n % batch_size != 0:
            mx = X[:, batch_size * math.floor(n / batch_size): n]
            my = y[:, batch_size * math.floor(n / batch_size): n]
            mini_batch = (mx, my)
            mini_batches.append(mini_batch)

        return mini_batches

    # sigmoid(w*x + b)
    # last layer: softmax(w*x + b)
    def forward_pass(self, X):
        cache = dict()

        cache['Z1'] = np.dot(self.params['W1'], X) + self.params['b1']
        cache['A1'] = self.sigmoid(cache['Z1'])
        cache['Z2'] = np.dot(self.params['W2'], cache['A1']) + self.params['b2']
        cache['A2'] = self.sigmoid(cache['Z2'])
        cache['Z3'] = np.dot(self.params['W3'], cache['A2']) + self.params['b3']
        cache['A3'] = self.softmax(cache['Z3'])

        return cache

    # backward pass
    def backward_pass(self, X, Y, cache):
        m = X.shape[1]

        dZ3 = cache['A3'] - Y

        dW3 = np.dot(dZ3, cache["A2"].T) / m
        db3 = np.sum(dZ3, axis=1, keepdims=True) / m

        dA2 = np.dot(self.params['W3'].T, dZ3)
        dZ2 = self.sigmoid_backward(dA2, cache['Z2'])

        dW2 = np.dot(dZ2, cache['A1'].T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(self.params['W2'].T, dZ2)
        dZ1 = self.sigmoid_backward(dA1, cache['Z1'])

        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        return {'dW3': dW3, 'db3': db3, 'dW2': dW2, 'db2': db2, 'dW1': dW1, 'db1': db1}

    def train(self, X, y):
        # i = 0
        for i in range(0, self.num_epochs):
            # j = 0
            random = np.arange(len(X[1]))
            np.random.shuffle(random)
            X_shuffle = X[:, random]
            y_shuffle = y[:, random]

            mini_batches = self.get_mini_batches(X_shuffle, y_shuffle, self.batch_size)

            for mini_batch in mini_batches:
                mx, my = mini_batch
                cache = self.forward_pass(mx)
                grads = self.backward_pass(mx, my, cache)

                self.params['W1'] = self.params['W1'] - (self.learning_rate * grads['dW1'])
                self.params['b1'] = self.params['b1'] - (self.learning_rate * grads['db1'])
                self.params['W2'] = self.params['W2'] - (self.learning_rate * grads['dW2'])
                self.params['b2'] = self.params['b2'] - (self.learning_rate * grads['db2'])
                self.params['W3'] = self.params['W3'] - (self.learning_rate * grads['dW3'])
                self.params['b3'] = self.params['b3'] - (self.learning_rate * grads['db3'])
                # print('Over {},{}'.format(i, j))
                # j += 1
            # i += 1


# network
nn = MultiLayerPerceptron(batch_size=32, learning_rate=0.01, num_epochs=80)
trainX, onehotY, testX = nn.start()
nn.train(trainX, onehotY)
out = nn.forward_pass(testX)
output = out['A3']
pred = np.argmax(output, axis=0)
pd.DataFrame(pred).to_csv('test_predictions.csv', header=None, index=None)
