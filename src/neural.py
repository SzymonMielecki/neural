import numpy as np
import time


def printProgress(epoch, totalEpochs, loss, timeTaken):
    barLength = 30
    progress = int(barLength * epoch / totalEpochs)
    eta = int(timeTaken * (totalEpochs - epoch))

    progress_bar = "[" + "=" * progress + \
        ">" + "." * (barLength - progress) + "]"
    print(
        f"\r{epoch}/{totalEpochs} {progress_bar} - ETA: {eta}s - loss: {loss:.4f}", end='')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class NN:
    def __init__(self):
        self.weights = np.random.randn(6)
        self.biases = np.random.randn(3)

    def feedforward(self, x):
        h1 = sigmoid(self.weights[0] * x[0] +
                    self.weights[1] * x[1] + self.biases[0])
        h2 = sigmoid(self.weights[2] * x[0] +
                    self.weights[4] * x[1] + self.biases[1])
        o1 = sigmoid(self.weights[4] * h1 +
                    self.weights[5] * h2 + self.biases[2])
        return o1
    
    def train(self, data, all_y_trues, learn_rate, epochs):

        for epoch in range(epochs):
            start = time.time()
            for x, y_true in zip(data, all_y_trues):

                sum_h1 = self.weights[0] * x[0] + \
                    self.weights[1] * x[1] + self.biases[0]
                h1 = sigmoid(sum_h1)

                sum_h2 = self.weights[2] * x[0] + \
                    self.weights[3] * x[1] + self.biases[1]
                h2 = sigmoid(sum_h2)

                sum_o1 = self.weights[4] * h1 + \
                    self.weights[5] * h2 + self.biases[2]
                o1 = sigmoid(sum_o1)
                y_pred = o1

                dMSE = -2 * (y_true - y_pred)

                dw5 = h1 * deriv_sigmoid(sum_o1)
                dw6 = h2 * deriv_sigmoid(sum_o1)
                db3 = deriv_sigmoid(sum_o1)

                dh1 = self.weights[4] * deriv_sigmoid(sum_o1)
                dh2 = self.weights[5] * deriv_sigmoid(sum_o1)

                dw1 = x[0] * deriv_sigmoid(sum_h1)
                dw2 = x[1] * deriv_sigmoid(sum_h1)
                db1 = deriv_sigmoid(sum_h1)

                dw3 = x[0] * deriv_sigmoid(sum_h2)
                dw4 = x[1] * deriv_sigmoid(sum_h2)
                db2 = deriv_sigmoid(sum_h2)

                self.weights[0] -= learn_rate * dMSE * dh1 * dw1
                self.weights[1] -= learn_rate * dMSE * dh1 * dw2
                self.biases[0] -= learn_rate * dMSE * dh1 * db1

                self.weights[2] -= learn_rate * dMSE * dh2 * dw3
                self.weights[3] -= learn_rate * dMSE * dh2 * dw4
                self.biases[1] -= learn_rate * dMSE * dh2 * db2

                self.weights[4] -= learn_rate * dMSE * dw5
                self.weights[5] -= learn_rate * dMSE * dw6
                self.biases[2] -= learn_rate * dMSE * db3

            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(all_y_trues, y_preds)
            end = time.time()
            printProgress(epoch, epochs, loss, end-start)
    def evaluate(self, X, y):
        yPreds = []
        for x in X:
            result = self.feedforward(x)
            yPreds.append(result)
        print(f"\n {yPreds}")
        return mse_loss(y, yPreds)

data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

all_y_trues = np.array([
    0,
    0,
    0,
    1,
])

learn_rate = 0.1
epochs = 10000

network = NN()
network.train(data, all_y_trues, learn_rate, epochs)