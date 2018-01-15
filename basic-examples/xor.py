# Exercise from 3nd lesson in DL course at Amat
import numpy as np

epochs = 60000  # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 3, 3, 1
# Last value in each entry is the bias
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# Labels
Y = np.array([[0], [1], [1], [0]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_(x):
    return sigmoid(x) * (1 - sigmoid(x))


# weights on layer inputs
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))
for i in range(epochs):
    hidden_input = np.dot(X, Wh)
    H = sigmoid(hidden_input)  # hidden layer results
    out_input = np.dot(H, Wz)
    Z = sigmoid(out_input)  # output layer results
    E = Y - Z  # how much we missed (error)
    dZ = E * sigmoid_(out_input)
    dH = np.dot(np.dot(dZ.T, sigmoid_(hidden_input)), Wh)
    Wz += (np.dot(dZ.T, H)).T
    Wh += np.dot(dH, X.T)
print(Z)  # what have we learnt?

