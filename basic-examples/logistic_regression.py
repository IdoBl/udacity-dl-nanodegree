# Exercise from 2nd lesson in DL course at Amat

import numpy as np
import matplotlib.pylab as plt

np.random.seed(12)
num_observation = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observation)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observation)
# Train set
x = np.vstack((x1, x2)).astype(np.float32)
print('Train set shape : {}'.format(x.shape))
# Labels
y = np.hstack((np.zeros(num_observation), np.ones(num_observation)))
print('Labels set shape: {}'.format(y.shape))
plt.figure(figsize=(12, 4))
plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.4)
plt.show()


def sigmoid(wx):
    return 1 / (1 + np.exp(-wx))


def log_likelihood(x, y, w):
    return np.dot(y, np.dot(x, w)) - np.log(1 + np.exp(np.dot(x, w)))


def logistic_regression(x, y, num_setps, learning_rate, add_interxept=False):
    if (add_interxept):
        intercept = np.ones((x.shape[0], 1))
        x = np.hstack((intercept, x))

    w = np.zeros(x.shape[1])

    for step in range(num_setps):
        xw = np.dot(x, w)
        est_y = sigmoid(xw)
        err = y - est_y
        gradient = np.dot(x.T, err)
        w += learning_rate * gradient

        if step % 10000 == 0:
            print('Step num : {} , Log likelihood : {}'.format(step, log_likelihood(x, y, w)))

    return w


weights = logistic_regression(x, y, num_setps=100000, learning_rate=5e-5, add_interxept=True)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x, y)

print(clf.intercept_, clf.coef_)
print(weights)

data_with_intercept = np.hstack((np.ones((x.shape[0], 1)), x))
final_scores = np.dot(data_with_intercept, weights)
preds = np.round(sigmoid(final_scores))

print('Accuracy from scratch: {0}'.format((preds == y).sum().astype(float) / len(preds)))
print('Accuracy from sk-learn: {0}'.format(clf.score(x, y)))

plt.figure(figsize=(12, 8))
plt.scatter(x[:, 0], x[:, 1],
            c=(preds == y) - 1, alpha=.8, s=50)
plt.show()





