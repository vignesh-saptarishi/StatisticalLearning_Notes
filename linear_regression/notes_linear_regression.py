def cost_fn(X, y, beta):
    return ((X.dot(beta) - y) ** 2).sum() / 2 / len(y)

def gradient_descent(X, y, beta, gamma, iters):
    cost_hist = numpy.zeros(iters)
    for i in range(iters):
        hypothesis = X.dot(beta)
        loss = hypothesis - y
        gradient = X.T.dot(loss) / len(y)
        new_beta = beta - gamma*gradient
        beta = new_beta
        cost_hist[i] = cost_fn(X, y, beta)
    return beta, cost_hist


import pandas
import numpy
import patsy
import matplotlib.pyplot as plt
%matplotlib inline


data = pandas.read_csv('ex1_simple.txt', names=['X', 'Y'])
print(data.sample(5))

fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(data['X'], data['Y'])
ax.set_xlabel('X')
ax.set_ylabel('Y')


dmat = patsy.dmatrices('Y ~ X', data)
X = numpy.array(dmat[1])
y = numpy.array(dmat[0])

iters = 2000
gamma = 0.01
beta_init = numpy.array([[0], [0]])
betas, cost_history = gradient_descent(X, y, beta_init, gamma, iters)
print(betas)


y_preds = X.dot(betas)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].scatter(data['X'], data['Y'], alpha=0.25)
axes[0].plot(data['X'], y_preds, color='red', linewidth=2)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

axes[1].plot(range(len(cost_history)), cost_history)
axes[1].set_xlabel('iteration #')
axes[1].set_ylabel('cost function')


def OLS(X, Y):
    return numpy.linalg.inv(numpy.dot(X.T, X)) @ X.T @ Y


data = pandas.read_csv('ex2.csv')
print(data.sample(5))


dmat = patsy.dmatrices('Sales ~ TV + Radio + Newspaper', data)
X = numpy.array(dmat[1])
y = numpy.array(dmat[0])

betas = OLS(X, y)
print(betas)


y_pred = X.dot(betas)


