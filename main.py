from cgi import test
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
Minimal implementation of logistic regression using batch gradient descent
'''

def sigmoid(x):
    # trick to avoid overflow [https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation]
    x = np.clip(x,-100,100)
    return 1.0/(1+np.exp(-x))

def logloss(X,Y,theta,zeta=1e-6):
    # zeta gives numerical stability to the log
    # vectorized
    v = -np.multiply(Y,np.log(sigmoid(np.matmul(X,theta))+zeta)) - np.multiply(1-Y,np.log(1-sigmoid(np.matmul(X,theta))+zeta))
    logloss = v.sum()
    return logloss

def train(X, Y, learning_rate=1e-6):
    max_epochs = 5000
    theta = np.random.randn(X.shape[1]) # initialize theta arbitrarily (here I'm using a normal distribution)
    losses = []
    for epoch in range(max_epochs):
        # vectorized
        grad = -np.matmul(X.transpose(), Y-sigmoid(np.matmul(X,theta)))
        theta = theta - learning_rate * grad
        losses.append(logloss(X,Y,theta))
    print(f"trained for {epoch+1} epochs")
    return theta, losses

def main():
    data,target = load_breast_cancer(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.3, random_state=1)
    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    scaler.transform(X_test)
    # loss_history can be used for debugging
    theta, loss_history = train(X_train, Y_train)
    Y_test_hat = (sigmoid(X_test@theta)>=0.5).astype('uint8')
    test_acc = np.sum(Y_test==Y_test_hat)/Y_test.shape[0]
    print(f"test accuracy: {test_acc}")

if __name__ == '__main__':
    main()