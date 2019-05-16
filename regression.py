import numpy as np
class LinearRegression:
    x = None
    theta = None
    y = None

    def fit(self, x, y):
        eta = 1e-5 # break condition with sum of error
        alpha = 0.01 # learning rate
        epochs = 10000 # break condition with the number of updating
        if self.theta == None:
            self.theta = np.zeros(x.shape[1])
        for i in range(epochs):
            error = self.predict(x) - y
            if sum(abs(error)) < eta:
                break
            temp_theta0 = self.theta[0] - alpha * (-1/len(x)) * sum(error)
            #temp_theta1 = self.theta[1] - alpha * (-1/len(x)) * sum(np.dot(x.T, error))
            temp_thetax = self.theta - alpha * (1/len(x)) * sum(np.dot(x.T, error))
            self.theta = temp_thetax
            self.theta[0] = temp_theta0
            #self.theta[1] = temp_theta1
            print(i, self.theta, sum(error))

    def predict(self, x):
        return np.dot(x, self.theta)

    def score(self, x, y):
        error = self.predict(x) - y
        return (error**2).sum()

class RidgeRegression(LinearRegression):
    alpha = None

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, input, output):
        xTx = np.dot(input.T, input)
        I = np.eye(len(xTx))
        self.theta = np.dot(np.dot(np.linalg.inv(xTx + self.alpha*I), input.T), output)

    # for scikit-learn
    def get_params(self, deep=True):
        return {'alpha':self.alpha}

    # (OPTION) the coefficient of determination R^2.
    # see sklearn.linear_model.Ridge().score()
    # http://goo.gl/v93tNM
    def score2(self, input, output):
        u = ((output - self.predict(input)) ** 2).sum()
        v = ((output - output.mean()) ** 2).sum()
        return (1 - u/v)