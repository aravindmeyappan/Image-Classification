import numpy as np 

"""
Generally any model will have 4 components:
1. The hyperparameters (defined by init or initiation function)
2. The fitting (estimating the model parameters using the given data)
3. Updating parameters
4. Predicting the new unseen examples
"""

class linear_svm_classifier():

    def __init__(self, learning_rate, num_iter, reg_lambda, thres=-1):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.reg_lambda = reg_lambda
        self.thres = thres
    
    def fit(self, X, Y):

        # getting the shape of x in order to find out the dimension of weights to initialize
        self.m, self.n = X.shape # m --> number of examples, n --> number of features in input data

        # initialize weights
        self.w = np.zeros(self.n) # w is a column vector of dimension n
        self.b = 0 # bias can be assigned a scalar and while computation it can be broadcasted according to dimension of w.T*x
        self.X = X
        self.Y = Y

        # gradient descent implementation
        for i in range(self.num_iter):
            self.update_weights()

    def update_weights(self):

        ## Label Encoding
        # In svm classifier instead of the traditional approach of giving 0 to absence and 1 to presence of a phenomena
        # we will encode it to 1 for presence and -1 for absence.
        y_label = np.where(self.Y<=0, -1, 1)

        # initialize gradients
        dw = np.zeros(self.n)
        db = 0

        # computing gradients at each pass
        for idx,xi in enumerate(self.X): # iterating through each example(n dimensional vector) and its index (to check label)
            condition = y_label[idx]*(np.dot(xi,self.w)-self.b) >= 1
            if not condition:
                dw += -np.dot(xi,y_label[idx]) 
                db += y_label[idx]

        # normalize the weights
        dw /= self.m

        # add the regularization 
        dw += 2*self.reg_lambda*self.w

        # update the parameters
        self.w -= self.learning_rate*dw
        self.b -= self.learning_rate*db

    def predict(self, X, threshold=None):
        if threshold is None:
            threshold = self.thres

        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels<=threshold, 0, 1)
        return y_hat