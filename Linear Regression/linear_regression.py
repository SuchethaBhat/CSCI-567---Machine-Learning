import numpy as np
import pandas as pd


############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """

    err = np.sum(np.subtract(np.dot(X, w), y) ** 2)/X.shape[0]
    return err


###### Part 1.2 ######
def linear_regression_noreg(X, y):
    """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """

    co_variance = np.dot(X.T,X)
    w = np.dot(np.dot(np.linalg.inv(co_variance + (lambd * np.identity(co_variance.shape[0]))),X.T),y)
    return w


###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    bestlambda = None
    best_msqe = 999
    for i in range(0,15):
        current_Lambda = 2**(-1*i)
        model_w = regularized_linear_regression(Xtrain,ytrain,current_Lambda)
        msqe = mean_square_error(model_w,Xval,yval)
        if msqe < best_msqe:
            best_msqe = msqe
            bestlambda = current_Lambda
            # print(i)
    return bestlambda


###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    Z = X
    for i in range(2,p+1):
        X = np.concatenate((X,Z**i),axis=1)
    return X


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""
