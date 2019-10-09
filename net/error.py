import numpy as np

class MSE(object):
    
    @staticmethod 
    def calc(y, yhat):
        return np.pow(yhat - y, 2)
    
    @staticmethod
    def calc_derivative(y, yhat):
        return 2 * (yhat - y)