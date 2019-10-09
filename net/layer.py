import numpy as np

class Dense(object):

    def __init__(self, ninput, noutput, af):
        self.ninput = int(ninput)
        self.noutput = int(noutput)
        self.activate = af

        self.weights = np.random.random_sample(
            size = (self.ninput+1, self.noutput)) * 2 -1
        self.x = np.zeros((1, self.ninput+1))
        self.y = np.zeros((1, self.noutput))

    def forward(self, inp):
        bias = np.ones((1, 1))
        self.x = np.concatenate((inp, bias), axis = 1)
        self.y = self.activate.calc(self.x @ self.weights)
        return self.y

