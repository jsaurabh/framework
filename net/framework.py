import numpy as np

class NN:
    def __init__(self, layers, range = (-1, 1)):
        """Initialize neural network"""
        self.layers = layers
        self.constant = 10
        self.range = range

    def train(self, train_batch):
        """Training loop"""
        for _ in range(self.constant):
            x = self.normalize(next(train_batch()).ravel())
            y = self.forward(x)
            print(y)
    
    def evaluate(self, valid_batch):
        """Evaluation loop"""
        for _ in range(self.constant):
            x = self.normalize(next(valid_batch()).ravel())
            y = self.forward(x)
            print(y)

    def normalize(self, value):
        """Normalize input values
        Normalized input range: -.5 to +.5"""
        
        min_val = self.range[0]
        max_val = self.range[1]

        scale = max_val - min_val
        val = (value - min_val)/ (scale) - 0.5
        return val

    def denorm(self, value):
        """Denormalize input values"""
        min_val = self.range[0]
        max_val = self.range[1]

        scale = 2/(max_val - min_val)
        return (value+0.5) * scale + min_val - 1

    def forward(self, inp):
        y = inp.ravel()[np.newaxis, :]
        for layer in self.layers:
            y = layer.forward(y)
        return y.ravel()