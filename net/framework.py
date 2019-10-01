class NN:
    def __init__(self, layers, range):
        """Initialize neural network"""
        self.layers = layers
        self.constant = 10
        self.range = range

    def train(self, train_batch):
        """Training loop"""
        for _ in range(self.layers):
            x = next(train_batch()).ravel()
            self.normalize(self.range, x)
            print(x)
    
    def evaluate(self, valid_batch):
        """Evaluation loop"""
        for _ in range(self.layers):
            x = next(valid_batch()).ravel()
            self.normalize(self.range, x)
            print(x)

    def normalize(self, range, example):
        pass