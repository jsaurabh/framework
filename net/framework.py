import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class NN:
    def __init__(self, layers, range = (-1, 1)):
        """Initialize neural network"""
        self.layers = layers
        self.constant = 10
        self.range = range
        self.iter = int(1e5)
        self.history = []
        self.interval = int(1e4) ##interval for report update
        self.report_path = Path.cwd()/'report'
        try:
            Path(self.report_path).mkdir(exist_ok = True, parents=True)
        except Exception:
            print("Error occured. ")

    def train(self, train_batch):
        """Training loop"""
        for itr in range(self.iter):
            x = self.normalize(next(train_batch()).ravel())
            y = self.forward(x)
            print(y)
            self.history.append(1)

            if (itr+1) % self.interval == 0:
                self.report()
    
    def evaluate(self, valid_batch):
        """Evaluation loop"""
        for itr in range(self.iter):
            x = self.normalize(next(valid_batch()).ravel())
            y = self.forward(x)
            print(y)
            self.history.append(1)

            if (itr+1) % self.interval == 0:
                self.report()
            
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

    def report(self):
        bins = int(len(self.history)) // self.interval
        histories = []
        for bin in range(bins):
            avg_err_history = np.mean(self.history[bin * 
                self.interval:(bin+1) * self.interval])
            histories.append(avg_err_history)
            
        ## Highlight and scale small differences. 
        ## Add 1e-10 so loop doesn't break for 0 error
        error_histories = np.log10(np.array(histories) + 1e-10)

        # Zoom in on interesting bits of loss landscape
        ymin = min(-3, np.min(error_histories))
        ymax = max(2, np.max(error_histories))

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(error_histories)
        ax.set_ylabel("Error")
        ax.set_xlabel("x {0} iterations".format(self.interval))
        ax.set_ylim(ymin, ymax)
        ax.grid()
        try:
            fig.savefig('report/eror.png')
        except Exception:
            print("Plotting Error")

        plt.close()
                   