import numpy as np

def get_data_sets():
    """Sample dataloader"""

    examples = [
        np.array([[0, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[0, 0], [1, 0]]),
        np.array([[0, 1], [1, 1]]),
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 1], [0, 1]]),
    ]

    def get_train():
        while True:
            idx = np.random.choice(len(examples))
            yield examples[idx]

    def get_valid():
        while True:
            idx = np.random.choice(len(examples))
            yield examples[idx]

    return get_train, get_valid
