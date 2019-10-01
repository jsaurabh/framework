from data_loader_two_by_two import get_data_sets
from net import framework

train, valid = get_data_sets()

sample = next(train())
h, w = sample.shape
#print([h, w])

network = framework.NN(10, range = (-.5, .5))
print(network.train(train))
print(network.evaluate(valid))