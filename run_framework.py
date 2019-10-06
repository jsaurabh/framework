from data_loader_two_by_two import get_data_sets
from net import framework, layer, activation

train, valid = get_data_sets()

sample = next(train())
h, w = sample.shape
npix = h*w
#print([h, w])

net = [layer.Dense(npix, npix, activation.tanh), layer.Dense(npix, npix, activation.tanh)]
encoder = framework.NN(net, range = (0, 1))
print(encoder.train(train))
#print(encoder.evaluate(valid))