from data_loader_two_by_two import get_data_sets
from net import framework, layer, activation

train, valid = get_data_sets()

sample = next(train())
h, w = sample.shape
npix = h*w
NODES = [7, 4, 6]
model = []
nodes = [npix] + NODES + [npix]

for layers in range(len(nodes)-1):
    print(layers)
    model.append(layer.Dense(
        nodes[layers],
        nodes[layers+1],
        activation.tanh,
    ))
    print(model)

encoder = framework.NN(model, range = (0, 1))
print(encoder.train(train))
#print(encoder.evaluate(valid))