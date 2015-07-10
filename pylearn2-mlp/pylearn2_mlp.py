from pylearn2.models.mlp import MLP, Sigmoid, Linear
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.termination_criteria import EpochCounter

import theano
import numpy as np

n = 200
p = 2
X = np.random.normal(0, 1, (n, p))
y = X[:,0]* X[:, 1] + np.random.normal(0, .1, n)
y.shape = (n, 1)

ds = DenseDesignMatrix(X=X, y=y)

hidden_layer = Sigmoid(layer_name='hidden', dim=10, irange=.1, init_bias=1.)
output_layer = Linear(dim=1, layer_name='y', irange=.1)
trainer = SGD(learning_rate=.05, batch_size=10,
              termination_criterion=EpochCounter(200))
layers = [hidden_layer, output_layer]
ann = MLP(layers, nvis=2)
trainer.setup(ann, ds)

while True:
    trainer.train(dataset=ds)
    ann.monitor.report_epoch()
    ann.monitor()
    if not trainer.continue_learning(ann):
        break

inputs = X 
y_est = ann.fprop(theano.shared(inputs, name='inputs')).eval()

print(y_est.shape)
