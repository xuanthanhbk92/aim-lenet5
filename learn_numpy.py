from numpy import matrix
import numpy as np
from lenet5_model import *
from load_data import *

from layers import *
from train import *

data = loaddata('dataset.npz', 5000, 1000)

# model = Lenet5ConvNet(input_dim=(1,28,28),
#                       weight_scale=1e-3, reg=10.0,
#                       dtype=np.float32)

model = Lenet5Simple(input_dim=(1,28,28),
                      weight_scale=1e-1, reg=0.0,
                      dtype=np.float32)

# model = TwoLayerNet(input_dim=784, weight_scale=1e-3, reg=0.0)

solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-1,
                },
                lr_decay=1,  #After each update, regurlarization decrease
                batch_size=10,
                num_epochs=10,
                num_train_samples=2000,
                num_val_samples=1000,
                print_every=1,
                checkpoint_name=None,
                verbose=True)
solver.train()
