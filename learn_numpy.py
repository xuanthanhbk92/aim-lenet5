from numpy import matrix
import numpy as np
from lenet5_model import *
from load_data import *

from layers import *
from train import *

data = loaddata('dataset.npz', 200, 100)

model = Lenet5ConvNet(input_dim=(1,28,28),
                      weight_scale=1e-3, reg=10.0,
                      dtype=np.float32)

solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 0.1,
                },
                lr_decay=1,  #After each update, regurlarization decrease
                batch_size=1,
                num_epochs=10,
                num_train_samples=200,
                num_val_samples=100,
                print_every=1,
                checkpoint_name=None,
                verbose=True)
solver.train()
