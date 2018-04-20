from solver import *
from load_data import *
from lenet5_model import *

data = loaddata('dataset.npz')

model = Lenet5ConvNet(input_dim=(1,28,28),
                      weight_scale=1e-1, reg=0.0,
                      dtype=np.float32)

# model = Lenet5Simple(input_dim=(1,28,28),
#                       weight_scale=1e-1, reg=0.0,
#                       dtype=np.float32)
# # model = TwoLayerNet(input_dim=784, weight_scale=1e-3, reg=0.0)
#
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-1,
                },
                lr_decay=1,  #After each update, regurlarization decrease
                batch_size=100,
                num_epochs=10,
                num_train_samples=None,
                num_val_samples=None,
                print_every=1,
                checkpoint_name=None,
                verbose=True)

solver.train()

