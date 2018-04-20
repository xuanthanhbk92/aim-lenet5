from solver import *
from load_data import *
from lenet5_model import *

# createCompressedData('dataset.npz')

data = loaddata('dataset.npz')

model = Lenet5ConvNet(input_dim=(1,28,28),
                      weight_scale=1e-1, reg=1e-2,
                      dtype=np.float32)

solver = Solver(model, data,
                update_rule='sgd_momentum',
                optim_config={
                  'learning_rate': 1e-3,
                },
                lr_decay=1,  #After each update, regurlarization decrease
                batch_size=100,
                num_epochs=10,
                num_train_samples=None,
                num_val_samples=None,
                print_every=10,
                checkpoint_name='lenet5',
                verbose=True)

solver.train()

solver.load_model('lenet5_epoch_10.pkl')
solver.test_accuracy()
