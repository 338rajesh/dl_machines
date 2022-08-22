import keras.activations
import tensorflow as tf
from tensorflow import keras
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.optimizers import Adam
from keras.activations import relu
from keras.datasets import mnist
from keras import layers
from NeuralNetworks import neural_networks, dataset_ops, conv_nets, utils
import os, time
import numpy as np

# ====================
#  Hyper-parameters
# ====================

NUM_EPOCHS = 2000
BATCH_SIZE = 64
WORKING_DIR = os.path.join(
    os.path.dirname(__file__),
    "_results",
    f"mnist_digits_test_results-{round(time.time())}"
)
if not os.path.isdir(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# ====================
#  Data preparation
# ====================

(train_x, train_y), (test_x, test_y) = mnist.load_data()
x, y = np.concatenate((train_x, test_x), axis=0), np.concatenate((train_y, test_y), axis=0)

ids = dataset_ops.ImageDataSet(x, y, name="MNIST DIGITS")
ids.normalize()
ids.set_data_type(datype="float32")
ids.add_channel_dim()
ids.shuffle()
ids.split_train_val_test(fractions=(0.835, 0.155, 0.01))
ids.prepare_tf_datasets(batch_size=BATCH_SIZE, buffer_factor=1)
ids.summary()


# ==============================
#   Building & Training model
# ==============================
#
cnn_model = conv_nets.ConvolutionalNetwork(
    inp_shape=(28, 28, 1),
    num_epochs=NUM_EPOCHS,
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=[SparseCategoricalAccuracy(),],
    optimizer=Adam(learning_rate=0.001),
    verbose=2,
    working_dir=WORKING_DIR,
    model_name="MNIST-DIGITS-CNN-VGG",
)
cnn_model.make_vgg_cnn(
    num_blocks=2,
    num_dense_layers=0,
    num_outputs=10,
    cl_filters = (8, 16),
    cl_droprate = (0.15, 0.30),
    dl_units = (32,),
    dl_droprate=(0.30,),
    )
cnn_model.plot(show_shapes=True)
cnn_model.add_data_sets(ids)
cnn_model.plot_examples(num_samples=16, cmap="gray", plt_size=15.0)
cnn_model.train_val_test()
cnn_model.plot_metrics()
cnn_model.write_report()
