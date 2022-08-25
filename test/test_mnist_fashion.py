import keras.activations
import tensorflow as tf
from tensorflow import keras
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.optimizers import Adam
from keras.activations import relu
from keras.datasets import fashion_mnist
from keras import layers
from NeuralNetworks import neural_networks, dataset_ops, conv_nets, utils
import os, time
import numpy as np


# ====================
#  Hyper-parameters
# ====================

NUM_EPOCHS = 200
BATCH_SIZE = 64

WORKING_DIR = os.path.join(os.path.dirname(__file__), "_results", f"mnist_fashion_test_results-{round(time.time())}")
if not os.path.isdir(WORKING_DIR):
    os.makedirs(WORKING_DIR)


# ====================
#  Data preparation
# ====================

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
x, y = np.concatenate((train_x, test_x), axis=0), np.concatenate((train_y, test_y), axis=0)

ids = dataset_ops.ImageDataSet(x, y)
ids.normalize()
ids.set_data_type(data_type="float32")
ids.add_channel_dim()
ids.shuffle()
ids.split_train_val_test(fractions=(0.80, 0.10, 0.10))
ids.prepare_tf_datasets(batch_size=BATCH_SIZE, buffer_factor=1)
ids.summary()


# ====================
#   Building model
# ====================
#
cnn_model = conv_nets.ConvolutionalNetwork(
    inp_shape=(28, 28, 1),
    num_epochs=NUM_EPOCHS,
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=[SparseCategoricalAccuracy()],
    optimizer=Adam(learning_rate=0.001),
    verbose=2,
    working_dir=WORKING_DIR,
    model_name="MNIST-FASHION-CNN-VGG"
)


cnn_model.make_vgg_cnn(
    num_blocks=3,
    num_dense_layers=0,
    num_outputs=10,
    aug_data=True,
    cl_filters = (16, 32, 64),
    cl_droprate = (0.10, 0.20, 0.30),
    dl_units = (64,),
    dl_droprate=(0.45,),
    hflip=False,
    vflip=False,
    rot90ccw=True,
    rot180ccw=True,
    rot270ccw=True,
)
cnn_model.plot(show_shapes=True)

# ====================
#   Training model
# ====================

cnn_model.add_data_sets(
    train_ds=ids.train_ds,
    val_ds=ids.val_ds,
    test_ds=ids.test_ds
)
cnn_model.plot_examples(num_samples=16, c_map="gray", plt_size=15.0)
cnn_model.train_val_test()
cnn_model.plot_metrics()
cnn_model.write_report()
