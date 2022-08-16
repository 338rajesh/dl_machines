import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.activations import relu, leaky_relu, linear
from keras import layers

from keras.datasets import boston_housing

import os, time

from NeuralNetworks import neural_networks as nn

# ====================
#  Hyper-parameters
# ====================

NUM_EPOCHS = 10
BUFFER = 60000
BATCH_SIZE = 64

WORKING_DIR = os.path.join(os.path.dirname(__file__), "mnist_test_results")
if not os.path.isdir(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# ====================
#  Data preparation
# ====================

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x, test_x = train_x / 255.0, test_x / 255.0
train_x = train_x[..., tf.newaxis].astype("float32")
test_x = test_x[..., tf.newaxis].astype("float32")
train_y = train_y.astype("float32").reshape((-1, 1))
test_y = test_y.astype("float32").reshape((-1, 1))
train_dataset = tf.data.Dataset.from_tensor_slices(
    tensors=(train_x, train_y), ).shuffle(BUFFER).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(
    tensors=(test_x, test_y), ).shuffle(BUFFER).batch(BATCH_SIZE)

cnn_model = neural_networks.ConvolutionalNetwork(
    inp_shape=(28, 28, 1),
    num_epochs=NUM_EPOCHS,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.001),
    verbose=2,
)

cnn_model.build(
    num_layers=3,
    num_filters=(16, 32, 64,),
    kernel_sizes=((2, 2), (2, 2), (2, 2),),
    strides=((2, 2), (2, 2), (1, 1),),
    activations=(keras.activations.relu, keras.activations.relu,
                 keras.activations.relu,),
    padding=("same", "same", "same",),
    drop_rate=(0.0, 0.2, 0.2),
    use_bias=(False, False, False),
    use_bn=(False, True, True),
)


cnn_model.plot(path=os.path.join(
    WORKING_DIR, "CNN_Model.png"), show_shapes=True)
cnn_model.plot_examples(num_samples=25, image_path=os.path.join(
    WORKING_DIR, f"sample_examples.png"))

cnn_model.train()