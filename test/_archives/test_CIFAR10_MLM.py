from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras import Model
import keras.activations
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam, SGD
from keras.activations import relu
from keras.datasets import cifar10
from keras import layers
from NeuralNetworks import neural_networks, dataset_ops, conv_nets
import os
import numpy as np

# ====================
#  Hyper-parameters
# ====================

NUM_EPOCHS = 400
BATCH_SIZE = 64

WORKING_DIR = os.path.join(os.path.dirname(__file__), "CIFAR10_test_results")
if not os.path.isdir(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# ====================
#  Data preparation
# ====================

(train_x, train_y), (test_x, test_y) = cifar10.load_data()
x, y = np.concatenate((train_x, test_x), axis=0), np.concatenate(
    (train_y, test_y), axis=0)

ids = dataset_ops.ImageDataSet(x, y)
ids.normalize()
ids.set_data_type(data_type="float32")
ids.add_channel_dim()
ids.shuffle()
ids.split_train_val_test(fractions=(0.835, 0.155, 0.01))
ids.prepare_tf_datasets(batch_size=BATCH_SIZE, buffer_factor=1)
ids.summary()


# ====================
#   Build Model
# ====================

cnn_model = conv_nets.ConvolutionalNetwork(
    inp_shape=(32, 32, 3),
    num_epochs=NUM_EPOCHS,
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=SGD(learning_rate=0.001, momentum=0.9),
    verbose=2,
)


def define_cnn_model():
    #
    img_inp = layers.Input(shape=(32, 32, 3))
    img_inp = dataset_ops.DataAugmentationLayer(
        True, True, True, True, True)(img_inp)
    #
    # ==============
    #  VGG Block-1
    # ==============
    x = layers.Conv2D(
        32, (3, 3), kernel_initializer="he_uniform", padding="same")(img_inp)
    x = layers.BatchNormalization()(x)
    x = layers.activation.ReLU()(x)
    #
    x = layers.Conv2D(
        32, (3, 3), kernel_initializer="he_uniform", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.activation.ReLU()(x)
    #
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Dropout(rate=0.2)(x)
    #
    # ==============
    #  VGG Block-2
    # ==============
    x = layers.Conv2D(
        64, (3, 3), kernel_initializer="he_uniform", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.activation.ReLU()(x)
    #
    x = layers.Conv2D(
        64, (3, 3), kernel_initializer="he_uniform", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.activation.ReLU()(x)
    #
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Dropout(rate=0.3)(x)
    #
    # ==============
    #  VGG Block-3
    # ==============
    x = layers.Conv2D(
        128, (3, 3), kernel_initializer="he_uniform", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.activation.ReLU()(x)
    #
    x = layers.Conv2D(
        128, (3, 3), kernel_initializer="he_uniform", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.activation.ReLU()(x)
    #
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Dropout(rate=0.4)(x)
    #
    # =============
    #  Dense Layer
    # =============
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, kernel_initializer="he_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.activation.ReLU()(x)
    x = layers.Dropout(rate=0.5)(x)
    #
    output_layer = layers.Dense(units=10,)(x)
    model = Model(img_inp, output_layer, name="CNN-model")
    return model


cnn_model.load_model(model=define_cnn_model())

cnn_model.plot(
    path=os.path.join(WORKING_DIR, "CNN_Model.png"),
    show_shapes=True
)


# ====================
#   Training model
# ====================
cnn_model.add_data_sets(
    train_ds=ids.train_ds,
    val_ds=ids.val_ds,
    test_ds=ids.test_ds
)

cnn_model.plot_examples(num_samples=25, image_path=os.path.join(
    WORKING_DIR, f"sample_examples.png"))

cnn_model.train()

cnn_model.plot_training_statistics(results_dir=WORKING_DIR)
