import keras.activations
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.activations import relu, linear
from keras.datasets import fashion_mnist
from keras import layers
from NeuralNetworks import neural_networks
import os
import numpy as np

# ====================
#  Hyper-parameters
# ====================

NUM_EPOCHS = 50
BATCH_SIZE = 128

WORKING_DIR = os.path.join(os.path.dirname(__file__), "mnist_fashion_test_results")
if not os.path.isdir(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# ====================
#  Data preparation
# ====================

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
x, y = np.concatenate((train_x, test_x), axis=0), np.concatenate((train_y, test_y), axis=0)

train_dataset, cv_dataset, test_dataset = neural_networks.prepare_datasets(
    x, y, 
    train_split=0.75, val_split=0.20, test_split=0.05,
    normalize_image_data=True,
    add_channel=True,
    datype="float32",
    buffer_factor=1,
    batch_size=BATCH_SIZE,
    shuffle_examples=True
)


cnn_model = neural_networks.ConvolutionalNetwork(
    inp_shape=(28, 28, 1),
    num_epochs=NUM_EPOCHS,
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.0005),
    verbose=2,
)

# ====================
#   Building model
# ====================
cnn_model.make_colnvolutional_layers(
    num_layers=5,
    num_filters=(16, 32, 32, 64, 64),
    kernel_sizes=((3, 3), (3, 3), (3, 3), (3, 3), (3, 3),),
    strides=((2, 2), (2, 2), (2, 2), (2, 2), (2, 2),),
    activations=(relu, relu, relu, relu, relu),
    padding=("same", "same", "same", "same", "same"),
    drop_rate=(0.0, 0.1, 0.10, 0.10, 0.10),
    use_bias=(True, True, True, True, True),
    use_bn=(True, True, True, True, True),
    pooling_obj=(
        None,
        layers.MaxPool2D(pool_size=(2,2), strides=(2, 2), padding="same"),
        None,
        layers.MaxPool2D(pool_size=(2,2), strides=(2, 2), padding="same"),
        None,
    ),
)

cnn_model.make_dense_layers(
    num_layers=2,
    num_units=(None, 10,),
    activations=(None, relu,),
    drop_rate=(0.2, 0.0,),
    use_bias=(None, True),
)

cnn_model.build()

cnn_model.plot(path=os.path.join(
    WORKING_DIR, "CNN_Model.png"), show_shapes=True)

# ====================
#   Training model
# ====================
cnn_model.add_data_sets(train_ds=train_dataset, val_ds=cv_dataset, test_ds=test_dataset)

cnn_model.plot_examples(num_samples=25, image_path=os.path.join(
    WORKING_DIR, f"sample_examples.png"), cmap="gray")

cnn_model.train()

cnn_model.plot_training_statistics(results_dir=WORKING_DIR)

cnn_model.save_nn_model(save_dir=WORKING_DIR, fname="mnist_fashion_cnn_model")
