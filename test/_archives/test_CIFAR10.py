import keras.activations
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam, SGD
from keras.activations import relu
from keras.datasets import cifar10
from keras import layers
from NeuralNetworks import neural_networks
import os
import numpy as np

# ====================
#  Hyper-parameters
# ====================

NUM_EPOCHS = 150
BATCH_SIZE = 256

WORKING_DIR = os.path.join(os.path.dirname(__file__), "CIFAR10_test_results")
if not os.path.isdir(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# ====================
#  Data preparation
# ====================

(train_x, train_y), (test_x, test_y) = cifar10.load_data()
x, y = np.concatenate((train_x, test_x), axis=0), np.concatenate((train_y, test_y), axis=0)

train_dataset, cv_dataset, test_dataset = neural_networks.prepare_datasets(
    x, y, 
    train_split=0.80, val_split=0.10, test_split=0.10,
    normalize_image_data=True,
    add_channel=True,
    datype="float32",
    buffer_factor=1,
    batch_size=BATCH_SIZE,
    shuffle_examples=True
)



cnn_model = neural_networks.ConvolutionalNetwork(
    inp_shape=(32, 32, 3),
    num_epochs=NUM_EPOCHS,
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    verbose=2,
)

cnn_model.make_colnvolutional_layers(
    num_layers=6,
    num_filters=(32, 32, 64, 64, 128, 128,),
    kernel_sizes=((3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3),),
    strides=((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1),),
    activations=(relu, relu, relu, relu, relu, relu,),
    padding=("same", "same", "same", "same", "same", "same",),
    drop_rate=(0.0, 0.2, 0.0, 0.3, 0.0, 0.4,),
    use_bias=(True, True, True, True, True, True,),
    use_bn=(False, True, False, True, False, True,),
    pooling_obj=(
        None,
        layers.MaxPool2D(pool_size=(2,2), strides=(2, 2), padding="same"),
        None,
        layers.MaxPool2D(pool_size=(2,2), strides=(2, 2), padding="same"),
        None,
        layers.MaxPool2D(pool_size=(2,2), strides=(2, 2), padding="same"),
    )
)


cnn_model.make_dense_layers(
    num_layers=3,
    num_units=(None, 128, 10,),
    activations=(None, relu, None,),
    drop_rate=(0.0, 0.5, 0.0,),
    use_bias=(True, True, True),
    use_bn=(False, True, False)
)

cnn_model.build()

cnn_model.plot(
    path=os.path.join(WORKING_DIR, "CNN_Model.png"),
    show_shapes=True
)


# ====================
#   Training model
# ====================
cnn_model.add_data_sets(train_ds=train_dataset, val_ds=cv_dataset, test_ds=test_dataset)

cnn_model.plot_examples(num_samples=25, image_path=os.path.join(
    WORKING_DIR, f"sample_examples.png"))

cnn_model.train()

cnn_model.plot_training_statistics(results_dir=WORKING_DIR)