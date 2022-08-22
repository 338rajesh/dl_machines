import keras.activations
import tensorflow as tf
from tensorflow import keras
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.optimizers import Adam
from keras.activations import relu
from keras.datasets import cifar10
from keras import layers
from NeuralNetworks import neural_networks, dataset_ops, conv_nets
import os, time
import numpy as np

# ====================
#  Hyper-parameters
# ====================

NUM_EPOCHS = 50
BATCH_SIZE = 256

WORKING_DIR = os.path.join(os.path.dirname(__file__), "_results", f"CIFAR10_test_results-{round(time.time())}")
if not os.path.isdir(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# ====================
#  Data preparation
# ====================

(train_x, train_y), (test_x, test_y) = cifar10.load_data()
x, y = np.concatenate((train_x, test_x), axis=0), np.concatenate((train_y, test_y), axis=0)

ids = dataset_ops.ImageDataSet(x, y)
ids.normalize()
ids.set_data_type(datype="float32")
ids.shuffle()
ids.split_train_val_test(fractions=(0.80, 0.10, 0.10))
ids.prepare_tf_datasets(batch_size=BATCH_SIZE, buffer_factor=1)
ids.summary()


# ====================
#   Building model
# ====================
#
cnn_model = conv_nets.ConvolutionalNetwork(
    inp_shape=(32, 32, 3),
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
cnn_model.plot_examples(num_samples=16, cmap="gray", plt_size=15.0)
cnn_model.train_val_test()
cnn_model.plot_metrics()
cnn_model.write_report()

# cnn_model.make_colnvolutional_layers(
#     num_layers=6,
#     num_filters=(32, 32, 64, 64, 128, 128,),
#     kernel_sizes=((3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3),),
#     strides=((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1),),
#     activations=(relu, relu, relu, relu, relu, relu,),
#     padding=("same", "same", "same", "same", "same", "same",),
#     drop_rate=(0.0, 0.2, 0.0, 0.3, 0.0, 0.4,),
#     use_bias=(True, True, True, True, True, True,),
#     use_bn=(False, True, False, True, False, True,),
#     pooling_obj=(
#         None,
#         layers.MaxPool2D(pool_size=(2,2), strides=(2, 2), padding="same"),
#         None,
#         layers.MaxPool2D(pool_size=(2,2), strides=(2, 2), padding="same"),
#         None,
#         layers.MaxPool2D(pool_size=(2,2), strides=(2, 2), padding="same"),
#     )
# )


# cnn_model.make_dense_layers(
#     num_layers=3,
#     num_units=(None, 128, 10,),
#     activations=(None, relu, None,),
#     drop_rate=(0.0, 0.5, 0.0,),
#     use_bias=(True, True, True),
#     use_bn=(False, True, False)
# )

# cnn_model.build()

# cnn_model.plot(
#     path=os.path.join(WORKING_DIR, "CNN_Model.png"),
#     show_shapes=True
# )


# # ====================
# #   Training model
# # ====================
# cnn_model.add_data_sets(train_ds=train_dataset, val_ds=cv_dataset, test_ds=test_dataset)

# cnn_model.plot_examples(num_samples=25, image_path=os.path.join(
#     WORKING_DIR, f"sample_examples.png"))

# cnn_model.train()

# cnn_model.plot_training_statistics(results_dir=WORKING_DIR)