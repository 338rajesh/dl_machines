import keras.activations
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.activations import relu, linear
from keras import layers
from keras.losses import MeanAbsolutePercentageError
from NeuralNetworks import dataset_ops, conv_nets, utils
import os
import numpy as np
from data_sets_library import benchmarks

# ====================
#  Hyper-parameters
# ====================

NUM_EPOCHS = 2
BATCH_SIZE = 64

WORKING_DIR = os.path.join(os.path.dirname(__file__), "age_prediction_test_results")
if not os.path.isdir(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# ====================
#  Data preparation
# ====================

print(f"Loading data >>", end="", flush=True)
x, y = benchmarks.load_20to50_400SamplesPerAge()
print(f"\tDone!")

utils.nparray_summary({"Images": x, "Labels": y})

ids = dataset_ops.ImageDataSet(x, y)
ids.normalize()
ids.set_data_type(data_type="float32")
ids.shuffle()
ids.split_train_val_test(fractions=(0.80, 0.10, 0.10))
ids.prepare_tf_datasets(batch_size=BATCH_SIZE, buffer_factor=1)
ids.summary()


# ====================
#   Model building
# ====================
cnn_model = conv_nets.ConvolutionalNetwork(
    inp_shape=(128, 128, 3),
    num_epochs=NUM_EPOCHS,
    loss=MeanAbsolutePercentageError(),  # TODO
    optimizer=Adam(learning_rate=0.001),
    verbose=2,
    working_dir=WORKING_DIR,
    model_name="AGE-PREDICTION-CNN-VGG",
    regression=True,
)

cnn_model.make_vgg_cnn(
    num_blocks=2,
    num_dense_layers=0,
    num_outputs=1,
    cl_filters = (8, 16, 32),
    cl_droprate = (0.10, 0.15, 0.30),
    dl_units = (32,),
    dl_droprate=(0.3,),
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

cnn_model.plot_examples(num_samples=16, plt_size=12.0)

# cnn_model.train()

# cnn_model.plot_training_statistics()

cnn_model.make_report()
