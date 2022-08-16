import keras.activations
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.activations import relu
from keras.datasets import mnist
from keras import layers
from NeuralNetworks import neural_networks, report, dataset_ops, conv_nets
import os
import numpy as np

# ====================
#  Hyper-parameters
# ====================

NUM_EPOCHS = 100
BATCH_SIZE = 512

WORKING_DIR = os.path.join(os.path.dirname(__file__), "mnist_digits_test_results")
if not os.path.isdir(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# ====================
#  Data preparation
# ====================

(train_x, train_y), (test_x, test_y) = mnist.load_data()
x, y = np.concatenate((train_x, test_x), axis=0), np.concatenate((train_y, test_y), axis=0)

ids = dataset_ops.ImageDataSet(x, y)
ids.normalize()
ids.set_data_type(datype="float32")
ids.add_channel_dim()
ids.shuffle()
ids.split_train_val_test(fractions=(0.835, 0.155, 0.01))
ids.prepare_tf_datasets(batch_size=BATCH_SIZE, buffer_factor=1)
ids.summary()


cnn_model = conv_nets.ConvolutionalNetwork(
    inp_shape=(28, 28, 1),
    num_epochs=NUM_EPOCHS,
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.001),
    verbose=2,
    working_dir=WORKING_DIR,
    model_name="MNIST-DIGITS-CNN-VGG"
)

cnn_model.make_colnvolutional_layers(
    num_layers=3,
    num_filters=(16, 32, 64,),
    kernel_sizes=((3, 3), (3, 3), (3, 3),),
    strides=((2, 2), (2, 2), (2, 2),),
    activations=(relu, relu, relu,),
    padding=("same", "same", "same",),
    drop_rate=(0.0, 0.1, 0.1),
    use_bias=(False, False, False),
    use_bn=(False, True, True),
    pooling_obj=(
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
    use_bn=(False, False)
)

cnn_model.build()

cnn_model.plot(show_shapes=True)

cnn_model.add_data_sets(
    train_ds=ids.train_ds,
    val_ds=ids.val_ds,
    test_ds=ids.test_ds
)

cnn_model.plot_examples(num_samples=25, cmap="gray")

cnn_model.train()

cnn_model.plot_training_statistics()

cnn_model.make_report()

# training_report = report.TrainingReport(
#     model=cnn_model,
#     file_path=os.path.join(WORKING_DIR, "training_report.txt"),
#     line_len=120, line_marker="*",
# )
# training_report.add_hyperparameters()
# training_report.add_model_summary(cnn_model.nn_model)
# training_report.write_to_file()
