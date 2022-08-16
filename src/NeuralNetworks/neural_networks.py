import keras.metrics
import tensorflow as tf
from keras import layers, Model, initializers
from numpy.ma.core import array
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os
import time
import numpy as np
from NeuralNetworks import report

from .utils import ProgressBar
plt.rcParams["font.size"] = 24.0
rng = np.random.default_rng()


class NeuralNetwork:

    def __init__(
            self,
            num_epochs: int = 10,
            loss=None,
            optimizer=None,
            verbose=1,
            regression: bool = False,
            working_dir = None,
            model_name = "Neural-Network"
    ):
        self.nn_model: Model = Model()
        self.num_epochs = num_epochs
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose
        self.metrics = {}
        self.wrokdir = working_dir
        self.model_name = model_name
        self.training_history = []
        self.training_time = 0.0
        if regression:
            self.fit_type = "REGRESSION"
        else:
            self.fit_type = "CLASSIFICATION"
            self.lna = {
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "test_loss": [],
                "train_acc": [],
                "val_acc": [],
                "test_acc": [],
            }

    # ==================================================
    #                   Setters and Getters
    # ==================================================

    def add_data_sets(self, train_ds: tf.data.Dataset, val_ds=None, test_ds=None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        #
        self.batch_size = len(next(train_ds.take(1).as_numpy_iterator())[1])

    def start_watch(self):
        self.start_time = time.time()
        return self
    def stop_watch(self):
        self.end_time = time.time()
        return self

    # ==================================================
    #                   PLOTS
    # ==================================================

    def plot(self, fname: str=None, fsuffix: str = "pdf", **kwargs):
        if fname is None:
            path = os.path.join(self.wrokdir, f"{self.model_name}.{fsuffix}")
        plot_model(
            self.nn_model,
            to_file=path,
            **kwargs
        )

    def plot_examples(self, image_fname: str = None, num_samples: int = 9, cmap=None):
        if image_fname is None:
            image_fname = f"{self.model_name}.png"
        image_path = os.path.join(self.wrokdir, f"samples-{image_fname}")
        def get_rows_cols():
            a, b, c = 1, num_samples, 0
            while a < b:
                c += 1
                if num_samples % c == 0:
                    a = c
                    b = num_samples // a
            return b, a

        m, n = get_rows_cols()
        labels = None
        for abatch_examples in self.train_ds.take(1):
            if len(abatch_examples)==2:
                images, labels = abatch_examples
            elif len(abatch_examples)==1:
                images = abatch_examples
            for i in range(num_samples):
                plt.subplot(m, n, i + 1)
                plt.imshow(X=array(images[i, :, :, 0]), cmap=cmap)
                plt.axis("off")
                if labels is not None:
                    plt.title(f"{labels[i]}")
            plt.savefig(image_path)

    def plot_training_statistics(self, results_dir: str=None):
        if results_dir is None:
            results_dir =  self.wrokdir
        if self.fit_type == "CLASSIFICATION":
            if len(self.lna['epochs']) == 0:
                raise ValueError(
                    "Training statistics are not available; consider plotting after training.")
            #
            plt.figure(figsize=(18, 9))
            # Accuracy plots
            plt.subplot(1, 2, 1)
            plt.plot(self.lna['epochs'], self.lna['train_acc'],
                     label="Training Accuracy")
            plt.plot(self.lna['epochs'], self.lna['val_acc'],
                     label="Validation Accuracy")
            plt.plot(self.lna['epochs'][-1], self.lna['test_acc'],
                     label="Test Accuracy", marker="*", ms=10.0)
            plt.title("Model Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(loc="best")
            #
            # Loss plots
            plt.subplot(1, 2, 2)
            plt.plot(self.lna['epochs'],
                     self.lna['train_loss'], label="Training Loss")
            plt.plot(self.lna['epochs'], self.lna['val_loss'],
                     label="Validation Loss")
            plt.plot(self.lna['epochs'][-1], self.lna['test_loss'],
                     label="Test Loss", marker="*", ms=10.0)
            plt.title("Model Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(loc="best")
            #
            plt.savefig(os.path.join(
                results_dir, "Model_Performance_Statistics.png"))
            plt.close()

        elif self.fit_type == "REGRESSION":
            raise NotImplementedError(
                "Statistics plotting is not implemented for regression.")
        return

    # ==================================================
    #                   TRAINING
    # ==================================================

    @tf.function
    def train_step(self, x, y):
        # open gradient tape
        with tf.GradientTape() as tape:
            # prediction: forward pass
            predictions = self.nn_model(x, training=True)
            # get the loss
            loss = self.loss(y, predictions)
        # find the gradient
        gradients = tape.gradient(loss, self.nn_model.trainable_variables)
        # update weights using the optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.nn_model.trainable_variables))

        self.metrics["train_loss"](loss)
        self.metrics["train_accuracy"](y, predictions)

    @tf.function
    def inference_step(self, x, y, inference_id=""):
        # prediction: forward pass
        predictions = self.nn_model(x, training=False)
        # get the loss
        loss = self.loss(y, predictions)
        self.metrics[f"{inference_id}_loss"](loss)
        self.metrics[f"{inference_id}_accuracy"](y, predictions)

    def train(self):
        self.set_metrics()
        self.start_watch()
        for a_epoch in range(self.num_epochs):
            self.reset_metrics()
            #
            # run over each batch of the training dataset
            if self.verbose > 1:
                t0 = time.time()
                train_batch_bar = ProgressBar(len(self.train_ds),)
            for (i, (tr_x_batch, tr_y_batch)) in enumerate(self.train_ds):
                self.train_step(tr_x_batch, tr_y_batch)
                if self.verbose > 1:
                    train_batch_bar.update(i)
            #
            # run over each batch of the validation dataset
            for (val_x_batch, val_y_batch) in self.val_ds:
                self.inference_step(
                    val_x_batch, val_y_batch, inference_id="val")

            self._collect_loss_and_accuracy(a_epoch+1, print_lna=True)
        #
        # run over each batch of the test dataset
        self.reset_metrics()
        for (test_x_batch, test_y_batch) in self.test_ds:
            self.inference_step(test_x_batch, test_y_batch,
                                inference_id="test")
        self._collect_loss_and_accuracy(
            a_epoch+1, print_lna=True, test_lna=True)
        #
        self.stop_watch()
        self.training_time = self.end_time - self.start_time
        return self

    def _collect_loss_and_accuracy(self, epch_num, print_lna: bool = False, test_lna=False):
        #
        if not test_lna:
            self.lna["epochs"].append(epch_num)
            self.lna['train_loss'].append(self.metrics['train_loss'].result())
            self.lna['val_loss'].append(self.metrics['val_loss'].result())
            self.lna['train_acc'].append(
                self.metrics['train_accuracy'].result()*100)
            self.lna['val_acc'].append(
                self.metrics['val_accuracy'].result()*100)
            if self.verbose > 1:
                bar_header = f"{'| epoch |':^9s}{'| Training Loss |':^17s}{'| Validation Loss |':^19s}" + \
                    f"{'| Training Accuracy |':^21s}{'| Validation Accuracy |':^23s}"
                if epch_num == 1:
                    self.training_history += [
                        f"{'='*len(bar_header)}", f"{bar_header}", f"{'='*len(bar_header)}"
                    ]
                    print("\n".join(self.training_history[-3:]))
                self.training_history.append(
                    f"|{self.lna['epochs'][-1]:^7d}|"
                    f"|{self.lna['train_loss'][-1]:^15.3f}|"
                    f"|{self.lna['val_loss'][-1]:^17.3f}|"
                    f"|{self.lna['train_acc'][-1]:^19.3f}|"
                    f"|{self.lna['val_acc'][-1]:^21.3f}|"
                )
                print(self.training_history[-1])
                if epch_num == self.num_epochs:
                    self.training_history.append(f"{'~'*len(bar_header)}")
                    print(self.training_history[-1])
        else:
            self.lna['test_loss'].append(self.metrics['test_loss'].result())
            self.lna['test_acc'].append(
                self.metrics['test_accuracy'].result()*100)
            if self.verbose > 1:
                bar_header = f"PERFORMANCE ON TEST SET: "
                self.training_history += [
                    f"{'~'*len(bar_header)}", bar_header,  f"{'~'*len(bar_header)}",
                ]
                print("\n".join(self.training_history[-3:]))
                self.training_history += [
                    f"Loss: {self.lna['test_loss'][-1]:4.3f}",
                    f"Accuracy: {self.lna['test_acc'][-1]:4.3f}",
                    f"{'~'*len(bar_header)}",
                ]
                print("\n".join(self.training_history[-3:]))
        return self

    def set_metrics(self):
        if self.fit_type == "CLASSIFICATION":
            self.metrics.update({
                "train_loss": keras.metrics.Mean(name="train_loss"),
                "train_accuracy": keras.metrics.SparseCategoricalAccuracy(name="train_accuracy"),
                "val_loss": keras.metrics.Mean(name="validation_loss"),
                "val_accuracy": keras.metrics.SparseCategoricalAccuracy(name="validation_accuracy"),
                "test_loss": keras.metrics.Mean(name="test_loss"),
                "test_accuracy": keras.metrics.SparseCategoricalAccuracy(name="test_accuracy"),
            })

        return

    def reset_metrics(self):
        if self.fit_type == "CLASSIFICATION":
            for v in self.metrics.values():
                v.reset_states()
        return

    # ==================================================
    #                   EXPORT
    # ==================================================

    def save_nn_model(self, fname: str = None, save_dir=None,):
        if save_dir is None:
            save_dir = self.wrokdir
        assert fname is not None, "Specify the file name for the model"
        fname_parts = fname.split(".")
        if len(fname_parts) > 1:
            assert fname_parts[-1] == "h5", f"Only HDF5 format is allowed but {fname_parts[-1]} is specified."
        #
        self.nn_model.save(os.path.join(save_dir, fname))
        return

    def make_report(self, fname=None):
        if fname is None:
            fname = f"{self.model_name}-report.txt"
        tr_report = report.TrainingReport(
            model=self,
            file_path=os.path.join(self.wrokdir, fname),
            line_len=120,
            line_marker="*",
        ).add_hyperparameters().add_model_summary(self.nn_model).add_model_performance(self).write_to_file()
        return
