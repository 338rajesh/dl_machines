import keras.metrics
import tensorflow as tf
from keras import layers, Model, initializers, metrics
from numpy.ma.core import array
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os
import time
import numpy as np
#
from .utils import ProgressBar, Timer, TrainingReport, error_msg
#
plt.rcParams["font.size"] = 24.0
rng = np.random.default_rng()


class NeuralNetwork:

    def __init__(
            self,
            num_epochs: int = 10,
            optimizer=None,
            loss=None,
            metrics: list = None,
            verbose: int = 1,
            working_dir: str = None,
            model_name="Neural-Network",
    ):
        self.model_name = model_name
        self.nn_model: Model = Model()
        self.num_epochs = num_epochs
        #
        self.train_ds: tf.data.Dataset = None
        self.val_ds: tf.data.Dataset = None
        self.test_ds: tf.data.Dataset = None
        #
        self.loss = loss
        self.metrics: list[keras.metrics.Metric] = metrics
        self.optimizer = optimizer
        #
        #
        self.train_metrics: dict[str, keras.metrics.Metric] = {
            i.name: i for i in metrics}
        self.validation_metrics: dict[str, keras.metrics.Metric] = {
            i.name: i for i in metrics}
        self.test_metrics: dict[str, keras.metrics.Metric] = {
            i.name: i for i in metrics}
        self.training_report: dict[str, list] = {
            "training_time": [],
            "iters": [],
            "epochs": [],
            "loss": [],
        }
        self.training_report.update({i.name: [] for i in metrics})
        self.validation_report = {
            "epochs": [],
            "loss": [],
        }
        self.validation_report.update({i.name: [] for i in metrics})
        self.testing_report = {
            "epochs": [],
            "loss": [],
        }
        self.testing_report.update({i.name: [] for i in metrics})
        #
        self.verbose = verbose
        self.wrokdir: str = working_dir

    # ==================================================
    #                   Setters and Getters
    # ==================================================

    def add_data_sets(self, dataset):
        if dataset.train_ds is None:
            raise ValueError(error_msg("Training data is missing!"))
        else:
            self.train_ds = dataset.train_ds
        #
        if dataset.val_ds is not None:
            self.val_ds = dataset.val_ds
        #
        if dataset.test_ds is not None:
            self.test_ds = dataset.test_ds
        #
        self.batch_size: int = len(next(self.train_ds.take(1).as_numpy_iterator())[1])
        self.dataset_summary: str = dataset.summary_table

    # ==================================================
    #                   PLOTS
    # ==================================================

    def plot(self, fname: str = None, fsuffix: str = "pdf", **kwargs):
        if fname is None:
            path = os.path.join(self.wrokdir, f"{self.model_name}.{fsuffix}")
        plot_model(
            self.nn_model,
            to_file=path,
            **kwargs
        )

    def plot_examples(self, image_fname: str = None, num_samples: int = 9, cmap=None, plt_size=10.0):
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
        plt.figure(figsize=(plt_size, plt_size))
        for abatch_examples in self.train_ds.take(1):
            if len(abatch_examples) == 2:
                images, labels = abatch_examples
            elif len(abatch_examples) == 1:
                images = abatch_examples
            for i in range(num_samples):
                plt.subplot(m, n, i + 1)
                plt.imshow(X=array(images[i, :, :, 0]), cmap=cmap)
                plt.axis("off")
                if labels is not None:
                    plt.title(f"{labels[i]}")
            plt.savefig(image_path)

    def plot_metrics(self, results_dir: str = None, fig_size: float = 8.0):
        _epochs = self.training_report["epochs"]
        if results_dir is None:
            results_dir = self.wrokdir
        #

        def _plot_a_metric(_metric_name,):
            plt.figure(figsize=(fig_size, fig_size))
            # plotting training metrics
            plt.plot(
                _epochs, self.training_report[_metric_name], label=f"Training")
            # plotting validation metrics
            plt.plot(
                _epochs, self.validation_report[_metric_name], label=f"Validation")
            # plotting testing metrics
            plt.plot(
                _epochs[-1], self.testing_report[_metric_name], label=f"Testing", ms=10.0, marker="*")
            #
            plt.xlabel(f"# of epochs")
            plt.ylabel(_metric_name)
            plt.legend(loc="best")
            #
            plt.savefig(os.path.join(
                results_dir, f"metric_{_metric_name}.png"))
            plt.close()
            return
        # -----------------
        # plotting loss
        # -----------------
        _plot_a_metric("loss")
        # -----------------
        # plotting metrics
        # -----------------
        for a_metric in self.metrics:
            _plot_a_metric(a_metric.name)

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
        #
        for a_metric in self.metrics:  # updating training metrics
            self.train_metrics[a_metric.name].update_state(y, predictions)
        #
        return loss

    @tf.function
    def inference_step(self, _x, _y, metrics_dict: dict):
        _y_pred = self.nn_model(_x, training=False)
        for a_metric in self.metrics:  # updating validation metrics
            metrics_dict[a_metric.name].update_state(_y, _y_pred)
        #
        return self.loss(_y, _y_pred)

    def _update_report(self, _loss, _metrics, _report,):
        _report["loss"].append(float(_loss))
        for a_metric in self.metrics:
            _report[a_metric.name].append(
                float(_metrics[a_metric.name].result())
            )
            _metrics[a_metric.name].reset_states()
        return

    def train_val_test(self):
        #
        iters_counter = 0
        for a_epoch in range(self.num_epochs):
            t0 = time.time()
            pbar = ProgressBar(len(self.train_ds),)
            epoch_id = f"{a_epoch+1}/{self.num_epochs}"
            # ==============================
            #       TRAINING
            # ==============================
            #
            train_loss = 0.0
            for (i, (tr_x_batch, tr_y_batch)) in enumerate(self.train_ds):
                # running over each batch of the training dataset
                train_loss += self.train_step(tr_x_batch, tr_y_batch)
                iters_counter += 1
                if self.verbose > 1:
                    pbar.update(i, f"@ epoch: {epoch_id}")
                #
            train_loss /= len(self.train_ds)
            self._update_report(
                train_loss, self.train_metrics, self.training_report,)
            self.training_report["epochs"].append(a_epoch+1)
            self.training_report['iters'].append(int(iters_counter))
            self.training_report["training_time"].append((time.time() - t0))

            # ==============================
            #       VALIDATION
            # ==============================
            val_loss = 0.0
            for (j, (val_x_batch, val_y_batch)) in enumerate(self.val_ds):
                # running over each batch of the validation dataset
                val_loss += self.inference_step(
                    val_x_batch, val_y_batch, self.validation_metrics)
            val_loss /= len(self.val_ds)
            self._update_report(
                val_loss, self.validation_metrics, self.validation_report,)
            if self.verbose > 1:
                # col_width = max([len(i.name) for i in self.metrics]) + 5
                if a_epoch == 0:
                    _superscore = u'\u203e'
                    metr_headers = []
                    for amh in (
                        'Epoch', 'Iterations', 'Training_loss', 'Validation_loss',
                        *(tuple(f"training-{a_metric.name}" for a_metric in self.metrics)),
                        *(tuple(f"validation-{a_metric.name}" for a_metric in self.metrics)),
                    ):
                        metr_headers.append(f"{amh:^{len(amh)+6}s}")
                    col_widths = [len(i) for i in metr_headers]
                    subfixes = ["_"*i for i in col_widths]
                    superfixes = [_superscore*i for i in col_widths]
                    _thline = "|" + "|".join(superfixes) + "|"
                    _header = "|" + "|".join(metr_headers) + "|"
                    _bhline = "|" + "|".join(subfixes) + "|"
                    
                    _hline_suffx = '_'*len(_header)
                    print(f"\n\n{_thline}\n{_header}\n{_bhline}")

                _print_metrics = []
                for (_k, amh) in enumerate([
                    epoch_id,
                    f"{int(self.training_report['iters'][-1])}",
                    f"{self.training_report['loss'][-1]:0.3f}",
                    f"{self.validation_report['loss'][-1]:0.3f}",
                ] + [f"{self.training_report[a_metric.name][-1]:0.3f}" for a_metric in self.metrics
                     ] + [f"{self.validation_report[a_metric.name][-1]:0.3f}" for a_metric in self.metrics]
                ):
                    _print_metrics.append(f"{amh:^{col_widths[_k]}s}")
                #
                print("|"+"|".join(_print_metrics)+"|")
                #
                if a_epoch+1 == self.num_epochs:
                    print(_bhline)

        #
        # ==============================
        #       TESTING
        # ==============================
        test_loss = 0.0
        for (k, (test_x_batch, test_y_batch)) in enumerate(self.test_ds):
            # running over each batch of the test dataset
            test_loss += self.inference_step(
                test_x_batch, test_y_batch, self.test_metrics)
        test_loss /= len(self.test_ds)
        self._update_report(
            test_loss, self.test_metrics, self.testing_report)
        return self

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

    def write_report(self, fname=None):
        if fname is None:
            fname = f"{self.model_name}-report.txt"
        tr_report = TrainingReport(
            model=self,
            file_path=os.path.join(self.wrokdir, fname),
            line_len=120,
            line_marker="*",
        )
        tr_report.add_hyperparameters()
        tr_report.add_model_summary(self.nn_model)
        tr_report.add_data_set_summary(self)
        tr_report.add_model_performance(self)
        tr_report.write_to_file()
