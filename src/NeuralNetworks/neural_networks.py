import keras.metrics
import tensorflow as tf
from keras import Model, layers
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os
import time
import numpy as np
#
from .utils import ProgressBar, TrainingReport, error_msg, unravel_list

#
plt.rcParams["font.size"] = 24.0
rng = np.random.default_rng()


class NeuralNetwork:

    def __init__(
            self,
            inp_shape: tuple,
            num_outputs: int,
            num_epochs: int = 10,
            optimizer=None,
            loss=None,
            performance_metrics: list = None,
            verbose: int = 1,
            working_dir: str = None,
            model_name="Neural-Network",
            size_per_plot=8.0,
    ):
        self.inp_shape = inp_shape
        self.num_outputs = num_outputs
        self.model_input = None
        self.current_layer = None  # this will be used to access the current layer, across different parts of code.
        self.model_output = None
        #
        self.dataset_summary = None
        self.batch_size: int = 0
        self.model_name: str = model_name
        self.nn_model: Model = Model()
        self.num_epochs: int = num_epochs
        self.size_per_plot: float = size_per_plot
        #
        self.train_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(np.array([]))
        self.val_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(np.array([]))
        self.test_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(np.array([]))
        #
        self.loss = loss
        self.metrics: list[keras.metrics.Metric] = performance_metrics
        self.optimizer = optimizer
        #
        self.y_train_predictions = []
        self.y_train_actual = []
        self.y_val_predictions = []
        self.y_val_actual = []
        self.y_test_predictions = []
        self.y_test_actual = []
        #
        self.train_metrics: dict[str, keras.metrics.Metric] = {
            i.name: i for i in performance_metrics}
        self.validation_metrics: dict[str, keras.metrics.Metric] = {
            i.name: i for i in performance_metrics}
        self.test_metrics: dict[str, keras.metrics.Metric] = {
            i.name: i for i in performance_metrics}
        self.training_report: dict[str, list] = {
            "training_time": [],
            "iters": [],
            "epochs": [],
            "loss": [],
        }
        self.training_report.update({i.name: [] for i in performance_metrics})
        self.validation_report = {
            "epochs": [],
            "loss": [],
        }
        self.validation_report.update({i.name: [] for i in performance_metrics})
        self.testing_report = {
            "epochs": [],
            "loss": [],
        }
        self.testing_report.update({i.name: [] for i in performance_metrics})
        #
        self.verbose = verbose
        self.working_dir: str = working_dir

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

    def plot(self, fpath: str = None, file_suffix: str = "pdf", **kwargs):
        if fpath is None:
            fpath = os.path.join(self.working_dir, f"{self.model_name}.{file_suffix}")
        plot_model(
            self.nn_model,
            to_file=fpath,
            **kwargs
        )

    def plot_metrics(self, results_dir: str = None, fig_size: float = 8.0):
        _epochs = self.training_report["epochs"]
        if results_dir is None:
            results_dir = self.working_dir

        #

        def _plot_a_metric(_metric_name, ):
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

    def plot_actual_vs_predictions(self):
        # plot at the last epoch of training, validation and testing.
        # need y_true and y_predictions of all the examples of each set, at last epoch
        # self.y_train_predictions =
        def _common_ops(_x, _y, _title):
            plt.scatter(_x, _y, s=3.0, c="b", marker='o')
            plt.plot(
                [min(_x), max(_x)], [min(_x), max(_x)],
                linewidth=2, color='red', linestyle='solid'
            )
            plt.title(_title)
            plt.xlabel("Actual")
            plt.ylabel("Predictions")
            return
            # Training scatter plots

        self.y_train_actual = np.array(self.y_train_actual)
        self.y_train_predictions = np.array(self.y_train_predictions)
        self.y_val_actual = np.array(self.y_val_actual)
        self.y_val_predictions = np.array(self.y_val_predictions)
        self.y_test_actual = np.array(self.y_test_actual)
        self.y_test_predictions = np.array(self.y_test_predictions)

        num_outputs = self.y_train_actual.shape[1]
        plt.figure(figsize=(num_outputs * self.size_per_plot, 3 * self.size_per_plot))

        for i in range(num_outputs):
            # Training plots
            plt.subplot(num_outputs, 3, 3 * i + 1)
            _common_ops(self.y_train_actual[:, i], self.y_train_predictions[:, i], "")
            # Validation plots
            plt.subplot(num_outputs, 3, 3 * i + 2)
            _common_ops(self.y_val_actual[:, i], self.y_val_predictions[:, i], "")
            # Testing plots
            plt.subplot(num_outputs, 3, 3 * i + 3)
            _common_ops(self.y_test_actual[:, i], self.y_test_predictions[:, i], "")
        plt.suptitle(f"Actual vs Predicted properties after {self.num_epochs} epochs")
        plt.savefig(os.path.join(self.working_dir, "yActual_yPredictions_last_epoch.png"))
        plt.close()
        return

    # ==================================================
    #                   MODEL BUILDING
    # ==================================================

    def make_dense_layers(
            self,
            num_hidden_layers: int = 0,
            **kwargs,
    ):
        #
        if self.current_layer is None:
            self.model_input = layers.Input(shape=self.inp_shape)
            self.current_layer = self.model_input
        #
        for i in range(num_hidden_layers):
            self.current_layer = layers.Dense(
                units=kwargs["hl_units"][i],
                activation="relu",
                kernel_initializer="he_uniform",
            )(self.current_layer)
            self.current_layer = layers.BatchNormalization()(self.current_layer)
            self.current_layer = layers.Dropout(rate=kwargs["hl_drop_rate"][i])(self.current_layer)
        #
        self.current_layer = layers.Dense(units=self.num_outputs, )(self.current_layer)
        if "outputs_activation_layer" in kwargs.keys():
            self.current_layer = kwargs["outputs_activation_layer"](self.current_layer)
        #
        return

    def build(self):
        if self.model_output is None:
            if self.current_layer is None:
                raise ValueError(error_msg(f"Output layer is required for building model!"))
            else:
                self.model_output = self.current_layer
        #
        self.nn_model = Model(self.model_input, self.model_output, name=self.model_name)
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
        return loss, predictions

    @tf.function
    def inference_step(self, _x, _y, metrics_dict: dict):
        _y_predictions = self.nn_model(_x, training=False)
        for a_metric in self.metrics:  # updating validation metrics
            metrics_dict[a_metric.name].update_state(_y, _y_predictions)
        #
        return self.loss(_y, _y_predictions), _y_predictions

    def _update_report(self, _loss, _metrics, _report, ):
        _report["loss"].append(float(_loss))
        for a_metric in self.metrics:
            _report[a_metric.name].append(
                float(_metrics[a_metric.name].result())
            )
            _metrics[a_metric.name].reset_states()
        return

    def train_val_test(self):
        #
        col_widths = []
        _bh_line = ""
        iters_counter = 0
        for a_epoch in range(self.num_epochs):
            t0 = time.time()
            pbar = ProgressBar(len(self.train_ds), )
            epoch_id = f"{a_epoch + 1}/{self.num_epochs}"
            # ==============================
            #       TRAINING
            # ==============================
            #
            cum_train_loss = 0.0
            for (i, (tr_x_batch, tr_y_batch)) in enumerate(self.train_ds):
                # running over each batch of the training dataset
                train_loss, y_predictions = self.train_step(tr_x_batch, tr_y_batch)
                cum_train_loss += train_loss
                if a_epoch + 1 == self.num_epochs:
                    self.y_train_predictions.append(y_predictions)
                    self.y_train_actual.append(tr_y_batch)
                iters_counter += 1
                if self.verbose > 1:
                    pbar.update(i, f"@ epoch: {epoch_id}")
                #
            self.y_train_predictions = unravel_list(self.y_train_predictions)
            self.y_train_actual = unravel_list(self.y_train_actual)
            cum_train_loss /= len(self.train_ds)
            self._update_report(
                cum_train_loss, self.train_metrics, self.training_report, )

            # ==============================
            #       VALIDATION
            # ==============================
            cum_val_loss = 0.0
            for (j, (val_x_batch, val_y_batch)) in enumerate(self.val_ds):
                # running over each batch of the validation dataset
                val_loss, y_predictions = self.inference_step(
                    val_x_batch, val_y_batch, self.validation_metrics)
                cum_val_loss += val_loss
                if a_epoch + 1 == self.num_epochs:
                    self.y_val_predictions.append(y_predictions)
                    self.y_val_actual.append(val_y_batch)
            #
            self.y_val_predictions = unravel_list(self.y_val_predictions)
            self.y_val_actual = unravel_list(self.y_val_actual)
            cum_val_loss /= len(self.val_ds)
            self._update_report(
                cum_val_loss, self.validation_metrics, self.validation_report, )

            # ==============================
            #       SUMMARIZING EPOCH
            # ==============================
            self.training_report["epochs"].append(a_epoch + 1)
            self.training_report['iters'].append(int(iters_counter))
            self.training_report["training_time"].append((time.time() - t0))
            if self.verbose > 1:
                # col_width = max([len(i.name) for i in self.metrics]) + 5
                if a_epoch == 0:
                    _super_score = u'\u203e'
                    metr_headers = []
                    for amh in (
                            'Epoch', 'Iterations', 'Training_loss', 'Validation_loss',
                            *(tuple(f"training-{a_metric.name}" for a_metric in self.metrics)),
                            *(tuple(f"validation-{a_metric.name}" for a_metric in self.metrics)),
                    ):
                        metr_headers.append(f"{amh:^{len(amh) + 6}s}")
                    col_widths = [len(i) for i in metr_headers]
                    sub_fixes = ["_" * i for i in col_widths]
                    super_fixes = [_super_score * i for i in col_widths]
                    _th_line = "|" + "|".join(super_fixes) + "|"
                    _header = "|" + "|".join(metr_headers) + "|"
                    _bh_line = "|" + "|".join(sub_fixes) + "|"

                    _h_line_suffix = '_' * len(_header)
                    print(f"\n\n{_th_line}\n{_header}\n{_bh_line}")

                _print_metrics = []
                for (_k, amh) in enumerate([
                                               epoch_id,
                                               f"{int(self.training_report['iters'][-1])}",
                                               f"{self.training_report['loss'][-1]:0.3f}",
                                               f"{self.validation_report['loss'][-1]:0.3f}",
                                           ] + [f"{self.training_report[a_metric.name][-1]:0.3f}" for a_metric in
                                                self.metrics
                                                ] + [f"{self.validation_report[a_metric.name][-1]:0.3f}" for a_metric in
                                                     self.metrics]
                                           ):
                    _print_metrics.append(f"{amh:^{col_widths[_k]}s}")
                #
                print("|" + "|".join(_print_metrics) + "|")
                #
                if a_epoch + 1 == self.num_epochs:
                    print(_bh_line)

        #
        # ==============================
        #       TESTING
        # ==============================
        cum_test_loss = 0.0
        for (k, (test_x_batch, test_y_batch)) in enumerate(self.test_ds):
            # running over each batch of the test dataset
            test_loss, test_y_predictions = self.inference_step(
                test_x_batch, test_y_batch, self.test_metrics)
            cum_test_loss += test_loss
            self.y_test_predictions.append(test_y_predictions)
            self.y_test_actual.append(test_y_batch)
            #
        #
        self.y_test_predictions = unravel_list(self.y_test_predictions)
        self.y_test_actual = unravel_list(self.y_test_actual)
        cum_test_loss /= len(self.test_ds)
        self._update_report(
            cum_test_loss, self.test_metrics, self.testing_report)
        return self

    # ==================================================
    #                   EXPORT
    # ==================================================

    def save_nn_model(self, fname: str = None, save_dir=None, ):
        if save_dir is None:
            save_dir = self.working_dir
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
            file_path=os.path.join(self.working_dir, fname),
            line_len=120,
            line_marker="*",
        )
        tr_report.add_hyperparameters()
        tr_report.add_model_summary(self.nn_model)
        tr_report.add_data_set_summary(self)
        tr_report.add_model_performance(self)
        tr_report.write_to_file()
