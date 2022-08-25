from numpy import expand_dims, arange, array, ndarray
from numpy.random import default_rng
from tensorflow import data
from keras import layers
from tensorflow import image
from .utils import SimpleTable, error_msg
import matplotlib.pyplot as plt
from os import path

rng = default_rng()


class ImageDataSet:
    def __init__(self, images=None, labels=None, data_type="float32", name=None, work_dir=None):
        self.images = images
        self.labels = labels
        self.data_type = data_type
        self.name = name
        self.work_dir = work_dir
        if images is not None and labels is not None:
            assert images.shape[0] == labels.shape[0], "Inconsistency in the number of images and the number of labels."

        self.num_ts_examples = None
        self.num_cv_examples = None
        self.num_tr_examples = None
        self.summary_table = None

        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.label_names = [""]

    @staticmethod
    def get_rows_cols(n_samples):
        a, b, c = 1, n_samples, 0
        while a < b:
            c += 1
            if n_samples % c == 0:
                a = c
                b = n_samples // a
        return b, a

    def normalize(self, min_pxv=0.0, max_pxv=1.0):
        self.images = min_pxv + ((max_pxv - min_pxv) * (self.images / 255.0))
        return self

    def add_channel_dim(self):
        self.images = expand_dims(self.images, axis=-1)
        return self

    def __len__(self):
        if self.labels is not None:
            assert self.images.shape[0] == self.labels.shape[
                0], "Inconsistency in the number of images and the number of labels."
        return self.images.shape[0]

    def set_data_type(self, data_type=None):
        if not data_type:
            data_type = self.data_type

        if self.images is not None:
            if self.images.dtype is not data_type:
                self.images = self.images.astype(data_type)
        if self.labels is not None:
            if self.labels.dtype is not data_type:
                self.labels = self.labels.astype(data_type)
        return self

    def shuffle(self):
        shuffling_indices = arange(0, self.__len__())
        default_rng().shuffle(shuffling_indices, axis=0)
        self.images = self.images[shuffling_indices]
        self.labels = self.labels[shuffling_indices]
        return self

    def split_train_val_test(self, fractions=(0.8, 0.1, 0.1)):
        train_split, val_split, test_split = fractions
        assert train_split + val_split + \
               test_split == 1.0, f"Fractions {fractions} are not adding to 1.0"
        self.num_tr_examples = round(train_split * self.__len__())
        self.num_cv_examples = round(val_split * self.__len__())
        nts = self.num_tr_examples + self.num_cv_examples
        self.num_ts_examples = self.__len__() - nts
        #
        self.train_images = self.images[:self.num_tr_examples]
        self.train_labels = self.labels[:self.num_tr_examples]
        self.val_images = self.images[self.num_tr_examples:nts]
        self.val_labels = self.labels[self.num_tr_examples:nts]
        self.test_images = self.images[nts:]
        self.test_labels = self.labels[nts:]
        return self

    def prepare_tf_datasets(self, batch_size=64, buffer_factor=1):
        def make_ds(_images, _labels):
            if _images is not None:
                ds = data.Dataset.from_tensor_slices(
                    tensors=(_images, _labels) if (_labels is not None) else (_images,)
                ).shuffle(self.__len__() * buffer_factor).batch(batch_size)
                return ds
            else:
                return None

        #
        self.train_ds = make_ds(self.train_images, self.train_labels)
        self.val_ds = make_ds(self.val_images, self.val_labels)
        self.test_ds = make_ds(self.test_images, self.test_labels)
        return self

    def summary(self):
        content = [
            ["", "Training Set", "Validation Set", "Test Set"],
            ["Number of examples", self.train_images.shape[0], self.val_images.shape[0], self.test_images.shape[0]],
            ["Image shape", str(self.train_images.shape[1:]), str(self.val_images.shape[1:]),
             str(self.test_images.shape[1:])],
        ]
        if self.name is None:
            self.name = ""
        self.summary_table = SimpleTable(title=f"{self.name} Dataset Summary")(content=content)
        print(self.summary_table)
        return self

    def set_label_names(self, label_names):
        self.label_names = label_names
        return

    def assert_work_dir(self):
        assert self.work_dir is not None, error_msg(f"Please set working directory!")
        assert path.isdir(self.work_dir), error_msg(f"Working directory {self.work_dir} doesn't exist")

    def plot_examples(
            self,
            file_name: str = None,
            num_samples: int = 9,
            c_map=None,
            plt_size=10.0,
            titles: bool = False,

    ):
        self.assert_work_dir()
        images_indices = arange(0, self.__len__())
        default_rng().shuffle(images_indices, axis=0)
        if self.labels is not None:
            images_labels = [(self.images[i], self.labels[i]) for i in images_indices[:num_samples]]
        else:
            images_labels = [self.images[i] for i in images_indices[:num_samples]]

        m, n = self.get_rows_cols(num_samples)
        plt.figure(figsize=(plt_size, plt_size))
        for (i, a_example) in enumerate(images_labels):
            if len(a_example) == 2:
                a_image, label = a_example
            else:
                a_image = a_example
                label = None
            #
            plt.subplot(m, n, i + 1)
            plt.imshow(X=array(a_image[:, :, 0]), cmap=c_map)
            plt.axis("off")
            if titles:
                if label is not None:
                    if isinstance(label, (ndarray,)):
                        _label_i = label.ravel().tolist()
                    elif isinstance(label, (int, float)):
                        _label_i = [label, ]
                    else:
                        raise ValueError(f"While plotting examples, found unknown label type {type(label)}")
                    #
                    _label_i = [round(i, 2) for i in _label_i]
                    plt.title("-".join(list(map(str, _label_i))))
        plt.savefig(path.join(self.work_dir, file_name))

    def plot_label_histograms(self, plt_name: str, subsets_plt_name: str = None, plt_size=10.0, **hist_kwargs):
        assert self.labels is not None, error_msg(f"As there are no labels, histograms cannot be plotted!")
        self.labels = array(self.labels)
        if self.labels.ndim == 1:
            self.labels = self.labels.reshape(-1, 1)
        assert self.labels.ndim == 2, error_msg(f"Number of dimensions of labels array > 2")
        num_labels_per_example = self.labels.shape[1]
        m, n = self.get_rows_cols(num_labels_per_example)
        plt.figure(figsize=(n * plt_size, m * plt_size))
        for i in range(num_labels_per_example):
            plt.subplot(m, n, i + 1)
            plt.hist(self.labels[:, i], **hist_kwargs)
            if len(self.labels) > 0:
                plt.title(f"{self.label_names[i]}")
        plt.savefig(path.join(self.work_dir, plt_name))
        plt.close()
        #
        if subsets_plt_name is not None:
            num_subsets = 3
            # num_subsets += 1 if self.train_labels is not None else 0
            # num_subsets += 1 if self.val_labels is not None else 0
            # num_subsets += 1 if self.test_labels is not None else 0
            # assert num_subsets > 0, error_msg(f"subset labels are not found so histograms can not be plotted")
            plt.figure(figsize=(num_subsets*plt_size, num_labels_per_example*plt_size))
            for i in range(num_labels_per_example):
                if self.train_labels is not None:
                    plt.subplot(num_labels_per_example, num_subsets, i * num_subsets + 1)
                    plt.hist(self.train_labels[:, i], **hist_kwargs)
                if self.val_labels is not None:
                    plt.subplot(num_labels_per_example, num_subsets, i * num_subsets + 2)
                    plt.hist(self.val_labels[:, i], **hist_kwargs)
                if self.test_labels is not None:
                    plt.subplot(num_labels_per_example, num_subsets, i * num_subsets + 3)
                    plt.hist(self.test_labels[:, i], **hist_kwargs)
            plt.savefig(path.join(self.work_dir, subsets_plt_name))
        return


class DataAugmentationLayer(layers.Layer):
    def __init__(
            self,
            lr_flip=False,
            ud_flip=False,
            rot90cw=False,
            rot90ccw=False,
            rot180=False,
            seed=None,
            trainable=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.lr_flip = lr_flip
        self.ud_flip = ud_flip
        self.rot90cw = rot90cw
        self.rot90ccw = rot90ccw
        self.rot180 = rot180
        self.aug_seed = seed
        self.trainable = trainable
        return

    #
    #

    def call(self, images, training=False, *args, **kwargs):
        if not training:
            return images
        # do ops
        if self.lr_flip:
            images = image.stateless_random_flip_left_right(
                images, seed=self.aug_seed)
        if self.ud_flip:
            images = image.stateless_random_flip_up_down(
                images, seed=self.aug_seed)
        #
        # random rotation to one of
        rot_decision = bool(rng.integers(0, 2))
        if rot_decision:
            num_ccw_90rots = rng.integers(1, 4)
            # 1, 2, 3 => 90CCW, 180, 270CCW rotations
            images = image.rot90(images, k=num_ccw_90rots)
        return images
    #
