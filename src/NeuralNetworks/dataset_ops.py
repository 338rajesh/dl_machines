from numpy import expand_dims, arange
from numpy.random import default_rng
from tensorflow import data, keras
from keras import layers
from tensorflow import image

rng = default_rng()


class ImageDataSet:
    def __init__(self, images=None, labels=None, datype="float32"):
        self.images = images
        self.labels = labels
        self.datype = datype
        if images is not None and labels is not None:
            assert images.shape[0] == labels.shape[0], "Inconsistency in the number of images and the number of labels."

        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def normalize(self, min_pxv=0.0, max_pxv=1.0):
        self.images = min_pxv + ((max_pxv-min_pxv)*(self.images/255.0))
        return self

    def add_channel_dim(self):
        self.images = expand_dims(self.images, axis=-1)
        return self

    def __len__(self):
        if self.labels is not None:
            assert self.images.shape[0] == self.labels.shape[0], "Inconsistency in the number of images and the number of labels."
        return self.images.shape[0]

    def set_data_type(self, datype=None):
        if not datype:
            datype = self.datype

        if not self.images is None:
            if self.images.dtype is not datype:
                self.images = self.images.astype(datype)
        if not self.labels is None:
            if self.labels.dtype is not datype:
                self.labels = self.labels.astype(datype)
        return self

    def shuffle(self):
        shuffling_indices = arange(0, self.__len__())
        default_rng().shuffle(shuffling_indices, axis=0)
        self.images = self.images[shuffling_indices]
        self.labels = self.labels[shuffling_indices]
        return self

    def split_train_val_test(self, fractions=(0.8, 0.1, 0.1)):
        train_split, val_split, test_split = fractions
        assert train_split+val_split + \
            test_split == 1.0, f"Fractions {fractions} are not adding to 1.0"
        self.num_tr_examples = round(train_split*self.__len__())
        self.num_cv_examples = round(val_split*self.__len__())
        kidx = self.num_tr_examples + self.num_cv_examples
        self.num_ts_examples = self.__len__() - kidx
        #
        self.train_images = self.images[:self.num_tr_examples]
        self.train_labels = self.labels[:self.num_tr_examples]
        self.val_images = self.images[self.num_tr_examples:kidx]
        self.val_labels = self.labels[self.num_tr_examples:kidx]
        self.test_images = self.images[kidx:]
        self.test_labels = self.labels[kidx:]
        return

    def prepare_tf_datasets(self, batch_size=64, buffer_factor=1):
        def make_ds(imgs, lbls):
            if not imgs is None:
                ds = data.Dataset.from_tensor_slices(
                    tensors=(imgs, lbls) if (not lbls is None) else (imgs,)
                ).shuffle(self.__len__()*buffer_factor).batch(batch_size)
                return ds
            else:
                return None
        #
        self.train_ds = make_ds(self.train_images, self.train_labels)
        self.val_ds = make_ds(self.val_images, self.val_labels)
        self.test_ds = make_ds(self.test_images, self.test_labels)
        return self

    def summary(self):
        print("="*50)
        print(f"\t Image Dataset Summary")
        print("="*50)
        print(
            f" {'Number of training Examples':.<40s}{self.train_images.shape[0]:<10d}")
        print(
            f" {'Number of validation Examples':.<40s}{self.val_images.shape[0]:<10d}")
        print(
            f" {'Number of test Examples':.<40s}{self.test_images.shape[0]:<10d}")
        print(f" {'-'*40:40s}")
        print(f" {'Number of total Examples':.<40s}{self.__len__():<10d}")
        print("="*50)
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
        return
    #
    #

    def call(self, images, training=False):
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
            rot_ids = [] # TODO
            num_ccw_90rots = rng.integers(1, 4)
            # 1, 2, 3 => 90CCW, 180, 270CCW rotations
            images = image.rot90(images, k=num_ccw_90rots)
        return images
    #
