from keras import layers

from .dataset_ops import DataAugmentationLayer
from .neural_networks import NeuralNetwork


class ConvolutionalNetwork(NeuralNetwork):
    def __init__(self,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        return

    # ======================================
    #               CONV BLOCKS
    # ======================================

    @staticmethod
    def vgg_block(
            x,
            num_filters: int,
            kernel_size: tuple = (3, 3),
            activation: str = "relu",
            kernel_init: str = "he_uniform",
            padding: str = "same",
            use_bn: bool = True,
            pool_size: tuple = (2, 2),
            pool_stride: tuple = (2, 2),
            drop_rate: float = 0.0,
    ):
        x = layers.Conv2D(
            num_filters,
            kernel_size,
            kernel_initializer=kernel_init,
            padding=padding,
            activation=activation,
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        #
        x = layers.Conv2D(
            num_filters,
            kernel_size,
            kernel_initializer=kernel_init,
            padding=padding,
            activation=activation
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        #
        x = layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride)(x)
        if 1.0 > drop_rate > 0.0:
            x = layers.Dropout(rate=drop_rate)(x)
        return x

    def make_vgg_conv_layers(
            self,
            num_blocks,
            aug_data=False, **kwargs
    ):
        """_summary_

        kwargs:
            >> cl_filters
            >> cl_drop_rate
            >> dl_drop_rate
            >> dl_units
        :param num_blocks: _description_
        :type num_blocks: _type_
        :param aug_data: _description_, defaults to False
        :type aug_data: bool, optional
        :return: _description_
        :rtype: _type_
        """
        img_inp = layers.Input(shape=self.inp_shape)
        self.model_input = img_inp
        self.current_layer = DataAugmentationLayer(
            kwargs["hflip"],
            kwargs["vflip"],
            kwargs["rot90ccw"],
            kwargs["rot180ccw"],
            kwargs["rot270ccw"],
        )(img_inp) if aug_data else img_inp
        #
        for i in range(num_blocks):
            self.current_layer = self.vgg_block(
                self.current_layer,
                num_filters=kwargs["cl_filters"][i],
                drop_rate=kwargs["cl_drop_rate"][i],
            )
        #
        self.current_layer = layers.Flatten()(self.current_layer)
        return self
