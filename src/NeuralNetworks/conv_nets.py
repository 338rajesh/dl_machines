from .neural_networks import NeuralNetwork
from .dataset_ops import DataAugmentationLayer
from tensorflow import keras
from keras import layers, initializers, Model


class ConvolutionalNetwork(NeuralNetwork):
    def __init__(self,
                 inp_shape: tuple,
                 num_epochs: int = 10,
                 loss=None,
                 optimizer=None,
                 verbose=1,
                 **kwargs,
                 ):
        # super().__init__(num_epochs, train_dataset, test_dataset, loss, optimizer, verbose)
        super().__init__(num_epochs, loss, optimizer, verbose, **kwargs)
        self.inp_shape = inp_shape
        return

    # ======================================
    #               CONV BLOCKS
    # ======================================

    @staticmethod
    def convolution_block(
            x,
            num_filters: int,
            activation=None,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="same",
            drop_rate=-1.0,
            use_bias: bool = True,
            use_bn: bool = False,
    ):
        """_summary_

        :param x: _description_
        :type x: _type_
        :param num_filters: _description_
        :type num_filters: int
        :param activation: _description_, defaults to None
        :type activation: _type_, optional
        :param kernel_size: _description_, defaults to (4, 4)
        :type kernel_size: tuple, optional
        :param strides: _description_, defaults to (1, 1)
        :type strides: tuple, optional
        :param padding: _description_, defaults to "same"
        :type padding: str, optional
        :param drop_rate: _description_, defaults to -1.0
        :type drop_rate: float, optional
        :param use_bias: _description_, defaults to True
        :type use_bias: bool, optional
        :param use_bn: _description_, defaults to False
        :type use_bn: bool, optional
        :return: _description_
        :rtype: _type_
        """
        x = layers.Conv2D(
            filters=num_filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            use_bias=use_bias,
            kernel_initializer=initializers.initializers_v2.HeUniform()
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = activation(x)
        #
        if drop_rate > 0.0:
            x = layers.Dropout(rate=drop_rate)(x)
        return x

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

    def make_vgg_cnn(
        self,
        num_blocks,
        num_dense_layers,
        num_outputs,
        aug_data=False, **kwargs
    ):
        """_summary_

        kwargs:
            >> cl_filters
            >> cl_droprate
            >> dl_droprate
            >> dl_units

        :param num_blocks: _description_
        :type num_blocks: _type_
        :param num_dense_layers: _description_
        :type num_dense_layers: _type_
        :param num_outputs: _description_
        :type num_outputs: _type_
        :param aug_data: _description_, defaults to False
        :type aug_data: bool, optional
        :return: _description_
        :rtype: _type_
        """    
        img_inp = layers.Input(shape=self.inp_shape)
        x = DataAugmentationLayer(
            kwargs["hflip"],
            kwargs["vflip"],
            kwargs["rot90ccw"],
            kwargs["rot180ccw"],
            kwargs["rot270ccw"],
        )(img_inp) if aug_data else img_inp
        #
        for i in range(num_blocks):
            x = self.vgg_block(
                x,
                num_filters=kwargs["cl_filters"][i],
                drop_rate=kwargs["cl_droprate"][i],
            )
        #
        x = layers.Flatten()(x)
        #
        for i in range(num_dense_layers):
            x = layers.Dense(
                units=kwargs["dl_units"][i],
                activation="relu",
                kernel_initializer="he_uniform",
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(rate=kwargs["dl_droprate"][i])(x)
        #
        output_layer = layers.Dense(units=num_outputs,)(x)
        self.nn_model = Model(img_inp, output_layer, name=self.model_name)
        return self

    def load_model(self, model):
        self.nn_model = model
        return self

    def make_colnvolutional_layers(self, num_layers=2, **layers_data):
        """_summary_

        :param num_layers: _description_, defaults to 2
        :type num_layers: int, optional
        :return: _description_
        :rtype: _type_
        """
        #
        #
        img_inp = layers.Input(shape=self.inp_shape)
        x = img_inp

        for i in range(num_layers):
            x = self.convolution_block(
                x,
                num_filters=layers_data["num_filters"][i],
                kernel_size=layers_data["kernel_sizes"][i],
                activation=layers_data["activations"][i],
                strides=layers_data["strides"][i],
                padding=layers_data["padding"][i],
                drop_rate=layers_data["drop_rate"][i],
                use_bias=layers_data["use_bias"][i],
                use_bn=layers_data["use_bn"][i],
            )
            # assert x.shape == layers_data["output_shapes"], f"Output inconsistency in  layer #{i}"
            if "pooling_obj" in layers_data.keys():
                if layers_data["pooling_obj"][i] is not None:
                    x = layers_data["pooling_obj"][i](x)
        self.model_input = img_inp
        self.conv_layers_part = x
        return self

    def make_dense_layers(self, num_layers=1, **layers_data):
        x = layers.Flatten()(self.conv_layers_part)
        #
        if 1.0 > layers_data["drop_rate"][0] > 0.0:
            x = layers.Dropout(rate=layers_data["drop_rate"][0])(x)
        #
        for i in range(num_layers-1):
            x = layers.Dense(
                units=layers_data["num_units"][i+1],
                activation=layers_data["activations"][i+1],
                use_bias=layers_data["use_bias"][i+1],
            )(x)
            if layers_data["use_bn"][i+1]:
                x = layers.BatchNormalization()(x)
            if 1.0 > layers_data["drop_rate"][i+1] > 0.0:
                x = layers.Dropout(rate=layers_data["drop_rate"][i+1])(x)
        self.model_output = x
        return self

    def build(self):
        self.nn_model = Model(
            self.model_input, self.model_output, name="CNN-model")
        return self
