import numpy as np
import matplotlib.pyplot as plt
import os
import time

import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, losses, models
from keras.utils.vis_utils import plot_model
#
# ===========================================

class GAN():
    def __init__(
        self, latent_dim, batch_size, num_epochs,
        bss_buffer, res_dir, num_test_examples=25,
    ):
        super(GAN,).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.res_dir = res_dir
        self.test_lat_points = tf.random.normal(
            shape=(num_test_examples, latent_dim))


class DCGAN():

    def __init__(
        self, images, latent_dim,
        batch_size, num_epochs, bss_buffer,
        res_dir, num_test_examples=25,
        verbose=1
    ):
        super(DCGAN,).__init__()
        num_im, im_w, im_h, num_ch = images.shape
        self.image_shape = (im_w, im_h, num_ch)
        self.latent_dim = latent_dim
        self.data_set = tf.data.Dataset.from_tensor_slices(
            images).shuffle(bss_buffer).batch(batch_size)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.res_dir = res_dir
        self.num_batches = int(num_im/batch_size)
        self.test_lat_points = tf.random.normal(
            shape=(num_test_examples, latent_dim))
        self.verbose = verbose


    def save_images(self, imgs: np.ndarray, fname):
        fig = plt.figure(figsize=(12, 12))
        nspr = int(np.sqrt(imgs.shape[0]))
        for i in range(self.test_lat_points.shape[0]):
            plt.subplot(nspr, nspr, 1+i)
            # FIXME formulae should consider range (0, 1) or (-1, 1)
            plt.imshow((imgs[i, :, :, 0]*127.5) + 127.5, cmap="gray")
            plt.axis("off")
        plt.savefig(os.path.join(self.res_dir, fname))
        plt.close()

    def generate_and_save_images(self,fname: str,):
        self.save_images(self.gen_model(self.test_lat_points, training=False), fname)

    def save_sample_train_images(self):
        num_test_samples = (self.test_lat_points).shape[0]
        random_images = np.array([img[i].numpy() for i in range(num_test_samples) for img in self.data_set.take(1)])
        print(f"random images shape {random_images.shape}")
        self.save_images(
            random_images,
            fname=f"randomly_selected_train_examples.png"
        )

    def plot_models(self, plot_gen: bool= True, plot_disc: bool = True):
        if plot_gen:
            plot_model(
                self.gen_model, 
                to_file=os.path.join(self.res_dir, "generator_model_plot.png"),
                show_shapes=True, show_layer_activations=True, show_layer_names=True
            )
        if plot_disc:
            plot_model(
                self.dsc_model, 
                to_file=os.path.join(self.res_dir, "discriminator_model_plot.png"),
                show_shapes=True, show_layer_activations=True, show_layer_names=True
            )

    @staticmethod
    def convolution_block(
        x, num_filters: int, activation,
        kernel_size=(4, 4), strides=(1, 1),
        padding="same", drop_rate=-1.0,
        use_bias: bool = True, use_bn: bool = False,
    ):
        x = layers.Conv2D(
            filters=num_filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            use_bias=use_bias,
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = activation(x)
        if drop_rate > 0.0:
            x = layers.Dropout(rate=drop_rate)(x)
        return x

    @staticmethod
    def convolution_2D_transpose_block(
        x,
        num_filters,
        activation=None,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding="same",
        use_bn=False,
        use_bias=True,
        drop_rate=-1.0,
    ):
        x = layers.Conv2DTranspose(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias
        )(x)
        #
        if use_bn:
            x = layers.BatchNormalization()(x)
        #
        if activation is not None:
            x = activation(x)
        #
        if drop_rate > 0.0:
            x = layers.Dropout(rate=drop_rate)(x)
        #
        return x

    # ================================================
    #                BUILDING GENERATOR
    # ================================================
    def build_generator(self, init_config: tuple, **layers_data):
        h0, w0, num_flt_0 = init_config
        num_nodes = h0 * w0 * num_flt_0
        #
        z_lp = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(num_nodes, use_bias=False)(z_lp)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        #
        x = layers.Reshape(target_shape=init_config)(x)
        #
        for (a_layer_name, a_layer) in layers_data.items():
            x = self.convolution_2D_transpose_block(
                x,
                num_filters=a_layer["num_filters"],
                kernel_size=a_layer["kernel_size"],
                activation=a_layer["activation"],
                strides=a_layer["strides"],
                padding=a_layer["padding"],
                drop_rate=a_layer["drop_rate"],
                use_bias=a_layer["use_bias"],
                use_bn=a_layer["use_bn"],
            )
            assert x.shape == a_layer["output_shape"], f"Output inconsistency in {a_layer_name} layer."
        #
        self.gen_model = models.Model(z_lp, x, name="Generator")

    # ================================================
    #                BUILDING DISCRIMINATOR 
    # ================================================
    def build_discriminator(self, **layers_data):
        img_inp = layers.Input(shape=self.image_shape)
        x = img_inp
        for (a_layer_name, a_layer) in layers_data.items():
            x = self.convolution_block(x,
                num_filters=a_layer["num_filters"],
                kernel_size=a_layer["kernel_size"],
                activation=a_layer["activation"],
                strides=a_layer["strides"],
                padding=a_layer["padding"],
                drop_rate=a_layer["drop_rate"],
                use_bias=a_layer["use_bias"],
                use_bn=a_layer["use_bn"],
            )
            assert x.shape == a_layer["output_shape"], f"Output inconsistency in a layer with ID  {a_layer_name}."
        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(units=1)(x)
        self.dsc_model = models.Model(img_inp, x, name="Discriminator")

    @staticmethod
    def get_gen_disc_losses(real_op, fake_op):
        cross_entropy = losses.BinaryCrossentropy(from_logits=True)
        disc_loss_on_real = cross_entropy(tf.ones_like(real_op), real_op)
        disc_loss_on_fake = cross_entropy(tf.zeros_like(fake_op), fake_op)
        gen_loss = cross_entropy(tf.ones_like(fake_op), fake_op)
        return gen_loss, disc_loss_on_fake+disc_loss_on_real

    gen_opt = optimizers.Adam(learning_rate=1e-04)
    dsc_opt = optimizers.Adam(learning_rate=1e-04)

    @tf.function
    def train_step(self, images_batch):
        batch_size = images_batch.shape[0]
        latent_points = tf.random.normal(shape=(batch_size, self.latent_dim))
        #
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.gen_model(latent_points, training=True)
            #
            real_dat_pred = self.dsc_model(images_batch, training=True)
            fake_dat_pred = self.dsc_model(generated_images, training=True)
            #
            g_loss, d_loss = self.get_gen_disc_losses(
                real_dat_pred, fake_dat_pred)
        #
        gen_grads = gen_tape.gradient(
            g_loss, self.gen_model.trainable_variables)
        dsc_grads = disc_tape.gradient(
            d_loss, self.dsc_model.trainable_variables)
        #
        self.gen_opt.apply_gradients(
            zip(gen_grads, self.gen_model.trainable_variables))
        self.dsc_opt.apply_gradients(
            zip(dsc_grads, self.dsc_model.trainable_variables))
    #

    def train(self):
        for aepoch in range(self.num_epochs):
            t0 = time.time()
            print(
                f"At epoch: {aepoch+1}/{self.num_epochs}, training in {self.num_batches} batches", end="", flush=True)
            for image_batch in self.data_set:
                print(".", end="", flush=True)
                self.train_step(image_batch)
            #
            self.generate_and_save_images(
                fname=f"gen_image_at_epoch{1+aepoch}.png")
            print(f" finished in {time.time()-t0} seconds.")
            # save model for every 15 epochs
            # if (aepoch+1)%15 == 0:
            #     check_point.save(file_prefix=check_point_prefix)
        self.generate_and_save_images(
            fname=f"final_gen_image_after_{self.num_epochs}_epochs.png")


class CNN():
    def __init__(self):
        super(CNN,).__init__()


class CNNRegression(CNN):
    def __init__(self, *args):
        super(CNNRegression, self).__init__(*args)
        

class CNNClassification(CNN):
    def __init__(self, *args):
        super(CNNClassification, self).__init__(*args)
        

