# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
import numpy as np
from instancenormalization import InstanceNormalization

CROP_SHAPE = (256, 256)
INPUT_SHAPE = CROP_SHAPE + (35 + 1,) # 35 labels, 1 boundary map
OUTPUT_SHAPE = CROP_SHAPE + (3,) # RGB image

# Weight Initializers
conv_init = tf.random_normal_initializer(0, 0.02)
batchnorm_init = tf.random_normal_initializer(1.0, 0.02)

# Applies reflection padding to an image_batch
refl_padding_num = 0
def reflection_pad(image_batch, pad):
    global refl_padding_num
    paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
    rval = tf.keras.layers.Lambda(lambda x: tf.pad(x, paddings, mode='REFLECT'), name='refl_padding_%d' % refl_padding_num)(image_batch)
    refl_padding_num += 1
    return rval
# A thin wrapper around conv2D that applies reflection_padding when required
def conv2D(img, k, f, s, reflect_pad):
    pad_mode = 'valid' if reflect_pad else 'same'
    if reflect_pad: 
        img = reflection_pad(img, f//2)
    return tf.keras.layers.Conv2D(k, (f, f), strides=(s, s), padding=pad_mode, kernel_initializer=conv_init)(img)

# Downsamples the image i times.
def downsample(img, i):
    raise RuntimeError('No average pooling!')
    return tf.layers.average_pooling2d(img, 2 ** i, 2 ** i)

def c7s1(x, k, activation, reflect_pad=True):    # 7×7 Convolution-InstanceNorm-Activation with k filters
    x = conv2D(x, k, 7, 1, reflect_pad=reflect_pad)
    x = InstanceNormalization(epsilon=1e-5)(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x

def d(x, k, reflect_pad=True):       # 3×3 Convolution-InstanceNorm-ReLU with k filters
    x = conv2D(x, k, 3, 2, reflect_pad=reflect_pad)
    x = InstanceNormalization(epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def R(x, k, reflect_pad=True):       # Residual block with 2 3×3 Convolution layers with k filters.
    y = x
    y = conv2D(y, k, 3, 1, reflect_pad=reflect_pad)
    y = InstanceNormalization(epsilon=1e-5)(y)
    y = tf.keras.layers.ReLU()(y)
    y = conv2D(y, k, 3, 1, reflect_pad=reflect_pad)
    y = InstanceNormalization(epsilon=1e-5)(y)
    y = tf.keras.layers.Add()([x, y])
    return y

def u(x, k):       # 3×3 Transposed Convolution-InstanceNorm-ReLU layer with k filters.
    x = tf.keras.layers.Conv2DTranspose(k, (3, 3), strides=(2, 2), padding='same', kernel_initializer=conv_init)(x)
    x = InstanceNormalization(epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


# Loss Components

def discriminator_loss(real_output, generated_output, lsgan):
    if lsgan:
        real_loss = tf.losses.mean_squared_error(tf.ones_like(real_output), real_output)
        generated_loss = tf.losses.mean_squared_error(tf.zeros_like(generated_output), generated_output)
    else:
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss
    return total_loss

def generator_gan_loss(generated_output, lsgan):
    if lsgan:
        return tf.losses.mean_squared_error(tf.ones_like(generated_output), generated_output)
    else:
        return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

def generator_feature_matching_loss(gen_activations, real_activations):
    total_loss = 0
    for g, r in zip(gen_activations, real_activations):
        num_elements = np.prod(g.shape.as_list()[1:])         # Number of elements excluding batch dimension
        total_loss += tf.losses.absolute_difference(g, r) / num_elements    # Paper suggests L1 loss
    return total_loss


# Generators

def define_global_generator(input_label_shape, output_channels, reflection_padding=True):
    ''' Define the coarse 'global' generator. '''

    down_layers = [128, 256, 512, 1024]
    residual_layers = [1024] * 9
    up_layers = [512, 256, 128, 64]

    input_label = tf.keras.Input(shape=input_label_shape)

    result = c7s1(input_label, 64, 'relu', reflect_pad=reflection_padding)

    for i, k in enumerate(down_layers):
        result = d(result, k, reflect_pad=False)
    for k in residual_layers:
        result = R(result, k, reflect_pad=reflection_padding)
    for k in up_layers:
        result = u(result, k)

    last_feature_map = result
    #result = c7s1(result, output_channels, 'tanh', reflect_pad=reflection_padding)
    #result = conv2D(result, output_channels, 7, 1, reflect_pad=reflection_padding)
    #result = tf.keras.layers.Activation('sigmoid')(result)
    
    return tf.keras.Model(inputs=input_label, outputs=[result, last_feature_map])

def define_enhancer_generator(input_label_shape, coarse_input_shape, output_channels, reflection_padding=True):
    ''' Define the fine 'enhancer' generator. '''

    input_label = tf.keras.Input(shape=input_label_shape)    
    coarse_feature_map = tf.keras.Input(shape=coarse_input_shape)

    residual_layers = [64] * 3

    result = c7s1(input_label, 32, 'relu', reflect_pad=reflection_padding)
    result = d(result, 64, reflect_pad=False)
    result = tf.keras.layers.Add()([result, coarse_feature_map])
    for k in residual_layers:
        result = R(result, k, reflect_pad=reflection_padding)
    result = u(result, 32)
    #result = c7s1(result, output_channels, 'tanh', reflect_pad=reflection_padding)
    result = conv2D(result, output_channels, 7, 1, reflect_pad=reflection_padding)
    result = tf.keras.layers.Activation('sigmoid')(result)

    return tf.keras.Model(inputs=[input_label, coarse_feature_map], outputs=result)


# Discriminators (must return the list of outputs of each layer for fm loss)

def define_patch_discriminator(label_shape, target_shape):
    
    def conv(units, stride=(2, 2)):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.keras.layers.Conv2D(units, (4, 4), strides=stride, padding='valid', kernel_initializer=initializer)

    def batchnorm():
        initializer = tf.random_normal_initializer(1.0, 0.02)
        return tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1, gamma_initializer=initializer)

    def leaky_relu():
        return tf.keras.layers.LeakyReLU(alpha=0.2)

    conv_units = [(128, (2,2)), (256, (2,2)), (512, (1,1))]
    label_img = tf.keras.Input(shape=label_shape)
    target_img = tf.keras.Input(shape=target_shape)

    result = tf.keras.layers.Concatenate()([label_img, target_img])

    layers = []

    result = conv(64)(result)
    result = leaky_relu()(result)
    layers.append(result)
    
    for i, (filters, stride) in enumerate(conv_units):
        result = conv(filters, stride)(result)
        result = batchnorm()(result)
        if 0 == i:
            print "!!!!!!! WARNING: USING BATCH NORM !!!!!!!!"
            print "PyTorch impl uses InstanceNormalization, but we are using batch size of 1 so."
            sys.stdout.flush()
        result = leaky_relu()(result)
        layers.append(result)
    
    result = conv(1, stride=(1,1))(result)
    result = tf.keras.layers.Activation('sigmoid')(result)
    layers.append(result)

    return tf.keras.Model(inputs=[label_img, target_img], outputs=layers)


def define_model(opt):
    ''' Defines the pix2pixHD model. Returns inputs, outputs and a dict of all the keras models used. 
        The dict is used for loading/saving models and hence the keras models must contain all trainable parameters. '''

    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    input_real = tf.placeholder(tf.float32, (None,) + OUTPUT_SHAPE, name="real")
    input_label = tf.placeholder(tf.float32, (None,) + INPUT_SHAPE, name="label")

    with tf.name_scope("Generator"):
        generator = define_global_generator(INPUT_SHAPE, OUTPUT_SHAPE[-1], reflection_padding=opt.reflect_padding)
    input_fake, _ = generator(input_label, training=True)

    # Pass real and generated images to discriminator
    with tf.name_scope("Discriminator"):
        discriminator = define_patch_discriminator(INPUT_SHAPE, OUTPUT_SHAPE)
    with tf.name_scope("real_activations"):
        real_activations = discriminator([input_real, input_label], training=True)
    with tf.name_scope("gen_activations"):
        gen_activations = discriminator([input_fake, input_label], training=True)

    # Compute losses
    with tf.name_scope("generator_gan_loss"):
        g_loss = generator_gan_loss(gen_activations[-1], lsgan=opt.lsgan)
    with tf.name_scope("discriminator_loss"):
        d_loss = discriminator_loss(real_activations[-1], gen_activations[-1], lsgan=opt.lsgan)

    # Apparently this is needed or batch_norm parameters will not update when optimizing
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        gen_optimizer = tf.train.AdamOptimizer(learning_rate, 0.5).minimize(g_loss, var_list=generator.variables)
        disc_optimizer = tf.train.AdamOptimizer(learning_rate, 0.5).minimize(d_loss, var_list=discriminator.variables)

    # Summaries
    scalar_summaries = [tf.summary.scalar('generator_loss', g_loss),
                        tf.summary.scalar('discriminator_loss', d_loss)]

    image_summaries = [tf.summary.image('real', input_real), tf.summary.image('fake', input_fake)]
    label_tmp = tf.dtypes.cast(tf.expand_dims(tf.dtypes.cast(tf.argmax(input_label[:, :, :, :-1], axis=-1), tf.uint8), -1), tf.float32)
    image_summaries += [tf.summary.image('label', tf.dtypes.cast(255 * (label_tmp / tf.math.reduce_max(label_tmp)), tf.uint8)),
                        tf.summary.image('edge_map', input_label[:, :, :, -1:])]

    inputs = {  'images': input_real, 
                'labels': input_label, 
                'lr': learning_rate}

    outputs = { 'output_scales': input_fake,
                'gen_optimizer': gen_optimizer,
                'disc_optimizer': disc_optimizer,
                'summary': tf.summary.merge(scalar_summaries),
                'image_summary': tf.summary.merge(image_summaries)}

    model_dict = {'generator': generator, 'discriminator': discriminator}
    return inputs, outputs, model_dict

