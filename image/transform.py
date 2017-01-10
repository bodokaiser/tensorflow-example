import tensorflow as tf

import math

def get_shape(tensor):
    shape = tensor.get_shape()
    return [dim.value for dim in shape]

def compatible_dim(image_dim, patch_dim):
    return math.ceil(image_dim/patch_dim)*patch_dim

def image_to_patches(image, patch_height, patch_width):
    image_height, image_width, _ = get_shape(image)

    image = tf.reshape(image, [image_height // patch_height, patch_height,
        -1, patch_width, 1])
    image = tf.transpose(image, [0, 2, 1, 3, 4])

    return tf.reshape(image, [-1, patch_height, patch_width, 1])

def patches_to_image(patches, image_height, image_width):
    _, patch_height, patch_width, _ = get_shape(patches)

    image = tf.reshape(patches, [image_height // patch_height,
        image_width // patch_width, patch_height, patch_width, 1])
    image = tf.transpose(image, [0, 2, 1, 3, 4])

    return tf.reshape(image, [image_height, image_width, 1])

def filter_indices(patches, value):
    indices = tf.where(tf.greater(tf.reduce_sum(patches, [1, 2]), value))
    return indices[:, 0]