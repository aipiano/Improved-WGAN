import tensorflow as tf
import tensorflow.contrib.slim as slim

# some shortcuts for tensorflow layers
relu = tf.nn.relu
tanh = tf.nn.tanh
sigmoid = tf.nn.sigmoid


def instance_norm(inputs, epsilon=1e-9, scope=None):
    """
    Instance Normalization. see paper "Instance Normalization: The Missing Ingredient for Fast Stylization"
    :param inputs:
    :param epsilon:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope, 'InstanceNorm'):
        # normalize each sample and each channel individually
        mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
        var_inv = tf.rsqrt(var + epsilon)
        return (inputs - mean) * var_inv


def batch_norm(inputs, decay=0.99, center=True, scale=False, epsilon=0.001, is_training=True, scope=None):
    """
    A simplified warp to slim's batch_norm
    :param inputs:
    :param decay:
    :param center: If True, subtract `beta`. If False, `beta` is ignored.
    :param scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    :param epsilon:
    :param is_training:
    :param scope:
    :return:
    """
    return slim.batch_norm(inputs, decay, center, scale, epsilon, is_training=is_training, scope=scope)


def linear(inputs, num_outputs, bias=False, scope=None):
    """
    Fully connected layer
    :param inputs: 2-D tensor with shape [batches, dims]
    :param num_outputs:
    :param bias:
    :param scope:
    :return:
    """
    num_inputs = inputs.get_shape()[-1].value
    stddev = tf.sqrt(1.3*2.0 / num_inputs)  # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
    weight_shape = [num_inputs, num_outputs]    # weight shape must be static

    with tf.variable_scope(scope, 'Linear'):
        weight_init = tf.truncated_normal_initializer(stddev=stddev)
        weights = tf.get_variable('weights', weight_shape, tf.float32, weight_init,
                                  collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES,
                                               tf.GraphKeys.TRAINABLE_VARIABLES])
        outputs = tf.matmul(inputs, weights)

        if bias:
            biases_init = tf.constant_initializer(0.1)
            biases_shape = [num_outputs]
            biases = tf.get_variable('biases', biases_shape, tf.float32, biases_init,
                                     collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES,
                                                  tf.GraphKeys.TRAINABLE_VARIABLES])
            return tf.nn.bias_add(outputs, biases)  # almost the same as tf.add
        return outputs


def leaky_relu(inputs, leaky=0.2, scope=None):
    with tf.variable_scope(scope, 'LeakyReLU'):
        return tf.maximum(inputs, leaky * inputs)


def conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', bias=False, scope=None):
    num_inputs = inputs.get_shape()[-1].value
    weight_shape = [kernel_size, kernel_size, num_inputs, num_outputs]
    fan_in = kernel_size * kernel_size * num_inputs
    stddev = tf.sqrt(1.3*2.0 / fan_in)

    with tf.variable_scope(scope, 'Conv2D'):
        weights_init = tf.truncated_normal_initializer(stddev=stddev)
        weights = tf.get_variable('weights', weight_shape, initializer=weights_init,
                                  collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES,
                                               tf.GraphKeys.TRAINABLE_VARIABLES])
        outputs = tf.nn.conv2d(inputs, weights, [1, stride, stride, 1], padding)

        if bias:
            biases_init = tf.constant_initializer(0.1)
            biases_shape = [num_outputs]
            biases = tf.get_variable('biases', biases_shape, initializer=biases_init,
                                     collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES,
                                                  tf.GraphKeys.TRAINABLE_VARIABLES])
            return tf.nn.bias_add(outputs, biases)  # almost the same as tf.add
        return outputs


def resize_conv2d(inputs, num_outputs, kernel_size, scale, stride=1, bias=False, scope=None):
    """
    An alternative of conv2d_transpose. see http://distill.pub/2016/deconv-checkerboard/
    The true scale to resize images is (scale * stride).
    The padding type is always 'SAME'
    :param inputs:
    :param num_outputs:
    :param kernel_size:
    :param scale:
    :param stride:
    :param bias:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope, 'ResizeConv2D'):
        inputs_shape = tf.shape(inputs)
        resize_factor = scale * stride
        resized_shape = tf.pack([resize_factor * inputs_shape[1], resize_factor * inputs_shape[2]])
        resized_inputs = tf.image.resize_nearest_neighbor(inputs, resized_shape, True)
        return conv2d(resized_inputs, num_outputs, kernel_size, stride, bias=bias)
