from layers import *


class DCWGAN:
    def __init__(self, images, noises, labels=None, is_training=True,
                 gen_output_size=64, gen_output_channels=3, gen_first_conv_channels=512, gen_kernel_size=5,
                 cri_first_conv_channels=64, cri_kernel_size=5):
        """
        :param images: real images
        :param noises: noises sampled from a distribution
        :param labels: optional labels
        """
        assert gen_output_size % 16 == 0
        assert gen_first_conv_channels % 8 == 0

        self.gen_output_size = gen_output_size
        self.gen_output_channels = gen_output_channels
        self.gen_first_conv_channels = gen_first_conv_channels
        self.gen_kernel_size = gen_kernel_size
        self.cri_first_conv_channels = cri_first_conv_channels
        self.cri_kernel_size = cri_kernel_size

        self.g = self.__generator(noises, labels, is_training, 'Generator')
        self.c_real = self.__critic(images, labels, is_training, scope='Critic')
        self.c_fake = self.__critic(self.g, labels, is_training, reuse=True, scope='Critic')

        alphas = tf.random_uniform([images.get_shape()[0].value, 1, 1, 1])
        self.xi = alphas * images + (1 - alphas) * self.g
        self.c_interp = self.__critic(self.xi, labels, is_training, reuse=True, scope='Critic')

    def get_generator_loss(self):
        return -tf.reduce_mean(self.c_fake)

    def get_critic_loss(self, weight_penalty=10):
        if weight_penalty <= 0:
            # flip the sign of critic loss so that we can minimize it instead of maximizing.
            return tf.reduce_mean(self.c_fake) - tf.reduce_mean(self.c_real)
        else:
            # weight penalty. see "Improved Training of Wasserstein GANs".
            gradients = tf.gradients(self.c_interp, self.xi)
            gradients = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
            penalty = tf.reduce_mean(tf.square(gradients - 1))
            return tf.reduce_mean(self.c_fake) - tf.reduce_mean(self.c_real) + weight_penalty * penalty

    def get_critic(self):
        """
        :return: c_real
        """
        return self.c_real

    def get_generator(self):
        return self.g

    def get_critic_variables(self):
        vs = [v for v in tf.trainable_variables() if v.name.startswith('Critic/')]
        return vs

    def get_generator_variables(self):
        vs = [v for v in tf.trainable_variables() if v.name.startswith('Generator/')]
        return vs

    def __generator(self, inputs, labels=None, is_training=True, scope=None):
        """
        :param inputs: A rank-2 Tensor in shape [batch_size, noises]
        :param labels: A rank-2 Tensor in shape [batch_size, labels]
        :param is_training:
        :param scope:
        :return: images with pixel range of [-1, 1]
        """
        s16 = self.gen_output_size // 16

        c1 = self.gen_first_conv_channels
        c2 = c1 // 2
        c4 = c2 // 2
        c8 = c4 // 2

        kernel_size = self.gen_kernel_size

        with tf.variable_scope(scope, 'Generator'):
            if labels is not None:
                inputs = tf.concat(1, [inputs, labels])

            net = tf.reshape(linear(inputs, s16 * s16 * c1), [-1, s16, s16, c1])
            net = leaky_relu(batch_norm(net, is_training=is_training))
            net = leaky_relu(batch_norm(resize_conv2d(net, c2, kernel_size, 2, 1), is_training=is_training))
            net = leaky_relu(batch_norm(resize_conv2d(net, c4, kernel_size, 2, 1), is_training=is_training))
            net = leaky_relu(batch_norm(resize_conv2d(net, c8, kernel_size, 2, 1), is_training=is_training))
            net = tanh(resize_conv2d(net, self.gen_output_channels, kernel_size, 2, 1, bias=True))
            return net

    def __critic(self, inputs, labels=None, is_training=True, reuse=None, scope=None):
        """
        :param inputs: A rank-4 Tensor in shape [batch_size, rows, cols, channels]
        :param labels: A rank-2 Tensor in shape [batch_size, labels]
        :param is_training:
        :param reuse:
        :param scope:
        :return: critic scores in shape [batch_size, 1]
        """
        c1 = self.cri_first_conv_channels
        c2 = c1 * 2
        c4 = c2 * 2
        c8 = c4 * 2

        s16 = self.gen_output_size // 16
        kernel_size = self.cri_kernel_size

        with tf.variable_scope(scope, 'Critic', reuse=reuse):
            if labels is not None:
                embedding = tf.reshape(linear(labels, s16 * s16, bias=True), [-1, s16, s16, 1])
                inputs = tf.concat(3, [inputs, embedding])

            net = leaky_relu(conv2d(inputs, c1, kernel_size, 2, bias=True))
            # net = leaky_relu(batch_norm(conv2d(net, c2, kernel_size, 2), is_training=is_training))
            # net = leaky_relu(batch_norm(conv2d(net, c4, kernel_size, 2), is_training=is_training))
            # net = leaky_relu(batch_norm(conv2d(net, c8, kernel_size, 2), is_training=is_training))

            # we should not use batch_norm in critic since it changes the original problem;
            # the one-norm-gradient property will never hold.
            net = leaky_relu(conv2d(net, c2, kernel_size, 2, bias=True))
            net = leaky_relu(conv2d(net, c4, kernel_size, 2, bias=True))
            net = leaky_relu(conv2d(net, c8, kernel_size, 2, bias=True))
            scores = linear(tf.reshape(net, [-1, s16 * s16 * c8]), 1, bias=True)
            return scores

