import tensorflow as tf


def wasserstein(critic_real, critic_fake):
    """
    min max E_r[f(x)] - E_z[f(g(z))]
     g  ||f||_L<=1
    :param critic_real: critic with real data inputs
    :param critic_fake: critic with fake data inputs
    :return: critic_loss, generator_loss
    """
    # flip the sign of critic loss so that we can minimize it instead of maximizing.
    critic_loss = tf.reduce_mean(critic_fake) - tf.reduce_mean(critic_real)
    generator_loss = -tf.reduce_mean(critic_fake)
    return critic_loss, generator_loss


def improved_wasserstein(critic_real, critic_fake, interpolated_x):
    """
    min E_z[f(g(z))] - E_r[f(x)] + lambda*E_xi[(||gradients(f(xi))|| - 1)^2]
     f
    min -E_z[f(g(z))]
     g
    :param critic_real:
    :param critic_fake:
    :param interpolated_x:
    :return:
    """
    # flip the sign of critic loss so that we can minimize it instead of maximizing.
    critic_loss = tf.reduce_mean(critic_fake) - tf.reduce_mean(critic_real)
    generator_loss = -tf.reduce_mean(critic_fake)


