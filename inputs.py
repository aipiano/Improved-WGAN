import tensorflow as tf
from os.path import join


def read_image(data_path, required_channels, required_size, image_format='jpg', num_epochs=None):
    """
    read an image and scale it to the required size.
    :param data_path: a directory which stores image files
    :param required_channels: must be the same as the original image
    :param required_size: patch size to output
    :param image_format: image file format. default is 'jpg'
    :param num_epochs:
    :return: an image patch that the pixel range is [-1, 1]
    """
    image_format = image_format.lower()
    assert image_format == 'jpg' or image_format == 'png'
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(join(data_path, '*.' + image_format)),
                                                    num_epochs, capacity=256)
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)

    if image_format == 'jpg':
        patch = tf.image.decode_jpeg(image_file, required_channels)
    elif image_format == 'png':
        patch = tf.image.decode_png(image_file, required_channels)
    else:
        raise ValueError('Unsupported image format.')

    patch = tf.to_float(patch)
    patch /= 127.5
    patch -= 1.0
    patch = tf.image.resize_nearest_neighbor([patch], [required_size, required_size])[0]
    patch.set_shape([required_size, required_size, required_channels])
    return patch


def sample_noises(batch_size, num_dims):
    return tf.random_normal([batch_size, num_dims])


def make_batch(tensors, batch_size, num_thread=1):
    batch = tf.train.batch(tensors, batch_size, num_thread, capacity=32 + 3*batch_size)
    return batch
