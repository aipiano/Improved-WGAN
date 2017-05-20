import tensorflow as tf
import inputs
import models
from os.path import join
import time
import numpy as np
import os

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'D:/Dataset/anime_faces', 'The path of directory which stores images.')
flags.DEFINE_string('image_format', 'jpg', 'Image format.')
flags.DEFINE_integer('image_channels', 3, 'The number of channels of images.')
flags.DEFINE_integer('gen_output_size', 64, 'The size of generated images.')
flags.DEFINE_integer('gen_input_dims', 100, 'The noise dimensions.')
flags.DEFINE_integer('gen_first_conv_channels', 512, 'Number of channels of the first upconv layer in generator.')
flags.DEFINE_integer('gen_kernel_size', 5, 'Kernel size of upconvs in generator.')
flags.DEFINE_integer('cri_first_conv_channels', 64, 'Number of channels of the first conv layer in critic.')
flags.DEFINE_integer('cri_kernel_size', 5, 'Kernel size of convs in critic.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('weight_clipping', 0.01, 'Clipping value of weights.')
flags.DEFINE_float('weight_penalty', 10.0, 'Use weight penalty instead of weight clipping if this parameter is lager than 0.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_critic_iterations', 5, 'The number of iterations of the critic per generator iteration.')
flags.DEFINE_integer('num_epochs', 300, 'Number of training epochs.')
FLAGS = flags.FLAGS


def train(_):
    with tf.Graph().as_default():
        # build input data
        images = inputs.read_image(FLAGS.data_dir, FLAGS.image_channels, FLAGS.gen_output_size, FLAGS.image_format,
                                   FLAGS.num_epochs)
        images = inputs.make_batch([images], FLAGS.batch_size, 4)
        noises = inputs.sample_noises(FLAGS.batch_size, FLAGS.gen_input_dims)

        # build model and losses
        dcwgan = models.DCWGAN(images, noises, None, True, FLAGS.gen_output_size, FLAGS.image_channels,
                               FLAGS.gen_first_conv_channels, FLAGS.gen_kernel_size, FLAGS.cri_first_conv_channels,
                               FLAGS.cri_kernel_size)
        loss_c = dcwgan.get_critic_loss(FLAGS.weight_penalty)
        loss_g = dcwgan.get_generator_loss()

        # build optimizers
        vars_c = dcwgan.get_critic_variables()
        vars_g = dcwgan.get_generator_variables()
        step_c = tf.Variable(0, trainable=False, name='step_c')
        step_g = tf.Variable(0, trainable=False, name='step_g')
        if FLAGS.weight_penalty > 0:
            train_g = tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5, 0.9).minimize(loss_g, step_g, vars_g, name='Adam_g')
            train_c = tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5, 0.9).minimize(loss_c, step_c, vars_c, name='Adam_c')
        else:
            train_g = tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(loss_g, step_g, vars_g, name='RMSProp_g')
            minimize_c = tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(loss_c, step_c, vars_c, name='RMSProp_c')
            with tf.control_dependencies([minimize_c]):
                train_c = [p.assign(tf.clip_by_value(p, -FLAGS.weight_clipping, FLAGS.weight_clipping)) for p in vars_c]

        # build_summaries
        tf.summary.scalar('losses/generator', loss_g)
        tf.summary.scalar('losses/critic', -loss_c)  # flip the sign back.
        fake_images = dcwgan.get_generator() + 1.0
        fake_images *= 127.5
        fake_images = tf.cast(tf.clip_by_value(fake_images, 0, 255), tf.uint8)
        real_images = images + 1.0
        real_images *= 127.5
        real_images = tf.cast(tf.clip_by_value(real_images, 0, 255), tf.uint8)
        tf.summary.image('real', real_images, max_outputs=20)
        tf.summary.image('fake', fake_images, max_outputs=20)
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./logs')

        # build variable saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            latest_ckpt = tf.train.latest_checkpoint('./checkpoints')
            if latest_ckpt is not None:
                print('restore checkpoint : %s.' % latest_ckpt)
                saver.restore(sess, latest_ckpt)

            start_time = time.time()
            print('start training...')
            step = 0
            try:
                while not coord.should_stop():
                    sec_per_iteration = time.time()

                    # update critic
                    num_critic_iterations = 30 if (step < 5 or step % 500 == 0) else FLAGS.num_critic_iterations
                    for i in range(num_critic_iterations):
                        sess.run(train_c)

                    batch_loss_c = sess.run(loss_c)
                    assert not np.isnan(batch_loss_c), 'Model diverged with dis loss = NaN'

                    # update generator
                    _, step = sess.run([train_g, step_g])
                    # assert not np.isnan(batch_loss_g), 'Model diverged with gen loss = NaN'

                    if step % 5 == 0:  # show training status
                        iterations_per_sec = 1.0 / (time.time() - sec_per_iteration)
                        print('#iteration %d. loss_c: %.3f. (%.3f iterations/sec)' %
                              (step, -batch_loss_c, iterations_per_sec))

                    if step % 50 == 0:
                        summary = sess.run(merged_summary)
                        summary_writer.add_summary(summary, global_step=step)

                    # Save the model checkpoint periodically.
                    if step % 1000 == 0:
                        saver.save(sess, join('./checkpoints', 'model.ckpt'), global_step=step)

            except tf.errors.OutOfRangeError:
                print('reach num_epochs. training finished.')
                print('saving final checkpoints...')
                saver.save(sess, join('./checkpoints', 'model.ckpt'), global_step=step)
            finally:
                summary = sess.run(merged_summary)
                summary_writer.add_summary(summary, step)
                coord.request_stop()  # stop all threads
                print('done!')

            coord.join(threads)  # wait threads to exit
            duration = (time.time() - start_time) / 3600.0
            print('total time: %f hours' % duration)

if __name__ == '__main__':
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    tf.app.run(train)
