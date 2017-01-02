import tensorflow as tf

from os import path
from glob import glob
from model import simple

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 30,
    """Number of epochs to train on.""")
tf.app.flags.DEFINE_integer('num_threads', 4,
    """Number of parallel threads to use.""")

tf.app.flags.DEFINE_integer('filter_size', 3,
    """Filter size of conv weights.""")
tf.app.flags.DEFINE_integer('patch_size', 30,
    """Patch size on us, mr volume slices.""")
tf.app.flags.DEFINE_integer('batch_size', 30,
    """Batch size of us, mr pairs for training.""")

tf.app.flags.DEFINE_integer('threshold', 1,
    """Remove patches with less or equal reduced sum.""")

tf.app.flags.DEFINE_string('log_path', '/tmp/mrtous',
    """Path to write summary events to.""")
tf.app.flags.DEFINE_string('record_path', 'data/test.tfrecord',
    """Path or wildcard to tfrecord to use for testing.""")
tf.app.flags.DEFINE_string('variable_path', '/tmp/mrtous/var.ckpt',
    """Filename of variable checkpoints.""")

def main(_):
    model = simple.Model(
        threshold=FLAGS.threshold,
        num_epochs=FLAGS.num_epochs,
        num_threads=FLAGS.num_threads,
        filter_size=FLAGS.filter_size,
        patch_size=FLAGS.patch_size,
        batch_size=FLAGS.batch_size)

    mr, us = model.placeholder()
    us_ = model.interference(mr)
    loss = model.loss(us, us_)

    train, global_step = model.train(loss)
    batch = model.inputs(glob(FLAGS.record_path))

    tf.summary.image('us_', us_, max_outputs=1)
    tf.summary.image('us', batch[1], max_outputs=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)

        try:
            while not coord.should_stop():
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _, _, step, norm, summary = sess.run(
                    [train, us_, global_step, loss, merged],
                    feed_dict={
                        mr: batch[0].eval(),
                        us: batch[1].eval(),
                    }, options=run_options, run_metadata=run_metadata)

                if step % 100 == 0:
                    writer.add_run_metadata(run_metadata,
                        'step{}'.format(step), step)
                    writer.add_summary(summary, step)

                    print('step: {}, norm: {:.0f}'.format(step, norm))

                global_step += 1

        except tf.errors.OutOfRangeError as error:
            coord.request_stop(error)
        finally:
            writer.close()

            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()