import os
import argparse
import tensorflow as tf

from model import simple
from hooks import signal

class Run(object):

    def __init__(self, model, conv1, conv2):
        self._model = model
        self._conv1 = conv1
        self._conv2 = conv2

    def build(self):
        self.re = self._model.interference(self._conv1, self._conv2, self.mr)
        self.df = self.us-self.re
        self.loss = self._model.loss(self.df)

    def summarize(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.image('mr', self.mr)
        tf.summary.image('us', self.us)
        tf.summary.image('re', self.re)
        tf.summary.image('df', self.df)

class PatchesRun(Run):

    def build(self, filename, epochs=0, train=False):
        if epochs > 0:
            filename = tf.train.limit_epochs(tf.convert_to_tensor(filename),
                epochs)

        self.mr, self.us = self._model.patches(filename)

        super().build()

        if train:
            self.train = self._model.train(self.loss)

class ImagesRun(Run):

    def build(self, filename):
        self.mr, self.us = self._model.images(filename)

        super().build()

        self.re = tf.where(tf.equal(self.us, 0),
            tf.zeros_like(self.re), self.re)

def main(args):
    model = simple.Model(
        filter_num=args.filter_num,
        filter_size=args.filter_size)

    conv1 = model.conv1()
    conv2 = model.conv2()

    with tf.name_scope('train'):
        with tf.name_scope('patches'):
            patches_train_run = PatchesRun(model, conv1, conv2)
            patches_train_run.build(args.train, args.epochs, train=True)
            patches_train_run.summarize()
        with tf.name_scope('images'):
            images_train_run = ImagesRun(model, conv1, conv2)
            images_train_run.build(args.train)
            images_train_run.summarize()

    with tf.name_scope('valid'):
        with tf.name_scope('patches'):
            patches_valid_run = PatchesRun(model, conv1, conv2)
            patches_valid_run.build(args.valid)
            patches_valid_run.summarize()
        with tf.name_scope('images'):
            images_valid_run = ImagesRun(model, conv1, conv2)
            images_valid_run.build(args.train)
            images_valid_run.summarize()

    with tf.name_scope('test'):
        with tf.name_scope('patches'):
            patches_test_run = PatchesRun(model, conv1, conv2)
            patches_test_run.build(args.test)
            patches_test_run.summarize()
        with tf.name_scope('images'):
            images_test_run = ImagesRun(model, conv1, conv2)
            images_test_run.build(args.test)
            images_test_run.summarize()

    with tf.Session() as session:
        session.run([
            tf.local_variables_initializer(),
            tf.global_variables_initializer(),
        ])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord)

        writer = tf.summary.FileWriter(args.logdir, session.graph)
        merged = tf.summary.merge_all()

        try:
            step = 0

            while not coord.should_stop():
                session.run([
                    patches_train_run.train,
                    images_train_run.loss,
                    patches_valid_run.loss,
                    images_valid_run.loss,
                    patches_test_run.loss,
                    images_test_run.loss,
                ])

                if step % 50 == 0:
                    print('step: {}'.format(step))
                    writer.add_summary(session.run(merged), step)

                step += 1
        except tf.errors.OutOfRangeError as error:
            coord.request_stop(error)
        finally:
            coord.request_stop()
            coord.join(threads)

        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='evaluate model on a given dataset')
    parser.add_argument('--epochs', type=int,
        help='number of epochs to run')
    parser.add_argument('--test', nargs='+',
        help='tfrecords to use for testing')
    parser.add_argument('--train', nargs='+',
        help='tfrecords to use for training')
    parser.add_argument('--valid', nargs='+',
        help='tfrecords to use for validation')
    parser.add_argument('--logdir',
        help='directory to write events to')
    parser.add_argument('--filter_num',
        help='number of filters')
    parser.add_argument('--filter_size',
        help='height and width of filters')
    parser.set_defaults(epochs=30, filter_num=3, filter_size=7,
        train=['data/train.tfrecord'],
        valid=['data/valid.tfrecord'],
        test=['data/test.tfrecord'],
        logdir='/tmp/mrtous')

    main(parser.parse_args())