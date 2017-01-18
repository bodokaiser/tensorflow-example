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

    def build(self, filename, train=False):
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
        with tf.name_scope('images'):
            images_train_run = ImagesRun(model, conv1, conv2)
            images_train_run.build(args.train)
            images_train_run.summarize()
        with tf.name_scope('patches'):
            patches_train_run = PatchesRun(model, conv1, conv2)
            patches_train_run.build(args.train, train=True)
            patches_train_run.summarize()

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

            while not coord.should_stop() and step < args.steps:
                session.run([
                    patches_train_run.train,
                    images_train_run.loss,
                ])
                summary = session.run(merged)

                if step % 50 == 0:
                    print('step: {}'.format(step))
                    writer.add_summary(summary, step)

                step += 1

        finally:
            coord.request_stop()
            coord.join(threads)

        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='evaluate model on a given dataset')
    parser.add_argument('--steps', type=int,
        help='number of steps to run')
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
    parser.set_defaults(train=['data/train.tfrecord'], logdir='/tmp/mrtous',
        steps=1000, filter_num=3, filter_size=7)

    main(parser.parse_args())