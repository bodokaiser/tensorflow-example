import os
import argparse
import tensorflow as tf

from model import simple
from hooks import signal

def test(model, conv1, conv2, filenames):
    mr, us = model.images(filenames)
    re = model.interference(conv1, conv2, mr)

    loss = model.loss(re-us)

    tf.summary.scalar('loss', loss)
    tf.summary.image('mr', mr)
    tf.summary.image('us', us)
    tf.summary.image('re', re)
    tf.summary.image('df', re-us)

    return [loss]

def train(model, conv1, conv2, filenames):
    mr, us = model.patches(filenames)
    re = model.interference(conv1, conv2, mr)

    loss = model.loss(re-us)
    train = model.train(loss)

    tf.summary.scalar('loss', loss)
    tf.summary.image('mr', mr)
    tf.summary.image('us', us)
    tf.summary.image('re', re)
    tf.summary.image('df', re-us)

    return [train, loss]

def main(args):
    model = simple.Model(
        filter_num=args.filter_num,
        filter_size=args.filter_size)

    conv1 = model.conv1()
    conv2 = model.conv2()

    with tf.name_scope('testing'):
        test_op = test(model, conv1, conv2, args.train)
    with tf.name_scope('training'):
        train_op = train(model, conv1, conv2, args.train)

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
                _, _, summary = session.run([test_op, train_op, merged])

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
    parser.add_argument('--train', nargs='+',
        help='tfrecords to use for training')
    parser.add_argument('--logdir',
        help='directory to write events to')
    parser.add_argument('--filter_num',
        help='number of filters')
    parser.add_argument('--filter_size',
        help='height and width of filters')
    parser.set_defaults(train=['data/train.tfrecord'], logdir='/tmp/mrtous',
        steps=1000, filter_num=3, filter_size=7)

    main(parser.parse_args())