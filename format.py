import os
import argparse
import tensorflow as tf

from ioutil import minc
from ioutil import tfrecord

def count(source):
    with tf.Session() as session:
        count = sum([1 for e in tfrecord.iter_tfrecord(source)])

    print('counted {} slices in {}'.format(count, source))

def convert(mr_source, us_source, target):
    dirname = os.path.dirname(target)

    if dirname != '':
        os.makedirs(dirname, exist_ok=True)

    mr = minc.read_minc(mr_source)
    us = minc.read_minc(us_source)
    assert len(mr) == len(us)

    tfrecord.write_tfrecord(target,
        [tfrecord.encode_example(mr[i], us[i]) for i in range(len(mr))])

    print('converted {} and {} to {} ({} slices)'.format(
        mr_source, us_source, target, len(mr)))

def partition(source, target, length, offset):
    count = 0
    slices = []

    for e in tfrecord.iter_tfrecord(source):
        if count > offset:
            if length > 0:
                if count - offset < length + 1:
                    slices.append(e)
                else:
                    break
            else:
                slices.append(e)
        count += 1

    tfrecord.write_tfrecord(target, slices)

    print('extracted {} slices from {} starting at {} and wrote to {}'.format(
        len(slices), source, FLAGS.offset, target))

def main(args):
    if args.action == 'count':
        count(args.source)
    elif args.action == 'convert':
        convert(args.mr_source, args.us_source, args.target)
    elif args.action == 'partition':
        partition(args.source, args.target, args.length, args.offset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='format minc and tfrecord datasets')
    subparsers = parser.add_subparsers(dest='action')

    subparser1 = subparsers.add_parser('count')
    subparser1.add_argument('source', help='path to tfrecord source')

    subparser2 = subparsers.add_parser('convert')
    subparser2.add_argument('mr_source', help='path to minc mr source')
    subparser2.add_argument('us_source', help='path to minc us source')
    subparser2.add_argument('target', help='path to tfrecord target')

    subparser3 = subparsers.add_parser('partition')
    subparser3.add_argument('--length', help='amount of slices', type=int)
    subparser3.add_argument('--offset', help='amount of slices to skip', type=int)
    subparser3.set_defaults(length=0, offset=0)

    main(parser.parse_args())