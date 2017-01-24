import tensorflow as tf

SLICE_HEIGHT = 466
SLICE_WIDTH = 394

TFRECORD_OPTIONS = tf.python_io.TFRecordOptions(
    tf.python_io.TFRecordCompressionType.ZLIB)

def encode_example(mr_slice, us_slice):
    return tf.train.Example(features=tf.train.Features(feature={
        'us': tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[us_slice.tostring()])),
        'mr': tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[mr_slice.tostring()])),
    })).SerializeToString()

def decode_example(example):
    features = tf.parse_single_example(example, features={
        'us': tf.FixedLenFeature([], tf.string),
        'mr': tf.FixedLenFeature([], tf.string),
    })
    shape = [SLICE_HEIGHT, SLICE_WIDTH, 1]
    mr = tf.reshape(tf.decode_raw(features['mr'], tf.float32),
        shape, 'reshape_mr')
    us = tf.reshape(tf.decode_raw(features['us'], tf.float32),
        shape, 'reshape_us')
    return mr, us

def iter_tfrecord(filename, options=TFRECORD_OPTIONS):
    return tf.python_io.tf_record_iterator(filename, options)

def read_tfrecord(filename, session, options=TFRECORD_OPTIONS):
    examples = iter_tfrecord(filename, options)
    return session.run(list(zip(*[decode_example(e) for e in examples])))

def write_tfrecord(filename, examples, options=TFRECORD_OPTIONS):
    with tf.python_io.TFRecordWriter(filename, options) as writer:
        for e in examples:
            writer.write(e)