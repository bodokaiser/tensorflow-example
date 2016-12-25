import tensorflow as tf

from ioutil.minc import SLICE_SHAPE

TFRECORD_OPTIONS = tf.python_io.TFRecordOptions(
    tf.python_io.TFRecordCompressionType.ZLIB)

def encode_example(us_slice, mr_slice):
    return tf.train.Example(features=tf.train.Features(feature={
        'us': tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[us_slice.tostring()])),
        'mr': tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[mr_slice.tostring()])),
    }))

def decode_example(example):
    features = tf.parse_single_example(example, features={
        'us': tf.FixedLenFeature([], tf.string),
        'mr': tf.FixedLenFeature([], tf.string),
    })
    us = tf.reshape(tf.decode_raw(features['us'], tf.float32),
        SLICE_SHAPE, 'reshape_us')
    mr = tf.reshape(tf.decode_raw(features['mr'], tf.float32),
        SLICE_SHAPE, 'reshape_mr')
    return us, mr

def read_tfrecord(filename, session, options=TFRECORD_OPTIONS):
    examples = tf.python_io.tf_record_iterator(filename, options)
    return sess.run(list(zip(*[decode_example(e) for e in examples])))

def write_tfrecord(filename, us, mr, options=TFRECORD_OPTIONS):
    assert len(us) == len(mr)
    with tf.python_io.TFRecordWriter(filename, options) as writer:
        for i in range(len(us)):
            writer.write(encode_example(us[i], mr[i]).SerializeToString())