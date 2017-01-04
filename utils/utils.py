import tensorflow as tf

def get_global_step():
    graph = tf.get_default_graph()
    return tf.contrib.framework.get_or_create_global_step(graph)