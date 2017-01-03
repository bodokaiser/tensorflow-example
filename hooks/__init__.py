import tensorflow as tf

from .signal import SignalHandlerHook

def get_global_step():
    return tf.train.get_global_step(tf.get_default_graph())

class LoggingHook(tf.train.LoggingTensorHook):

    def __init__(self, tensors, steps):
        super().__init__({'step': get_global_step(), **tensors},
            every_n_iter=steps)

class SummarySaverHook(tf.train.SummarySaverHook):

    def __init__(self, logdir, steps):
        super().__init__(output_dir=logdir, save_steps=steps,
            summary_op=tf.summary.merge_all())

class CheckpointSaverHook(tf.train.CheckpointSaverHook):

    def __init__(self, logdir, steps, scaffold):
        super().__init__(checkpoint_dir=logdir, save_steps=steps,
            scaffold=scaffold)