import tensorflow as tf

from hooks.signal import SignalHandlerHook

def get_global_step():
    graph = tf.get_default_graph()
    return tf.contrib.framework.get_or_create_global_step(graph)

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

class StopAfterNStepsHook(tf.train.StopAtStepHook):

    def __init__(self, steps):
        self._steps = steps
        super().__init__(steps)

    def after_run(self, run_context, run_values):
        if self._steps > 0:
            super().after_run(run_context, run_values)