import tensorflow as tf

from utils import utils

class StepCounterHook(tf.train.SessionRunHook):
    """Restores and updates global_step after each run."""

    def begin(self):
        self._step = utils.get_global_step()
        self._count = tf.assign_add(self._step, 1)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._count)