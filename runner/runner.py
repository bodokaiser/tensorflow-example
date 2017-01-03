import tensorflow as tf

import hooks

class Runner(object):

    def __init__(self, logdir, save_summary_steps=0, save_checkpoint_steps=0):
        def init_fn(scaffold, session):
            latest_ckpt = tf.train.latest_checkpoint(logdir)

            if latest_ckpt:
                scaffold.saver.restore(session, latest_ckpt)

        self._hooks = [
            hooks.SignalHandlerHook(),
        ]
        self._saver = tf.train.Saver()
        self._scaffold = tf.train.Scaffold(
            saver=self._saver, init_fn=init_fn)

        if save_summary_steps > 0:
            self.add_hook(hooks.SummarySaverHook(logdir, save_summary_steps))
        if save_checkpoint_steps > 0:
            self.add_hook(hooks.CheckpointSaverHook(logdir, save_checkpoint_steps,
                self.scaffold))

    def add_hook(self, hook):
        self._hooks.append(hook)

    @property
    def saver(self):
        return self._saver

    @property
    def scaffold(self):
        return self._scaffold

    def monitored_session(self):
        return tf.train.MonitoredTrainingSession(
            hooks=self._hooks, scaffold=self.scaffold)