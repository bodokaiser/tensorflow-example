import tensorflow as tf

class Runner(object):

    def __init__(self, vardir):
        def init_fn(scaffold, session):
            latest_ckpt = tf.train.latest_checkpoint(vardir)

            if latest_ckpt:
                scaffold.saver.restore(session, latest_ckpt)

        self._hooks = []
        self._saver = tf.train.Saver()
        self._scaffold = tf.train.Scaffold(
            saver=self._saver, init_fn=init_fn)

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