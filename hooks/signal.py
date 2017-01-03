import signal as sgn
import tensorflow as tf

class SignalHandlerHook(tf.train.SessionRunHook):
    """Stops evaluation when receiving SIGINT signal."""

    def begin(self):
        self._registered = False

    def before_run(self, run_context):
        def handler(signal, frame):
            run_context.request_stop()

        if not self._registered:
            sgn.signal(sgn.SIGINT, handler)

            self._registerd = True