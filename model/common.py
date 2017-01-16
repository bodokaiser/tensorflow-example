DEFAULT_THRESHOLD = 0
DEFAULT_BATCH_SIZE = 10
DEFAULT_FILTER_SIZE = 3
DEFAULT_NUM_EPOCHS = 10
DEFAULT_NUM_THREADS = 8
DEFAULT_NUM_FILTERS = 3

class Model(object):

    def __init__(self,
        filter_size=DEFAULT_FILTER_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        num_epochs=DEFAULT_NUM_EPOCHS,
        num_threads=DEFAULT_NUM_THREADS,
        num_filters=DEFAULT_NUM_FILTERS,
        threshold=DEFAULT_THRESHOLD):
        self._threshold = threshold
        self._batch_size = batch_size
        self._filter_size = filter_size
        self._num_epochs = num_epochs
        self._num_threads = num_threads
        self._num_filters = num_filters

    @property
    def threshold(self):
        return self._threshold

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def filter_size(self):
        return self._filter_size

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def num_threads(self):
        return self._num_threads

    @property
    def num_filters(self):
        return self._num_filters