import abc

class Model(abc.ABC):
    """Registers required parameters and makes them available as properties."""

    def __init__(self, threshold, batch_size, patch_size, filter_size,
        num_epochs, num_threads):
        self._threshold = threshold
        self._batch_size = batch_size
        self._patch_size = patch_size
        self._filter_size = filter_size
        self._num_epochs = num_epochs
        self._num_threads = num_threads

    @property
    def threshold(self):
        return self._threshold

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def filter_size(self):
        return self._filter_size

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def num_threads(self):
        return self._num_threads