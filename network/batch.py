import random

class StandardBatchIterator:
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def n_updates_per_epoch(self):
        return len(list(range(0, len(self.data), self.batch_size)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
        for start in range(0, len(self.data), self.batch_size):
            yield self.data[start : start + self.batch_size]