# https://github.com/keras-rl/keras-rl/blob/d9e3b64a20f056030c02bfe217085b7e54098e48/rl/memory.py#L10
import numpy

class RingBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = numpy.empty(maxlen, dtype=numpy.object)#[None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def sample(self, batchSize):
        batch = []
        for num in numpy.random.randint(0, self.length, batchSize):
            batch.append(self[num])
        return batch