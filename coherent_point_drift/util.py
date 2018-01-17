class frange:
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step

    def __len__(self):
        return int((self.stop - self.start) / self.step)

    def __getitem__(self, idx):
        if not 0 <= idx < (self.stop - self.start) / self.step:
            raise IndexError("Index {} out of range".format(idx))
        return self.start + self.step * idx

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def last(seq):
    from functools import reduce

    return reduce(lambda acc, x: x, seq)
