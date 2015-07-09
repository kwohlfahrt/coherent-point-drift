class frange:
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step

    def __getitem__(self, idx):
        if not 0 <= idx < stop / step:
            raise IndexError("Index {} out of range".format(idx))
        return self.step * idx

    def __iter__(self):
        i = self.start
        while i < self.stop:
            yield i
            i += self.step

def last(seq):
    from functools import reduce, partial
    return reduce(lambda acc, x: x, seq)
