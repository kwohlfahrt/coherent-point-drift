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

def argmin(seq, key=lambda x: x):
    amin = next(s)
    for s in seq:
        if key(s) < key(amin):
            amin = s
    return current
