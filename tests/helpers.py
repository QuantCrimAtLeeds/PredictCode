import io
import numpy as np

# ARGH: https://github.com/pytest-dev/pytest/issues/2180

class MockOpen():
    def __init__(self, string_data):
        self._original_open = open
        self.data = string_data
        self.filter = FirstOnlyMockOpenFilter()
        self._calls = []

    def __call__(self, *args, **kwargs):
        if self.filter(args, kwargs):
            self.calls.append((args, kwargs))
            return io.StringIO(self.data)#, newline="\n")
        return self._original_open(*args, **kwargs)

    @property
    def calls(self):
        return self._calls

class FirstOnlyMockOpenFilter():
    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.count == 1
    
    
class RandomCyclicBuffer():
    def __init__(self, numbers):
        self._numbers = numbers
        self.offset = -1
        
    def _one_dimensional_array(self, length):
        out = np.empty(length)
        for i in range(length):
            self.offset += 1
            if self.offset == len(self._numbers):
                self.offset = 0
            out[i] = self._numbers[self.offset]
        return out
        
    def __call__(self, size=None):
        if size is None:
            return self._one_dimensional_array(1)[0]
        return self._one_dimensional_array(np.prod(size)).reshape(size)