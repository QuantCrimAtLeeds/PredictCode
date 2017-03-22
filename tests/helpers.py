# ARGH: https://github.com/pytest-dev/pytest/issues/2180

import io

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