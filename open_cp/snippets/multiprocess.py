import concurrent.futures
import multiprocessing

def func(q=None):
    return 5

class Wrapper():
    def __init__(self, task):
        self._task = task

    def __call__(self):
        return self._task()


def fails_silently():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(lambda : func())]

        for result in concurrent.futures.as_completed(futures):
            print(result.result())

def fails_loudly():
    try:
        with multiprocessing.Pool() as pool:
            w = Wrapper(func)
            result = pool.apply_async(lambda : w)
            assert result.get() ==  5
    except Exception as ex:
        print("Caught", ex, type(ex))

if __name__ == "__main__":
    fails_loudly()