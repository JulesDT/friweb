import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        elapsed_time = (te-ts)*1000 # ms
        return (result, elapsed_time)
    return timed
