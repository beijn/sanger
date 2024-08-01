from collections import defaultdict
import time

__EXECUTION_TIMES__ = defaultdict(list)

def timeit(f):
  def timed(*args, **kw):
    global __EXECUTION_TIMES__
    ts = time.time()
    result = f(*args, **kw)
    te = time.time()
    __EXECUTION_TIMES__[f.__name__].append(te-ts)
    return result
  return timed

def print_times():
  for k, v in __EXECUTION_TIMES__.items():
    print(f'Execution time: {k} was executed {(n:=len(v))} times and took {sum(v)/n} seconds on average, in total {sum(v)} seconds.')
