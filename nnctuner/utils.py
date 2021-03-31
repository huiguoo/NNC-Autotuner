import torch

import contextlib
import logging
import time
from collections import Counter
from functools import partial, reduce
from operator import mul

log = logging.getLogger(__name__)

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C._te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

class Once(set):
    def __call__(self, *x):
        return x not in self and (self.add(x) or True)

timers = Counter()

@contextlib.contextmanager
def timer(name):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    timers[name] += t1 - t0
