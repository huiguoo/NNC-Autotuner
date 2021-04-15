import argparse
import csv
import json
import logging
import os
import sys
import timeit
from ast import literal_eval
from collections import Counter
from subprocess import check_call

import pandas as pd
import torch
import torch.nn
from numpy import median

from .operators.matmul import _matmul
from .operators.convolution import _convolution

import nnctuner.utils
from .utils import kernel_arena_scope, Once

log = logging.getLogger(__name__)
stats = Counter()
results = []


def measure_testcase(cfg, op, init_args, input_shape):
    with kernel_arena_scope():
        if op == "matmul":
            op = _matmul(*input_shape)
        if op == "conv2d":
            op = _convolution(init_args, input_shape)
        op.verbose = args.verbose
        if cfg:
            op.apply_config(cfg)
        op.validation()

        inputs = op.gen_inputs()
        sec1 = median(timeit.repeat(lambda: op.baseline(inputs), number=args.times, repeat=args.repeat))
        sec2 = median(timeit.repeat(lambda: op.run(inputs), number=args.times, repeat=args.repeat))
        gf = op.gflops() * args.times

        return gf / sec1, gf / sec2, sec1 / sec2

def report_testcase(op, init_args, input_shape):
    print(f"{op}/{init_args}/{input_shape}")
    if args.noconf:
        pytorch, autotuned, speedup = measure_testcase(
            None, op, init_args, input_shape)
    else:
        filename = f"./configs/{op}{repr(init_args)},{repr(input_shape)}.json"
        if args.autotune or not os.path.exists(filename):
            check_call([
                sys.executable,
                sys.argv[0],
                "autotune",
                "--op", repr(op),
                "--init", repr(init_args),
                "--input", repr(input_shape),
                f"--test-limit={args.test_limit}",
                f"--verbose={args.verbose}",
            ])
        cfg = json.load(open(filename))
        pytorch, autotuned, speedup = measure_testcase(
            cfg, op, init_args, input_shape)
    print(f"{pytorch:.1f} gflops => {autotuned:.1f} gflops ({speedup:.2f}x)")
    results.append([repr(op), repr(init_args), repr(input_shape),
                    f"{pytorch:.4f}", f"{autotuned:.4f}", f"{speedup:.4f}"])
    if speedup >= 1:
        stats["speedup_count"] += 1
        stats["speedup_factor"] += speedup
    stats["total"] += 1
    sys.stdout.flush()


def main(argv=None):
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--times', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--case', type=int)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--autotune', action="store_true")
    parser.add_argument('--noconf', action="store_true")
    parser.add_argument('--limit', '-l', default=50, type=int)
    parser.add_argument('--test-limit', default=500, type=int)
    parser.add_argument('--testcases-filename', default="testcases.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    with nnctuner.utils.timer("total"):
        first = Once()
        #testcases = pd.read_csv(args.testcases_filename).sort_values("gflops")
        testcases = pd.read_csv(args.testcases_filename)
        for _, row in testcases.iterrows():
            op = row["op"]
            init = literal_eval(row["init_args"])
            input_shape = literal_eval(row["input"])
            if first(op, init, input_shape) and (args.case is None or args.case == len(first)):
            #if args.case is None:
                sys.stdout.write(f"{len(first)}: ")
                report_testcase(op, init, input_shape)
            if len(first) >= args.limit and args.case is None:
                break

    stats["speedup_factor"] /= max(1, stats["speedup_count"])

    with open("results.csv", "w") as fd:
        csv.writer(fd).writerows(
            [["op", "init", "input", "pytorch_gflops", "autotuner_gflops", "speedup"]] +
            results)

    log.info("STATS %s", [f"{k}:{v}" for k, v in sorted(stats.items())])
    log.info("TIMERS %s", [f"{k}:{v:.2f}" for k, v in nnctuner.utils.timers.most_common()])

if __name__ == '__main__':
    main()
