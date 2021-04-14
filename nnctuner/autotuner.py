import argparse
import ast
import json
import logging
import multiprocessing
import os, sys
import random
import shutil
import timeit

import opentuner
import torch
from numpy import median
from opentuner import Result
from opentuner.measurement import MeasurementInterface
from opentuner.search.bandittechniques import AUCBanditMetaTechnique
from opentuner.search.evolutionarytechniques import NormalGreedyMutation, UniformGreedyMutation
from opentuner.search.manipulator import BooleanParameter
from opentuner.search.manipulator import ConfigurationManipulator
from opentuner.search.manipulator import EnumParameter
from opentuner.search.manipulator import NumericParameter
from opentuner.search.manipulator import PermutationParameter

from .operators.matmul import _matmul
from .operators.convolution import _convolution
from .utils import kernel_arena_scope

log = logging.getLogger(__name__)
REPEAT = 5

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument("--op")
parser.add_argument("--init")
parser.add_argument("--input")
parser.add_argument("--verbose")
parser.set_defaults(**vars(parser.parse_args([
    "--no-dups",
    #"--stop-after=600",
    "--test-limit=1000",
    #f"--parallelism={multiprocessing.cpu_count()}",
    #"--parallel-compile",
    "--technique=nnctuner",
    "--op", "matmul",
    "--init", "",
    "--input", "(32, 32, 32)",
    "--verbose", False,
])))

#opentuner.search.technique.register(AUCBanditMetaTechnique([
#    NormalGreedyMutation(name="Normal5", mutation_rate=0.05),
#    NormalGreedyMutation(name="Normal10", mutation_rate=0.10),
#    DifferentialEvolution(population_size=10, cr=0.8, information_sharing=1),
#    PatternSearch(),
#], name="convtuner"))
#DifferentialEvolution(population_size=10, cr=0.8, information_sharing=1),

# Used for finetuning
opentuner.search.technique.register(AUCBanditMetaTechnique([
    NormalGreedyMutation(name="Normal4", mutation_rate=0.04),
    NormalGreedyMutation(name="Normal8", mutation_rate=0.08),
    NormalGreedyMutation(name="Normal16", mutation_rate=0.16),
    UniformGreedyMutation(name="Uniform32", mutation_rate=0.32)
], name="nnctuner"))


class autoTuner(MeasurementInterface):
    def __init__(self, args=None):
        manipulator = ConfigurationManipulator()
        op = ast.literal_eval(args.op)
        init_args = ast.literal_eval(args.init)
        input_shape = ast.literal_eval(args.input)
        if op == 'matmul':
            self.op = _matmul(*input_shape)
        if op == 'conv2d':
            self.op = _convolution(init_args, input_shape)
        assert(self.op)
        self.base = self.op.ref
        self.inputs = self.op.gen_inputs()
        self.op.verbose = args.verbose

        # Dry run codegen to capture config
        manipulator = self.op.create_search_space(manipulator)
        assert len(manipulator.params)
        # import pprint; pprint.pprint(manipulator.random())
        sec = float(median(timeit.repeat(lambda: self.op.run(self.inputs), number=1, repeat=REPEAT)))
        self.times = max(1, int(0.1 / sec))

        super(autoTuner, self).__init__(
            args=args,
            project_name="NNCAutoTuner",
            program_name=repr(self.op)+repr(init_args),
            program_version=repr(input_shape),
            manipulator=manipulator,
        )

    def default_config(self):
        return {p.name: min_value(p) for p in self.manipulator().params}

    def compile_and_run(self, desired_result, input, limit):
        return Result(time=self.measure_cfg(desired_result.configuration.data))

    def measure_cfg(self, cfg):
        self.op.apply_config(cfg)
        self.op.run(self.inputs)  # warmup
        sec = median(timeit.repeat(lambda: self.op.run(self.inputs), number=self.times, repeat=REPEAT))
        return float(sec)

    def save_final_config(self, configuration):
        cfg = configuration.data
        sec = self.measure_cfg(cfg)
        gf = self.op.gflops() * self.times / sec
        print(f"Final configuration ({sec:.2f}s, {gf:.1f} gflops): {self.config_filename()}")
        with open(self.config_filename(), "w") as fd:
            json.dump(cfg, fd, indent=2, sort_keys=True)
            fd.write("\n")

    def config_filename(self):
        os.path.exists("../configs") or os.mkdir("../configs")
        return f"./configs/{self.program_name()},{self.program_version()}.json"


def min_value(p):
    if isinstance(p, NumericParameter):
        return p.min_value
    if isinstance(p, BooleanParameter):
        return False
    if isinstance(p, PermutationParameter):
        return list(p._items)
    if isinstance(p, EnumParameter):
        return p.options[0]
    assert False


def main(args):
    os.path.exists("bins") or os.mkdir("bins")
    try:
        multiprocessing.set_start_method("spawn")
        opentuner.init_logging()
        with kernel_arena_scope():
            autoTuner.main(parser.parse_args(args))
    finally:
        shutil.rmtree("./bins")


if __name__ == '__main__':
    main()
