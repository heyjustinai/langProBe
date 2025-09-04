from ..benchmark import Benchmark
import dspy

from datasets import load_dataset


class MATHBench(Benchmark):
    def init_dataset(self):
        # Using MATH-Hard dataset which contains only Level 5 (hardest) problems
        raw_datasets = load_dataset("lighteval/MATH-Hard", "default")
        self.dataset = [
            dspy.Example(**x).with_inputs("problem") for x in raw_datasets["train"]
        ]
        self.test_set = [
            dspy.Example(**x).with_inputs("problem") for x in raw_datasets["test"]
        ]
