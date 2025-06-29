#!/usr/bin/env python
"""
Script to evaluate a pre-generated optimized prompt on the HeartDisease benchmark
without running the optimizers again.
"""

import argparse
import os
import sys
from pathlib import Path

import dspy
from langProBe.evaluation import evaluate, evaluate_all
from langProBe.custom_prompt_loader import create_custom_prompt_optimizer
from langProBe.register_benchmark import register_all_benchmarks
from langProBe.analysis import read_evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a custom prompt on the HeartDisease benchmark")
    
    parser.add_argument(
        "--prompt_file",
        help="Path to the JSON file containing the optimized prompt",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--lm",
        help="The language model to use for evaluation",
        type=str,
        default="openai/gpt-4o-mini",
    )

    parser.add_argument(
        "--lm_api_base",
        help="The language model API base to use for evaluation",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--lm_api_key",
        help="The language model API key to use for evaluation",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--file_path",
        help="The file path to save the evaluation results",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--dataset_mode",
        help="The dataset mode to use for evaluation. Options are: full, lite (500), tiny (200), test (20).",
        type=str,
        default="test",
    )
    
    parser.add_argument(
        "--num_threads",
        help="The number of threads to use for evaluation",
        type=int,
        default=8,
    )
    
    parser.add_argument(
        "--program_class",
        help="The program class to evaluate, available options: single, archon, all",
        type=str,
        default="all",
    )
    
    parser.add_argument(
        "--benchmarks",
        help="Comma-separated list of benchmarks to evaluate (e.g., 'HeartDisease,hover')",
        type=str,
        default=None,
    )
    
    args = parser.parse_args()
    
    # Validate that the prompt file exists
    prompt_file_path = Path(args.prompt_file)
    if not prompt_file_path.exists():
        print(f"Error: Prompt file {prompt_file_path} does not exist.")
        sys.exit(1)
    
    # Set up the retrieval model
    rm = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
    
    # Register the benchmarks
    # You can specify which benchmarks to evaluate here
    benchmark_names = []
    
    # Add benchmarks based on command line arguments or use defaults
    if args.benchmarks:
        # Split comma-separated benchmark names
        for benchmark in args.benchmarks.split(','):
            benchmark_name = f".{benchmark.strip()}"
            benchmark_names.append(benchmark_name)
    else:
        # Default to HeartDisease if no benchmarks specified
        benchmark_names = [".HeartDisease"]
    
    # Register all specified benchmarks
    benchmarks = register_all_benchmarks(benchmark_names)
    
    # Create a custom optimizer config
    custom_optimizer = create_custom_prompt_optimizer(str(prompt_file_path))
    
    # Replace the default optimizers with our custom optimizer for each benchmark
    for benchmark_meta in benchmarks:
        benchmark_meta.optimizers = [custom_optimizer]
    
    # Set up the file path for results
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    file_path = args.file_path or f"evaluation_custom_prompt_{current_time}"
    
    # Run the evaluation directly without going through evaluate_all
    # This avoids the double registration issue
    for benchmark_meta in benchmarks:
        evaluate(
            benchmark_meta,
            args.lm,
            rm,
            file_path=file_path,
            suppress_dspy_output=True,
            dataset_mode=args.dataset_mode,
            num_threads=args.num_threads,
            use_devset=False,
            missing_mode=False,
            program_class="all",
            api_key=args.lm_api_key,
            api_base=args.lm_api_base,
        )
    
    # Generate consolidated CSV results (similar to evaluate_all function)
    try:
        df = read_evaluation_results(file_path)
        df["model"] = args.lm
        csv_path = f"{file_path}/evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Evaluation results saved to CSV: {csv_path}")
    except Exception as e:
        print(f"Warning: Could not generate CSV file: {e}")
    
    print(f"Evaluation complete. Results saved to {file_path}")


if __name__ == "__main__":
    main()
