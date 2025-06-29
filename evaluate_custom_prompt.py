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
    
    parser.add_argument(
        "--output_suffix",
        help="Suffix to add to output filenames (e.g., for different prompt variations)",
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
    
    # Clear any existing result files to avoid cached contamination
    if args.output_suffix:
        print(f"🧹 Clearing cache for prompt variation: {args.output_suffix}")
        # Clear DSPy history and cache to avoid contamination
        if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
            dspy.settings.lm.history = []
    
    # Run the evaluation directly without going through evaluate_all
    # This avoids the double registration issue
    for benchmark_meta in benchmarks:
        print(f"🚀 Running evaluation for {benchmark_meta.name} with prompt: {args.output_suffix or 'default'}")
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
    
    # Generate CSV results with unique naming and strict filtering to avoid conflicts
    try:
        df = read_evaluation_results(file_path)
        df["model"] = args.lm
        
        # Filter results to only include those relevant to this prompt run
        if args.output_suffix:
            csv_filename = f"evaluation_results_{args.output_suffix}.csv"
            df["prompt_variation"] = args.output_suffix
            
            # Extract the prompt filename from args.prompt_file for filtering
            prompt_filename = Path(args.prompt_file).name
            
            # Filter to only include results that match this specific prompt
            # Include results with the custom prompt filename OR baseline results without custom prompts
            custom_mask = df['file_name'].str.contains(prompt_filename.replace('.json', ''), na=False)
            baseline_mask = (
                (df['optimizer'] == 'Baseline') & 
                (~df['file_name'].str.contains('CustomPrompt', na=False))
            )
            
            # Keep either results from this specific prompt or genuine baseline results
            filtered_df = df[custom_mask | baseline_mask].copy()
            
            print(f"📊 Filtered results: {len(filtered_df)} rows from {len(df)} total (prompt: {args.output_suffix})")
            
        else:
            csv_filename = "evaluation_results.csv"
            df["prompt_variation"] = "default"
            filtered_df = df
            
        csv_path = f"{file_path}/{csv_filename}"
        filtered_df.to_csv(csv_path, index=False)
        print(f"Evaluation results saved to CSV: {csv_path}")
        
        # Also update any JSON configuration files to include the suffix
        if args.output_suffix:
            # Look for generated JSON files and rename them to include the suffix
            import glob
            json_files = glob.glob(f"{file_path}/*_Predict_*.json")
            for json_file in json_files:
                if "CustomPrompt" in json_file and args.output_suffix not in json_file:
                    # Create new filename with suffix
                    base_name = Path(json_file).stem
                    new_name = f"{base_name}_{args.output_suffix}.json"
                    new_path = Path(json_file).parent / new_name
                    
                    # Copy the file with the new name
                    import shutil
                    shutil.copy2(json_file, new_path)
                    print(f"JSON config saved with suffix: {new_path}")
        
    except Exception as e:
        print(f"Warning: Could not generate CSV file: {e}")
    
    print(f"Evaluation complete. Results saved to {file_path}")


if __name__ == "__main__":
    main()
