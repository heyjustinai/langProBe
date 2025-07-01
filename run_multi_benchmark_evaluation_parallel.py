#!/usr/bin/env python
"""
Parallel multi-benchmark evaluation script with multiple prompt support.
Runs multiple prompts for each benchmark simultaneously.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

# Configuration for benchmark-prompt mappings with multiple prompts per benchmark
BENCHMARK_CONFIGS = [
    {
        "benchmark": "HeartDisease",
        "prompts": [
            {
                "name": "baseline",
                "file": "meta-optimize-prompt/example_heartdisease_prompt.json",
                "description": "Standard heart disease prompt"
            },
            {
                "name": "cot",
                "file": "meta-optimize-prompt/example_heartdisease_cot_prompt.json",
                "description": "Chain of thought heart disease prompt"
            },
            {
                "name": "react", 
                "file": "meta-optimize-prompt/example_heartdisease_react_prompt.json",
                "description": "ReAct reasoning heart disease prompt"
            }
        ],
        "description": "Heart Disease prediction benchmark"
    },
    {
        "benchmark": "hover",
        "prompts": [
            {
                "name": "baseline",
                "file": "meta-optimize-prompt/example_hover_prompt.json",
                "description": "Standard HoVer prompt"
            },
            {
                "name": "cot",
                "file": "meta-optimize-prompt/example_hover_cot_prompt.json", 
                "description": "Chain of thought HoVer prompt"
            }
        ],
        "description": "HoVer fact verification benchmark"
    },
    {
        "benchmark": "judgebench",
        "prompts": [
            {
                "name": "baseline",
                "file": "meta-optimize-prompt/example_judgebench_prompt.json",
                "description": "Standard judge evaluation prompt"
            },
            {
                "name": "detailed",
                "file": "meta-optimize-prompt/example_judgebench_detailed_prompt.json",
                "description": "Detailed judge evaluation prompt"
            }
        ],
        "description": "Judge evaluation benchmark"
    }
]

def flatten_benchmark_prompt_configs(benchmark_configs):
    """
    Flatten the nested benchmark-prompt structure into individual configurations.
    Each configuration represents one benchmark-prompt combination.
    """
    flattened_configs = []
    
    for benchmark_config in benchmark_configs:
        benchmark = benchmark_config["benchmark"]
        benchmark_desc = benchmark_config["description"]
        
        for prompt_config in benchmark_config["prompts"]:
            flattened_config = {
                "benchmark": benchmark,
                "benchmark_description": benchmark_desc,
                "prompt_name": prompt_config["name"],
                "prompt_file": prompt_config["file"],
                "prompt_description": prompt_config["description"],
                "run_id": f"{benchmark}_{prompt_config['name']}"
            }
            flattened_configs.append(flattened_config)
    
    return flattened_configs

def create_output_directory(base_path, benchmark):
    """Create and return the output directory path for a benchmark (shared by all prompts)"""
    output_dir = Path(base_path) / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)

def consolidate_benchmark_results(output_base_path, benchmarks_with_results):
    """
    Consolidate individual prompt CSV files into a single benchmark CSV file.
    Filters out cached/duplicate results to only include fresh results from each prompt run.
    """
    import pandas as pd
    import glob
    
    print(f"\n🔄 Consolidating results for {len(benchmarks_with_results)} benchmarks...")
    
    for benchmark in benchmarks_with_results:
        benchmark_dir = Path(output_base_path) / benchmark
        
        if not benchmark_dir.exists():
            continue
            
        print(f"  📊 Consolidating {benchmark}...")
        
        # Find all individual result CSV files for this benchmark
        csv_files = list(benchmark_dir.glob("evaluation_results_*.csv"))
        
        if not csv_files:
            print(f"    ⚠️  No individual CSV files found for {benchmark}")
            continue
        
        # Read and filter CSV files to avoid cached/duplicate results
        consolidated_rows = []
        seen_combinations = set()
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                prompt_name = csv_file.stem.replace("evaluation_results_", "")
                
                print(f"    📝 Processing {csv_file.name} (prompt: {prompt_name})")
                
                # Filter to only include results that match this specific prompt
                filtered_df = df[df['prompt_variation'] == prompt_name].copy()
                
                # Remove any rows that don't have a custom prompt in the filename
                # (these are likely cached baseline results)
                if prompt_name != 'default':
                    # Only include rows with CustomPrompt in the optimizer or filename
                    custom_mask = (
                        filtered_df['optimizer'].str.contains('CustomPrompt', na=False) |
                        filtered_df['file_name'].str.contains('CustomPrompt', na=False)
                    )
                    
                    # For baseline comparison, also include genuine baseline rows
                    baseline_mask = (
                        (filtered_df['optimizer'] == 'Baseline') & 
                        (~filtered_df['file_name'].str.contains('CustomPrompt', na=False))
                    )
                    
                    # Keep either custom prompt results or genuine baseline
                    filtered_df = filtered_df[custom_mask | baseline_mask]
                
                # Remove duplicates based on file_name + optimizer + prompt_variation
                for _, row in filtered_df.iterrows():
                    combination_key = (row['file_name'], row['optimizer'], row['prompt_variation'])
                    
                    if combination_key not in seen_combinations:
                        seen_combinations.add(combination_key)
                        consolidated_rows.append(row)
                        
                print(f"      ✅ Added {len(filtered_df)} unique results")
                
            except Exception as e:
                print(f"    ❌ Error reading {csv_file.name}: {e}")
        
        if consolidated_rows:
            # Create consolidated dataframe
            consolidated_df = pd.DataFrame(consolidated_rows)
            
            # Sort by prompt_variation for better readability
            consolidated_df = consolidated_df.sort_values(['prompt_variation', 'file_name'])
            
            # Save consolidated results
            consolidated_path = benchmark_dir / "evaluation_results.csv"
            consolidated_df.to_csv(consolidated_path, index=False)
            
            print(f"    ✅ Consolidated into: evaluation_results.csv")
            print(f"    📈 Total unique rows: {len(consolidated_df)}")
            print(f"    🏷️  Prompt variations: {sorted(consolidated_df['prompt_variation'].unique())}")
        else:
            print(f"    ⚠️  No valid results found after filtering")
        
        # Check for JSON configuration files
        json_files = list(benchmark_dir.glob("*_Predict_*.json"))
        if json_files:
            print(f"    📄 Found {len(json_files)} JSON configuration files")
        else:
            print(f"    ⚠️  No JSON configuration files found")
    
    print("✅ Consolidation complete!")

def clean_and_reconsolidate_results(output_base_path="evaluation_llama3370b"):
    """
    Clean up existing consolidated files and re-consolidate to remove cached duplicates.
    """
    print(f"🧹 Cleaning and re-consolidating results in {output_base_path}...")
    
    base_path = Path(output_base_path)
    if not base_path.exists():
        print(f"❌ Directory {output_base_path} does not exist")
        return
    
    # Find all benchmark directories
    benchmark_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    benchmark_names = [d.name for d in benchmark_dirs]
    
    if not benchmark_names:
        print("⚠️  No benchmark directories found")
        return
        
    print(f"📁 Found benchmarks: {benchmark_names}")
    
    # Remove existing consolidated files
    for benchmark_dir in benchmark_dirs:
        consolidated_file = benchmark_dir / "evaluation_results.csv"
        if consolidated_file.exists():
            consolidated_file.unlink()
            print(f"  🗑️  Removed old consolidated file: {benchmark_dir.name}/evaluation_results.csv")
    
    # Re-consolidate with improved logic
    consolidate_benchmark_results(output_base_path, benchmark_names)

def clean_all_results(output_base_path="evaluation_llama3370b"):
    """
    Completely clean all result files to ensure fresh evaluation runs.
    Use this when cached results are contaminating the outputs.
    """
    print(f"🧹🧹 DEEP CLEANING all results in {output_base_path}...")
    print("⚠️  This will remove ALL evaluation results and require re-running evaluations!")
    
    base_path = Path(output_base_path)
    if not base_path.exists():
        print(f"❌ Directory {output_base_path} does not exist")
        return
    
    # Find all benchmark directories
    benchmark_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for benchmark_dir in benchmark_dirs:
        print(f"  🗑️  Cleaning {benchmark_dir.name}...")
        
        # Remove all CSV files
        csv_files = list(benchmark_dir.glob("*.csv"))
        for csv_file in csv_files:
            csv_file.unlink()
            print(f"    - Removed: {csv_file.name}")
        
        # Remove all TXT result files  
        txt_files = list(benchmark_dir.glob("*.txt"))
        for txt_file in txt_files:
            txt_file.unlink()
            print(f"    - Removed: {txt_file.name}")
            
        # Remove JSON config files (they'll be regenerated)
        json_files = list(benchmark_dir.glob("*_Predict_*.json"))
        for json_file in json_files:
            json_file.unlink()
            print(f"    - Removed: {json_file.name}")
            
        # Clean cache directories
        cache_dirs = list(benchmark_dir.glob(".dspy_cache*"))
        for cache_dir in cache_dirs:
            if cache_dir.is_dir():
                import shutil
                shutil.rmtree(cache_dir)
                print(f"    - Removed cache: {cache_dir.name}")
    
    print("✅ Deep cleaning complete! Ready for fresh evaluation runs.")

def main():
    """Main execution function for parallel evaluation with multiple prompts"""
    
    # Flatten the configurations
    all_configs = flatten_benchmark_prompt_configs(BENCHMARK_CONFIGS)
    
    # Base arguments that are common to all evaluations
    base_args = [
        "--dataset_mode=test",
        "--lm=openrouter/meta-llama/llama-3.3-70b-instruct",
        "--lm_api_base=https://openrouter.ai/api/v1",
        f"--lm_api_key={os.getenv('OPENROUTER_API_KEY')}"
    ]
    
    print("🎯 Multi-Benchmark Multi-Prompt PARALLEL Evaluation")
    print(f"🤖 Model: openrouter/meta-llama/llama-3.3-70b-instruct")
    print(f"📊 Dataset mode: test")
    print(f"🔢 Total benchmark-prompt combinations: {len(all_configs)}")
    print("⚡ Execution mode: PARALLEL")
    
    # Print configuration summary
    print(f"\n📋 Configuration Summary:")
    benchmark_counts = {}
    for config in all_configs:
        benchmark = config["benchmark"]
        benchmark_counts[benchmark] = benchmark_counts.get(benchmark, 0) + 1
    
    for benchmark, count in benchmark_counts.items():
        print(f"  📊 {benchmark}: {count} prompts")
    
    # Set up environment
    env = os.environ.copy()
    
    processes = []
    start_time = time.time()
    
    # Start all processes in parallel
    print(f"\n🚀 Starting all {len(all_configs)} evaluations in parallel...")
    
    for config in all_configs:
        benchmark = config["benchmark"]
        prompt_name = config["prompt_name"]
        prompt_file = config["prompt_file"]
        run_id = config["run_id"]
        prompt_description = config["prompt_description"]
        
        # Check if prompt file exists
        if not Path(prompt_file).exists():
            print(f"❌ Error: Prompt file {prompt_file} not found for {run_id}!")
            continue
        
        # Create shared output directory for this benchmark
        output_path = create_output_directory("evaluation_llama3370b", benchmark)
        
        # Set cache directory specific to this prompt within the benchmark
        run_env = env.copy()
        run_env["DSPY_CACHEDIR"] = f"{output_path}/.dspy_cache_{prompt_name}"
        
        # Build the command with benchmark output path and prompt-specific naming
        cmd = [
            "python", "evaluate_custom_prompt.py",
            f"--prompt_file={prompt_file}",
            f"--benchmarks={benchmark}",
            f"--file_path={output_path}",
            f"--output_suffix={prompt_name}",
            *base_args
        ]
        
        print(f"🔧 Starting [{run_id}]: {prompt_description}")
        print(f"   📁 Output: {output_path}")
        print(f"   📝 Prompt: {prompt_file}")
        
        try:
            # Start the process
            process = subprocess.Popen(
                cmd, 
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            processes.append({
                'process': process,
                'benchmark': benchmark,
                'prompt_name': prompt_name,
                'run_id': run_id,
                'description': prompt_description,
                'output_path': output_path,
                'start_time': time.time()
            })
            
        except Exception as e:
            print(f"❌ Failed to start {run_id}: {e}")
    
    print(f"\n👥 Started {len(processes)} parallel evaluations")
    print("⏳ Waiting for all evaluations to complete...")
    print("📊 Progress updates will appear as each evaluation finishes")
    
    # Wait for all processes to complete
    results = {}
    successful_runs = 0
    completed = 0
    
    while processes:
        for i, proc_info in enumerate(processes[:]):
            process = proc_info['process']
            run_id = proc_info['run_id']
            benchmark = proc_info['benchmark']
            prompt_name = proc_info['prompt_name']
            description = proc_info['description']
            output_path = proc_info['output_path']
            
            # Check if process has finished
            if process.poll() is not None:
                completed += 1
                duration = time.time() - proc_info['start_time']
                return_code = process.returncode
                
                if return_code == 0:
                    print(f"✅ [{prompt_name}] Completed ({completed}/{len(all_configs)}): {benchmark} - SUCCESS ({duration:.1f}s)")
                    print(f"   📁 Results saved to: {output_path}")
                    results[run_id] = {
                        'success': True,
                        'benchmark': benchmark,
                        'prompt_name': prompt_name,
                        'output_path': output_path,
                        'duration': duration
                    }
                    successful_runs += 1
                else:
                    print(f"❌ [{prompt_name}] Completed ({completed}/{len(all_configs)}): {benchmark} - FAILED ({duration:.1f}s, exit code: {return_code})")
                    results[run_id] = {
                        'success': False,
                        'benchmark': benchmark,
                        'prompt_name': prompt_name,
                        'output_path': output_path,
                        'duration': duration
                    }
                    
                    # Print some error output
                    try:
                        stdout, stderr = process.communicate(timeout=1)
                        if stdout:
                            last_lines = stdout.strip().split('\n')[-3:]  # Last 3 lines
                            print(f"   💬 Last output: {' | '.join(last_lines)}")
                    except:
                        pass
                
                # Remove completed process from list
                processes.pop(i)
                break
        
        # Small delay to avoid busy waiting
        time.sleep(2)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Consolidate results for each benchmark
    successful_benchmarks = set()
    for run_id, result in results.items():
        if result['success']:
            successful_benchmarks.add(result['benchmark'])
    
    if successful_benchmarks:
        consolidate_benchmark_results("evaluation_llama3370b", successful_benchmarks)
    
    # Summary report
    print(f"\n{'='*80}")
    print("📊 PARALLEL MULTI-PROMPT EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful_runs}/{len(all_configs)}")
    print(f"❌ Failed: {len(all_configs) - successful_runs}/{len(all_configs)}")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print()
    
    # Group results by benchmark for better readability
    by_benchmark = {}
    for run_id, result in results.items():
        benchmark = result['benchmark']
        if benchmark not in by_benchmark:
            by_benchmark[benchmark] = []
        by_benchmark[benchmark].append(result)
    
    for benchmark, benchmark_results in by_benchmark.items():
        print(f"📊 {benchmark}:")
        for result in benchmark_results:
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            prompt_name = result['prompt_name']
            duration = result['duration']
            print(f"  [{prompt_name:<10}] : {status} ({duration:.1f}s)")
            if result['success']:
                print(f"      📁 {result['output_path']}")
        print()
    
    if successful_runs == len(all_configs):
        print(f"🎉 All evaluations completed successfully!")
        print(f"📁 Check results organized by benchmark in: evaluation_llama3370b/")
        print(f"📊 Each benchmark has consolidated and individual prompt results")
        print(f"🚀 Parallel execution completed in {total_time/60:.1f} minutes!")
    else:
        print(f"⚠️  Some evaluations failed. Check the logs above for details.")
        print(f"✅ {successful_runs} evaluations succeeded and have results available.")
        # Don't exit with error if some succeeded
        
    # Print directory structure for easy navigation
    print(f"\n📂 Output Directory Structure:")
    print(f"evaluation_llama3370b/")
    for benchmark, benchmark_results in by_benchmark.items():
        successful_prompts = [r for r in benchmark_results if r['success']]
        if successful_prompts:
            print(f"├── {benchmark}/")
            print(f"│   ├── evaluation_results.csv              # 📊 Consolidated all prompts")
            
            # Show individual prompt files
            for i, result in enumerate(successful_prompts):
                prompt_name = result['prompt_name']
                print(f"│   ├── evaluation_results_{prompt_name}.csv        # 📝 Individual results")
            
            # Show JSON config files
            for i, result in enumerate(successful_prompts):
                prompt_name = result['prompt_name']
                print(f"│   ├── {benchmark}_Predict_*_{prompt_name}.json    # ⚙️  Prompt config")
            
            # Show cache directories
            print(f"│   └── .dspy_cache_{{prompt_name}}/              # 💾 Cache per prompt")
    
    print(f"\n🎯 Analysis Tips:")
    print(f"• Use evaluation_results.csv for comparing all prompt strategies")
    print(f"• Individual files help analyze specific prompt performance")
    print(f"• JSON files contain the exact prompt configurations used")

if __name__ == "__main__":
    import sys
    
    # Check for cleanup arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clean":
            print("🧹 Running cleanup and re-consolidation...")
            clean_and_reconsolidate_results()
        elif sys.argv[1] == "--deep-clean":
            print("🧹🧹 Running deep clean of all results...")
            clean_all_results()
        else:
            print("❌ Unknown argument. Use --clean or --deep-clean")
    else:
        main() 