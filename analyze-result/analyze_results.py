import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from pathlib import Path

def analyze_benchmark_results(csv_path):
    """
    Analyze benchmark results from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing benchmark results
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Print basic information
    print(f"Analyzing results from: {os.path.basename(csv_path)}")
    print(f"Total number of benchmark runs: {len(df)}")
    print(f"Unique benchmarks: {df['benchmark'].nunique()}")
    print(f"Unique programs: {df['program'].nunique()}")
    print(f"Unique optimizers: {df['optimizer'].nunique()}")
    
    # Calculate performance metrics
    print("\n=== Performance Summary ===")
    print(f"Average score across all benchmarks: {df['score'].mean():.2f}")
    print(f"Median score: {df['score'].median():.2f}")
    print(f"Min score: {df['score'].min():.2f}")
    print(f"Max score: {df['score'].max():.2f}")
    
    # Group by benchmark and optimizer to find the best optimizer for each benchmark
    best_by_benchmark = df.groupby('benchmark').apply(
        lambda x: x.loc[x['score'].idxmax()]
    )[['optimizer', 'score', 'program']]
    
    print("\n=== Best Optimizer per Benchmark ===")
    print(best_by_benchmark[['optimizer', 'score', 'program']])
    
    # Compare baseline vs optimized performance
    baseline_df = df[df['optimizer'] == 'Baseline']
    optimized_df = df[df['optimizer'] != 'Baseline']
    
    if not baseline_df.empty and not optimized_df.empty:
        baseline_avg = baseline_df['score'].mean()
        optimized_avg = optimized_df['score'].mean()
        improvement = optimized_avg - baseline_avg
        
        print(f"\n=== Baseline vs Optimized ===")
        print(f"Baseline average score: {baseline_avg:.2f}")
        print(f"Optimized average score: {optimized_avg:.2f}")
        print(f"Average improvement: {improvement:.2f} points ({improvement/baseline_avg*100:.2f}%)")
    
    # Create output directory for plots
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    
    # 1. Score distribution by optimizer
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='optimizer', y='score', data=df)
    plt.xticks(rotation=90)
    plt.title('Score Distribution by Optimizer')
    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_score_distribution.png')
    
    # 2. Benchmark performance comparison
    plt.figure(figsize=(14, 10))
    benchmark_pivot = df.pivot_table(
        index='benchmark', 
        columns='optimizer', 
        values='score', 
        aggfunc='mean'
    )
    
    # Sort benchmarks by baseline performance if available
    if 'Baseline' in benchmark_pivot.columns:
        benchmark_pivot = benchmark_pivot.sort_values('Baseline')
    
    sns.heatmap(benchmark_pivot, annot=True, cmap='viridis', fmt='.1f')
    plt.title('Benchmark Performance by Optimizer')
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_optimizer_heatmap.png')
    
    # 3. Top optimizers by average performance
    plt.figure(figsize=(10, 6))
    optimizer_avg = df.groupby('optimizer')['score'].mean().sort_values(ascending=False)
    sns.barplot(x=optimizer_avg.index, y=optimizer_avg.values)
    plt.xticks(rotation=90)
    plt.title('Average Score by Optimizer')
    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_average_score.png')
    
    # 4. Program comparison
    plt.figure(figsize=(12, 8))
    program_pivot = df.pivot_table(
        index='program', 
        columns='optimizer', 
        values='score', 
        aggfunc='mean'
    )
    sns.heatmap(program_pivot, annot=True, cmap='viridis', fmt='.1f')
    plt.title('Program Performance by Optimizer')
    plt.tight_layout()
    plt.savefig(output_dir / 'program_optimizer_heatmap.png')
    
    # 5. Token usage analysis
    plt.figure(figsize=(12, 8))
    token_df = df.copy()
    token_df['total_tokens'] = token_df['input_tokens'] + token_df['output_tokens']
    token_df['optimizer_total_tokens'] = token_df['optimizer_input_tokens'] + token_df['optimizer_output_tokens']
    
    # Create a scatter plot of score vs token usage
    sns.scatterplot(
        x='total_tokens', 
        y='score', 
        hue='optimizer',
        size='optimizer_total_tokens',
        sizes=(20, 200),
        alpha=0.7,
        data=token_df
    )
    plt.title('Score vs Token Usage')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'score_vs_tokens.png')
    
    print(f"\nAnalysis complete! Visualizations saved to {output_dir}")
    
    return df, best_by_benchmark

def compare_models(csv_paths, model_names=None):
    """
    Compare results across different model files.
    
    Args:
        csv_paths: List of paths to CSV files
        model_names: Optional list of model names (defaults to filenames)
    """
    if model_names is None:
        model_names = [os.path.basename(path).split('.')[0] for path in csv_paths]
    
    all_dfs = []
    for path, name in zip(csv_paths, model_names):
        df = pd.read_csv(path)
        df['model'] = name
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs)
    
    # Create output directory for plots
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # Model comparison visualizations
    
    # 1. Overall model performance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='score', data=combined_df)
    plt.title('Average Score by Model')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png')
    
    # 2. Model performance by benchmark
    plt.figure(figsize=(14, 10))
    benchmark_model_pivot = combined_df.pivot_table(
        index='benchmark', 
        columns='model', 
        values='score', 
        aggfunc='mean'
    )
    sns.heatmap(benchmark_model_pivot, annot=True, cmap='viridis', fmt='.1f')
    plt.title('Benchmark Performance by Model')
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_model_heatmap.png')
    
    # 3. Model performance by optimizer
    plt.figure(figsize=(14, 10))
    optimizer_model_pivot = combined_df.pivot_table(
        index='optimizer', 
        columns='model', 
        values='score', 
        aggfunc='mean'
    )
    sns.heatmap(optimizer_model_pivot, annot=True, cmap='viridis', fmt='.1f')
    plt.title('Optimizer Performance by Model')
    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_model_heatmap.png')
    
    print(f"\nModel comparison complete! Visualizations saved to {output_dir}")
    
    return combined_df

if __name__ == "__main__":
    # Analyze a single model's results
    csv_path = "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/llama3370b_0305.csv"
    df, best_optimizers = analyze_benchmark_results(csv_path)
    
    # Uncomment to compare multiple models
    """
    csv_paths = [
        "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/llama3370b_0305.csv",
        "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/gpt4o_0305.csv",
        "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/gpt4omini_0305.csv"
    ]
    model_names = ["Llama-3-70B", "GPT-4o", "GPT-4o-mini"]
    combined_df = compare_models(csv_paths, model_names)
    """
