import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from pathlib import Path

def visualize_benchmark_results(csv_path, output_dir='visualization_results'):
    """
    Create comprehensive visualizations for benchmark results.
    
    Args:
        csv_path: Path to the CSV file containing benchmark results
        output_dir: Directory to save visualizations
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Print basic information
    print(f"Analyzing results from: {os.path.basename(csv_path)}")
    print(f"Total number of benchmark runs: {len(df)}")
    print(f"Unique benchmarks: {df['benchmark'].nunique()}")
    print(f"Unique programs: {df['program'].nunique()}")
    print(f"Unique optimizers: {df['optimizer'].nunique()}")
    
    # Get unique values
    benchmarks = df['benchmark'].unique()
    programs = df['program'].unique()
    optimizers = df['optimizer'].unique()
    
    # Set the style for all plots
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 10})
    
    # 1. Overall optimizer performance across all benchmarks
    plt.figure(figsize=(14, 8))
    optimizer_avg = df.groupby('optimizer')['score'].mean().sort_values(ascending=False)
    ax = sns.barplot(x=optimizer_avg.index, y=optimizer_avg.values)
    plt.title('Average Score by Optimizer (All Benchmarks)', fontsize=16)
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of each bar
    for i, v in enumerate(optimizer_avg.values):
        ax.text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    plt.savefig(output_path / 'overall_optimizer_performance.png', dpi=300)
    plt.close()
    
    # 2. Overall program performance across all benchmarks
    plt.figure(figsize=(12, 8))
    program_avg = df.groupby('program')['score'].mean().sort_values(ascending=False)
    ax = sns.barplot(x=program_avg.index, y=program_avg.values)
    plt.title('Average Score by Program (All Benchmarks)', fontsize=16)
    plt.xlabel('Program', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of each bar
    for i, v in enumerate(program_avg.values):
        ax.text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    plt.savefig(output_path / 'overall_program_performance.png', dpi=300)
    plt.close()
    
    # 3. For each benchmark, create a bar chart showing optimizer performance
    for benchmark in benchmarks:
        benchmark_df = df[df['benchmark'] == benchmark]
        
        plt.figure(figsize=(14, 8))
        optimizer_scores = benchmark_df.groupby('optimizer')['score'].mean().sort_values(ascending=False)
        ax = sns.barplot(x=optimizer_scores.index, y=optimizer_scores.values)
        plt.title(f'Optimizer Performance for {benchmark}', fontsize=16)
        plt.xlabel('Optimizer', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on top of each bar
        for i, v in enumerate(optimizer_scores.values):
            ax.text(i, v + 0.5, f"{v:.2f}", ha='center')
        
        plt.savefig(output_path / f'{benchmark}_optimizer_performance.png', dpi=300)
        plt.close()
    
    # 4. For each program, create a bar chart showing optimizer performance
    for program in programs:
        program_df = df[df['program'] == program]
        
        plt.figure(figsize=(14, 8))
        optimizer_scores = program_df.groupby('optimizer')['score'].mean().sort_values(ascending=False)
        ax = sns.barplot(x=optimizer_scores.index, y=optimizer_scores.values)
        plt.title(f'Optimizer Performance for {program} Program', fontsize=16)
        plt.xlabel('Optimizer', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on top of each bar
        for i, v in enumerate(optimizer_scores.values):
            ax.text(i, v + 0.5, f"{v:.2f}", ha='center')
        
        plt.savefig(output_path / f'{program}_optimizer_performance.png', dpi=300)
        plt.close()
    
    # 5. Heatmap of benchmark vs optimizer performance
    plt.figure(figsize=(16, 10))
    pivot_table = df.pivot_table(index='benchmark', columns='optimizer', values='score', aggfunc='mean')
    
    # Sort benchmarks by baseline performance if available
    if 'Baseline' in pivot_table.columns:
        pivot_table = pivot_table.sort_values('Baseline', ascending=False)
    
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.1f', linewidths=0.5)
    plt.title('Benchmark Performance by Optimizer (Heatmap)', fontsize=16)
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel('Benchmark', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'benchmark_optimizer_heatmap.png', dpi=300)
    plt.close()
    
    # 6. Heatmap of program vs optimizer performance
    plt.figure(figsize=(16, 10))
    pivot_table = df.pivot_table(index='program', columns='optimizer', values='score', aggfunc='mean')
    
    # Sort programs by baseline performance if available
    if 'Baseline' in pivot_table.columns:
        pivot_table = pivot_table.sort_values('Baseline', ascending=False)
    
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.1f', linewidths=0.5)
    plt.title('Program Performance by Optimizer (Heatmap)', fontsize=16)
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel('Program', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'program_optimizer_heatmap.png', dpi=300)
    plt.close()
    
    # 7. Improvement over baseline for each optimizer
    if 'Baseline' in optimizers:
        plt.figure(figsize=(14, 8))
        
        # Calculate average baseline score for each benchmark
        baseline_scores = df[df['optimizer'] == 'Baseline'].groupby('benchmark')['score'].mean()
        
        # Filter out baseline entries
        non_baseline_df = df[df['optimizer'] != 'Baseline'].copy()
        
        # Merge baseline scores
        non_baseline_df = non_baseline_df.merge(
            baseline_scores.reset_index().rename(columns={'score': 'baseline_score'}),
            on='benchmark'
        )
        
        # Calculate improvement
        non_baseline_df['improvement'] = non_baseline_df['score'] - non_baseline_df['baseline_score']
        non_baseline_df['improvement_percent'] = (non_baseline_df['improvement'] / non_baseline_df['baseline_score']) * 100
        
        # Average improvement by optimizer
        improvement_by_optimizer = non_baseline_df.groupby('optimizer')['improvement_percent'].mean().sort_values(ascending=False)
        
        ax = sns.barplot(x=improvement_by_optimizer.index, y=improvement_by_optimizer.values)
        plt.title('Average Percentage Improvement Over Baseline', fontsize=16)
        plt.xlabel('Optimizer', fontsize=12)
        plt.ylabel('Improvement (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on top of each bar
        for i, v in enumerate(improvement_by_optimizer.values):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha='center')
        
        plt.savefig(output_path / 'improvement_over_baseline.png', dpi=300)
        plt.close()
    
    # 8. Token usage analysis
    plt.figure(figsize=(14, 8))
    token_df = df.copy()
    token_df['total_tokens'] = token_df['input_tokens'] + token_df['output_tokens']
    token_df['optimizer_total_tokens'] = token_df['optimizer_input_tokens'] + token_df['optimizer_output_tokens']
    
    # Group by optimizer and calculate average token usage
    token_usage = token_df.groupby('optimizer')[['total_tokens', 'optimizer_total_tokens']].mean().sort_values('total_tokens', ascending=False)
    
    # Create a stacked bar chart
    token_usage.plot(kind='bar', stacked=False, figsize=(14, 8))
    plt.title('Average Token Usage by Optimizer', fontsize=16)
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel('Average Token Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Inference Tokens', 'Optimization Tokens'])
    plt.tight_layout()
    plt.savefig(output_path / 'token_usage_by_optimizer.png', dpi=300)
    plt.close()
    
    # 9. Score vs Token Usage Scatter Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=token_df, 
        x='total_tokens', 
        y='score', 
        hue='optimizer',
        size='optimizer_total_tokens',
        sizes=(20, 200),
        alpha=0.7
    )
    plt.title('Score vs Token Usage', fontsize=16)
    plt.xlabel('Total Tokens (Inference)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path / 'score_vs_tokens_scatter.png', dpi=300)
    plt.close()
    
    # 10. Box plots of score distributions by optimizer
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='optimizer', y='score', data=df)
    plt.title('Score Distribution by Optimizer', fontsize=16)
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'score_distribution_boxplot.png', dpi=300)
    plt.close()
    
    print(f"Visualizations complete! All graphs saved to {output_path}")
    return df

if __name__ == "__main__":
    # Analyze the llama3370b results
    csv_path = "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/llama3370b_0305.csv"
    df = visualize_benchmark_results(csv_path)
