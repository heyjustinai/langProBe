import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from pathlib import Path

def compare_model_results(csv_paths, model_names=None, output_dir='model_comparison_results'):
    """
    Compare benchmark results across different models.
    
    Args:
        csv_paths: List of paths to CSV files containing benchmark results
        model_names: Optional list of model names (defaults to filenames)
        output_dir: Directory to save visualizations
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    if model_names is None:
        model_names = [os.path.basename(path).split('.')[0] for path in csv_paths]
    
    # Load data from all files
    all_dfs = []
    for path, name in zip(csv_paths, model_names):
        df = pd.read_csv(path)
        df['model'] = name
        all_dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs)
    
    # Print basic information
    print(f"Comparing {len(model_names)} models: {', '.join(model_names)}")
    print(f"Total number of benchmark runs: {len(combined_df)}")
    print(f"Unique benchmarks: {combined_df['benchmark'].nunique()}")
    print(f"Unique programs: {combined_df['program'].nunique()}")
    print(f"Unique optimizers: {combined_df['optimizer'].nunique()}")
    
    # Set the style for all plots
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 10})
    
    # 1. Overall model performance comparison
    plt.figure(figsize=(12, 8))
    model_avg = combined_df.groupby('model')['score'].mean().sort_values(ascending=False)
    ax = sns.barplot(x=model_avg.index, y=model_avg.values)
    plt.title('Average Score by Model (All Benchmarks)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of each bar
    for i, v in enumerate(model_avg.values):
        ax.text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    plt.savefig(output_path / 'overall_model_performance.png', dpi=300)
    plt.close()
    
    # 2. Model performance by benchmark
    plt.figure(figsize=(16, 10))
    benchmark_model_pivot = combined_df.pivot_table(
        index='benchmark', 
        columns='model', 
        values='score', 
        aggfunc='mean'
    )
    
    # Sort benchmarks by average performance
    benchmark_model_pivot['avg'] = benchmark_model_pivot.mean(axis=1)
    benchmark_model_pivot = benchmark_model_pivot.sort_values('avg', ascending=False)
    benchmark_model_pivot = benchmark_model_pivot.drop('avg', axis=1)
    
    sns.heatmap(benchmark_model_pivot, annot=True, cmap='viridis', fmt='.1f', linewidths=0.5)
    plt.title('Benchmark Performance by Model (Heatmap)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Benchmark', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'benchmark_model_heatmap.png', dpi=300)
    plt.close()
    
    # 3. Model performance by optimizer
    plt.figure(figsize=(16, 10))
    optimizer_model_pivot = combined_df.pivot_table(
        index='optimizer', 
        columns='model', 
        values='score', 
        aggfunc='mean'
    )
    
    # Sort optimizers by average performance
    optimizer_model_pivot['avg'] = optimizer_model_pivot.mean(axis=1)
    optimizer_model_pivot = optimizer_model_pivot.sort_values('avg', ascending=False)
    optimizer_model_pivot = optimizer_model_pivot.drop('avg', axis=1)
    
    sns.heatmap(optimizer_model_pivot, annot=True, cmap='viridis', fmt='.1f', linewidths=0.5)
    plt.title('Optimizer Performance by Model (Heatmap)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Optimizer', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'optimizer_model_heatmap.png', dpi=300)
    plt.close()
    
    # 4. Model performance by program
    plt.figure(figsize=(16, 10))
    program_model_pivot = combined_df.pivot_table(
        index='program', 
        columns='model', 
        values='score', 
        aggfunc='mean'
    )
    
    # Sort programs by average performance
    program_model_pivot['avg'] = program_model_pivot.mean(axis=1)
    program_model_pivot = program_model_pivot.sort_values('avg', ascending=False)
    program_model_pivot = program_model_pivot.drop('avg', axis=1)
    
    sns.heatmap(program_model_pivot, annot=True, cmap='viridis', fmt='.1f', linewidths=0.5)
    plt.title('Program Performance by Model (Heatmap)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Program', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'program_model_heatmap.png', dpi=300)
    plt.close()
    
    # 5. Baseline performance comparison across models
    baseline_df = combined_df[combined_df['optimizer'] == 'Baseline']
    
    plt.figure(figsize=(14, 8))
    baseline_pivot = baseline_df.pivot_table(
        index='benchmark', 
        columns='model', 
        values='score', 
        aggfunc='mean'
    )
    
    # Sort benchmarks by average baseline performance
    baseline_pivot['avg'] = baseline_pivot.mean(axis=1)
    baseline_pivot = baseline_pivot.sort_values('avg', ascending=False)
    baseline_pivot = baseline_pivot.drop('avg', axis=1)
    
    sns.heatmap(baseline_pivot, annot=True, cmap='viridis', fmt='.1f', linewidths=0.5)
    plt.title('Baseline Performance by Model (Heatmap)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Benchmark', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'baseline_model_heatmap.png', dpi=300)
    plt.close()
    
    # 6. Best optimizer for each model
    best_optimizers = {}
    for model in model_names:
        model_df = combined_df[combined_df['model'] == model]
        best_optimizer = model_df.groupby('optimizer')['score'].mean().sort_values(ascending=False).index[0]
        best_optimizers[model] = best_optimizer
    
    # Create a bar chart for each benchmark showing model performance
    benchmarks = combined_df['benchmark'].unique()
    for benchmark in benchmarks:
        benchmark_df = combined_df[combined_df['benchmark'] == benchmark]
        
        plt.figure(figsize=(12, 8))
        model_scores = benchmark_df.groupby('model')['score'].mean().sort_values(ascending=False)
        ax = sns.barplot(x=model_scores.index, y=model_scores.values)
        plt.title(f'Model Performance for {benchmark}', fontsize=16)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on top of each bar
        for i, v in enumerate(model_scores.values):
            ax.text(i, v + 0.5, f"{v:.2f}", ha='center')
        
        plt.savefig(output_path / f'{benchmark}_model_performance.png', dpi=300)
        plt.close()
    
    # 7. Improvement over baseline for each model
    if 'Baseline' in combined_df['optimizer'].unique():
        plt.figure(figsize=(14, 8))
        
        # Calculate average baseline score for each benchmark and model
        baseline_scores = combined_df[combined_df['optimizer'] == 'Baseline'].groupby(['benchmark', 'model'])['score'].mean()
        
        # Filter out baseline entries
        non_baseline_df = combined_df[combined_df['optimizer'] != 'Baseline'].copy()
        
        # Merge baseline scores
        non_baseline_df = non_baseline_df.merge(
            baseline_scores.reset_index().rename(columns={'score': 'baseline_score'}),
            on=['benchmark', 'model']
        )
        
        # Calculate improvement
        non_baseline_df['improvement'] = non_baseline_df['score'] - non_baseline_df['baseline_score']
        non_baseline_df['improvement_percent'] = (non_baseline_df['improvement'] / non_baseline_df['baseline_score']) * 100
        
        # Average improvement by model
        improvement_by_model = non_baseline_df.groupby('model')['improvement_percent'].mean().sort_values(ascending=False)
        
        ax = sns.barplot(x=improvement_by_model.index, y=improvement_by_model.values)
        plt.title('Average Percentage Improvement Over Baseline by Model', fontsize=16)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Improvement (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on top of each bar
        for i, v in enumerate(improvement_by_model.values):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha='center')
        
        plt.savefig(output_path / 'improvement_over_baseline_by_model.png', dpi=300)
        plt.close()
    
    # 8. Token usage comparison across models
    plt.figure(figsize=(14, 8))
    token_df = combined_df.copy()
    token_df['total_tokens'] = token_df['input_tokens'] + token_df['output_tokens']
    token_df['optimizer_total_tokens'] = token_df['optimizer_input_tokens'] + token_df['optimizer_output_tokens']
    
    # Group by model and calculate average token usage
    token_usage = token_df.groupby('model')[['total_tokens', 'optimizer_total_tokens']].mean()
    
    # Create a bar chart
    token_usage.plot(kind='bar', stacked=False, figsize=(14, 8))
    plt.title('Average Token Usage by Model', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Token Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Inference Tokens', 'Optimization Tokens'])
    plt.tight_layout()
    plt.savefig(output_path / 'token_usage_by_model.png', dpi=300)
    plt.close()
    
    # 9. Score distribution by model (boxplot)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='model', y='score', data=combined_df)
    plt.title('Score Distribution by Model', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'score_distribution_by_model.png', dpi=300)
    plt.close()
    
    # 10. Best model-optimizer combinations
    plt.figure(figsize=(16, 10))
    model_optimizer_pivot = combined_df.pivot_table(
        index='optimizer', 
        columns='model', 
        values='score', 
        aggfunc='mean'
    )
    
    # Find the best model-optimizer combination
    best_combinations = []
    for optimizer in model_optimizer_pivot.index:
        for model in model_optimizer_pivot.columns:
            score = model_optimizer_pivot.loc[optimizer, model]
            best_combinations.append((model, optimizer, score))
    
    # Sort by score and get top 10
    best_combinations.sort(key=lambda x: x[2], reverse=True)
    top_combinations = best_combinations[:10]
    
    # Create a bar chart for top combinations
    plt.figure(figsize=(14, 8))
    models = [combo[0] for combo in top_combinations]
    optimizers = [combo[1] for combo in top_combinations]
    scores = [combo[2] for combo in top_combinations]
    
    # Create labels for x-axis
    labels = [f"{model}\n{optimizer}" for model, optimizer in zip(models, optimizers)]
    
    ax = sns.barplot(x=range(len(labels)), y=scores)
    plt.title('Top 10 Model-Optimizer Combinations', fontsize=16)
    plt.xlabel('Model-Optimizer Combination', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of each bar
    for i, v in enumerate(scores):
        ax.text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    plt.savefig(output_path / 'top_model_optimizer_combinations.png', dpi=300)
    plt.close()
    
    print(f"Model comparison complete! All graphs saved to {output_path}")
    return combined_df

if __name__ == "__main__":
    # Compare multiple models
    csv_paths = [
        "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/llama3370b_0305.csv",
        "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/gpt4o_0305.csv",
        "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/gpt4omini_0305.csv"
    ]
    model_names = ["Llama-3-70B", "GPT-4o", "GPT-4o-mini"]
    combined_df = compare_model_results(csv_paths, model_names)
