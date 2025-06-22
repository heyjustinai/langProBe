import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def create_miprov2_teacher_heatmap(csv_paths, model_names=None, output_dir='miprov2_teacher_analysis'):
    """
    Create a benchmark model heatmap specifically for MIPROv2-teacher-gpt4o optimizer.
    
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
    
    # Filter for MIPROv2-teacher-gpt4o optimizer
    miprov2_teacher_df = combined_df[combined_df['optimizer'] == 'MIPROv2-teacher-gpt4o']
    
    # Print basic information
    print(f"Analyzing MIPROv2-teacher-gpt4o optimizer across {len(model_names)} models")
    print(f"Models: {', '.join(model_names)}")
    print(f"Total number of benchmark runs with MIPROv2-teacher-gpt4o: {len(miprov2_teacher_df)}")
    print(f"Unique benchmarks: {miprov2_teacher_df['benchmark'].nunique()}")
    
    # Set the style for all plots
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 10})
    
    # Create benchmark model heatmap for MIPROv2-teacher-gpt4o
    plt.figure(figsize=(16, 12))
    
    # Create pivot table for benchmark vs model with MIPROv2-teacher-gpt4o
    benchmark_model_pivot = miprov2_teacher_df.pivot_table(
        index='benchmark', 
        columns='model', 
        values='score', 
        aggfunc='mean'
    )
    
    # Sort benchmarks by average performance
    benchmark_model_pivot['avg'] = benchmark_model_pivot.mean(axis=1)
    benchmark_model_pivot = benchmark_model_pivot.sort_values('avg', ascending=False)
    benchmark_model_pivot = benchmark_model_pivot.drop('avg', axis=1)
    
    # Create heatmap
    ax = sns.heatmap(benchmark_model_pivot, annot=True, cmap='viridis', fmt='.1f', linewidths=0.5)
    plt.title('Benchmark Performance by Model with MIPROv2-teacher-gpt4o Optimizer', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Benchmark', fontsize=12)
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_path = output_path / 'miprov2_teacher_benchmark_model_heatmap.png'
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    # Also create a comparison with baseline
    if 'Baseline' in combined_df['optimizer'].unique():
        baseline_df = combined_df[combined_df['optimizer'] == 'Baseline']
        
        # Create a DataFrame to store improvement percentages
        improvement_data = []
        
        for model in model_names:
            for benchmark in miprov2_teacher_df['benchmark'].unique():
                # Get baseline score for this model and benchmark
                baseline_score = baseline_df[(baseline_df['model'] == model) & 
                                           (baseline_df['benchmark'] == benchmark)]['score'].mean()
                
                # Get MIPROv2-teacher-gpt4o score for this model and benchmark
                mipro_score = miprov2_teacher_df[(miprov2_teacher_df['model'] == model) & 
                                              (miprov2_teacher_df['benchmark'] == benchmark)]['score'].mean()
                
                # Calculate improvement percentage
                if pd.notna(baseline_score) and pd.notna(mipro_score) and baseline_score > 0:
                    improvement_pct = ((mipro_score - baseline_score) / baseline_score) * 100
                else:
                    improvement_pct = float('nan')
                
                improvement_data.append({
                    'model': model,
                    'benchmark': benchmark,
                    'improvement_pct': improvement_pct
                })
        
        # Create DataFrame from improvement data
        improvement_df = pd.DataFrame(improvement_data)
        
        # Create pivot table for improvement percentages
        plt.figure(figsize=(16, 12))
        improvement_pivot = improvement_df.pivot_table(
            index='benchmark',
            columns='model',
            values='improvement_pct'
        )
        
        # Sort benchmarks by average improvement
        improvement_pivot['avg'] = improvement_pivot.mean(axis=1)
        improvement_pivot = improvement_pivot.sort_values('avg', ascending=False)
        improvement_pivot = improvement_pivot.drop('avg', axis=1)
        
        # Create heatmap for improvement percentages
        ax = sns.heatmap(improvement_pivot, annot=True, cmap='RdYlGn', fmt='.1f', linewidths=0.5,
                        center=0, vmin=-10, vmax=100)
        plt.title('MIPROv2-teacher-gpt4o Improvement Over Baseline (%)', fontsize=16)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Benchmark', fontsize=12)
        plt.tight_layout()
        
        # Save the improvement heatmap
        improvement_path = output_path / 'miprov2_teacher_improvement_heatmap.png'
        plt.savefig(improvement_path, dpi=300)
        plt.close()
    
    print(f"Analysis complete! Visualizations saved to {output_path}")
    return miprov2_teacher_df

if __name__ == "__main__":
    # Compare multiple models with MIPROv2-teacher-gpt4o optimizer
    csv_paths = [
        "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/llama3370b_0305.csv",
        "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/gpt4o_0305.csv",
        "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/gpt4omini_0305.csv"
    ]
    model_names = ["Llama-3-70B", "GPT-4o", "GPT-4o-mini"]
    miprov2_teacher_df = create_miprov2_teacher_heatmap(csv_paths, model_names)
