import csv
import os
from collections import defaultdict

def analyze_benchmark_results(csv_path):
    """
    Analyze benchmark results from a CSV file using only standard library.
    
    Args:
        csv_path: Path to the CSV file containing benchmark results
    """
    # Load the data
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Convert numeric fields to appropriate types
    for row in data:
        for key in ['score', 'cost', 'optimizer_cost']:
            row[key] = float(row[key])
        for key in ['input_tokens', 'output_tokens', 'optimizer_input_tokens', 'optimizer_output_tokens']:
            row[key] = int(row[key])
    
    # Print basic information
    print(f"Analyzing results from: {os.path.basename(csv_path)}")
    print(f"Total number of benchmark runs: {len(data)}")
    
    # Get unique values
    benchmarks = set(row['benchmark'] for row in data)
    programs = set(row['program'] for row in data)
    optimizers = set(row['optimizer'] for row in data)
    
    print(f"Unique benchmarks: {len(benchmarks)}")
    print(f"Unique programs: {len(programs)}")
    print(f"Unique optimizers: {len(optimizers)}")
    
    # Calculate performance metrics
    scores = [row['score'] for row in data]
    avg_score = sum(scores) / len(scores)
    scores.sort()
    median_score = scores[len(scores) // 2] if len(scores) % 2 == 1 else (scores[len(scores) // 2 - 1] + scores[len(scores) // 2]) / 2
    
    print("\n=== Performance Summary ===")
    print(f"Average score across all benchmarks: {avg_score:.2f}")
    print(f"Median score: {median_score:.2f}")
    print(f"Min score: {min(scores):.2f}")
    print(f"Max score: {max(scores):.2f}")
    
    # Group by benchmark and optimizer to find the best optimizer for each benchmark
    best_by_benchmark = {}
    benchmark_optimizer_scores = defaultdict(lambda: defaultdict(list))
    
    for row in data:
        benchmark = row['benchmark']
        optimizer = row['optimizer']
        score = row['score']
        program = row['program']
        
        benchmark_optimizer_scores[benchmark][optimizer].append((score, program))
    
    print("\n=== Best Optimizer per Benchmark ===")
    for benchmark in sorted(benchmark_optimizer_scores.keys()):
        best_optimizer = None
        best_score = -1
        best_program = None
        
        for optimizer, scores_programs in benchmark_optimizer_scores[benchmark].items():
            avg_optimizer_score = sum(score for score, _ in scores_programs) / len(scores_programs)
            if avg_optimizer_score > best_score:
                best_score = avg_optimizer_score
                best_optimizer = optimizer
                # Get the program from the highest scoring entry
                best_program = max(scores_programs, key=lambda x: x[0])[1]
        
        best_by_benchmark[benchmark] = (best_optimizer, best_score, best_program)
        print(f"{benchmark}: {best_optimizer} (Score: {best_score:.2f}, Program: {best_program})")
    
    # Compare baseline vs optimized performance
    baseline_scores = [row['score'] for row in data if row['optimizer'] == 'Baseline']
    optimized_scores = [row['score'] for row in data if row['optimizer'] != 'Baseline']
    
    if baseline_scores and optimized_scores:
        baseline_avg = sum(baseline_scores) / len(baseline_scores)
        optimized_avg = sum(optimized_scores) / len(optimized_scores)
        improvement = optimized_avg - baseline_avg
        
        print(f"\n=== Baseline vs Optimized ===")
        print(f"Baseline average score: {baseline_avg:.2f}")
        print(f"Optimized average score: {optimized_avg:.2f}")
        print(f"Average improvement: {improvement:.2f} points ({improvement/baseline_avg*100:.2f}%)")
    
    # Analyze optimizer performance
    optimizer_scores = defaultdict(list)
    for row in data:
        optimizer_scores[row['optimizer']].append(row['score'])
    
    print("\n=== Optimizer Performance ===")
    for optimizer in sorted(optimizer_scores.keys()):
        avg_score = sum(optimizer_scores[optimizer]) / len(optimizer_scores[optimizer])
        print(f"{optimizer}: {avg_score:.2f}")
    
    # Analyze token usage
    print("\n=== Token Usage Analysis ===")
    total_input_tokens = sum(row['input_tokens'] for row in data)
    total_output_tokens = sum(row['output_tokens'] for row in data)
    total_optimizer_input_tokens = sum(row['optimizer_input_tokens'] for row in data)
    total_optimizer_output_tokens = sum(row['optimizer_output_tokens'] for row in data)
    
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Total optimizer input tokens: {total_optimizer_input_tokens:,}")
    print(f"Total optimizer output tokens: {total_optimizer_output_tokens:,}")
    
    # Analyze program performance
    program_scores = defaultdict(list)
    for row in data:
        program_scores[row['program']].append(row['score'])
    
    print("\n=== Program Performance ===")
    for program in sorted(program_scores.keys()):
        avg_score = sum(program_scores[program]) / len(program_scores[program])
        print(f"{program}: {avg_score:.2f}")
    
    return data, best_by_benchmark

if __name__ == "__main__":
    # Analyze a single model's results
    csv_path = "/Users/justinai/Documents/Code/langProBe/experiment_data/20250305/llama3370b_0305.csv"
    data, best_optimizers = analyze_benchmark_results(csv_path)
