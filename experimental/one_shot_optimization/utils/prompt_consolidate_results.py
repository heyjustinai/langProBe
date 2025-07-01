#!/usr/bin/env python3
"""
Consolidate evaluation results from multiple prompt variations into one master CSV file.

Usage:
    python prompt_consolidate_results.py [results_directory]
    
Example:
    python prompt_consolidate_results.py meta-optimize-prompt/generated/v2025_06_29_0222/evaluation_results
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse


def find_evaluation_csv_files(results_dir: Path) -> List[Path]:
    """Find all evaluation result CSV files in the directory."""
    csv_files = []
    
    # Look for files matching pattern: evaluation_results_*.csv
    for benchmark_dir in results_dir.iterdir():
        if benchmark_dir.is_dir():
            for csv_file in benchmark_dir.glob("evaluation_results_*.csv"):
                if csv_file.name != "evaluation_results.csv":  # Skip generic name
                    csv_files.append(csv_file)
    
    return sorted(csv_files)


def read_csv_file(csv_file: Path) -> List[Dict[str, Any]]:
    """Read a CSV file and return list of dictionaries."""
    rows = []
    
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"❌ Error reading {csv_file}: {e}")
        
    return rows


def consolidate_csv_files_by_benchmark(csv_files: List[Path], results_dir: Path) -> Dict[str, Any]:
    """Consolidate CSV files by benchmark into separate per-benchmark files."""
    from collections import defaultdict
    
    # Group CSV files by benchmark (based on parent directory)
    benchmark_files = defaultdict(list)
    
    for csv_file in csv_files:
        benchmark = csv_file.parent.name
        benchmark_files[benchmark].append(csv_file)
    
    print(f"🔗 Consolidating {len(csv_files)} CSV files by benchmark...")
    print(f"📊 Found {len(benchmark_files)} benchmarks: {list(benchmark_files.keys())}")
    
    all_results = {}
    total_files_processed = 0
    
    for benchmark, files in benchmark_files.items():
        print(f"\n📊 Processing benchmark: {benchmark}")
        
        benchmark_rows = []
        file_stats = {}
        
        for csv_file in files:
            print(f"  📁 Processing: {csv_file.name}")
            
            rows = read_csv_file(csv_file)
            if rows:
                benchmark_rows.extend(rows)
                file_stats[csv_file.name] = len(rows)
                print(f"    ✅ Added {len(rows)} rows")
            else:
                print(f"    ⚠️  File is empty or unreadable")
                file_stats[csv_file.name] = 0
        
        if benchmark_rows:
            # Sort rows by score (highest to lowest)
            def get_score(row):
                try:
                    return float(row.get('score', 0))
                except (ValueError, TypeError):
                    return 0.0
            
            sorted_rows = sorted(benchmark_rows, key=get_score, reverse=True)
            
            # Create benchmark-specific output file
            benchmark_output = results_dir / benchmark / "consolidated_results.csv"
            
            print(f"  💾 Writing to: {benchmark_output}")
            
            try:
                # Ensure directory exists
                benchmark_output.parent.mkdir(parents=True, exist_ok=True)
                
                with open(benchmark_output, 'w', newline='', encoding='utf-8') as f:
                    if sorted_rows:
                        fieldnames = sorted_rows[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(sorted_rows)
                
                print(f"  ✅ Successfully created consolidated file!")
                print(f"  📈 Total rows: {len(sorted_rows)} (sorted by score)")
                
                # Calculate summary statistics for this benchmark
                stats = analyze_consolidated_results(sorted_rows)
                
                all_results[benchmark] = {
                    "success": True,
                    "output_file": str(benchmark_output),
                    "total_rows": len(sorted_rows),
                    "files_processed": len(files),
                    "file_stats": file_stats,
                    "performance_stats": stats
                }
                
                total_files_processed += len(files)
                
            except Exception as e:
                print(f"  ❌ Error writing consolidated file: {e}")
                all_results[benchmark] = {
                    "success": False,
                    "error": str(e)
                }
        else:
            print(f"  ❌ No valid data found for benchmark: {benchmark}")
            all_results[benchmark] = {
                "success": False,
                "error": "No valid data found"
            }
    
    return {
        "success": len(all_results) > 0,
        "benchmarks_processed": len(all_results),
        "total_files_processed": total_files_processed,
        "benchmark_results": all_results
    }


def analyze_consolidated_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the consolidated results and return performance statistics."""
    
    if not rows:
        return {}
    
    # Group by prompt_variation and calculate stats
    variation_stats = {}
    
    for row in rows:
        variation = row.get('prompt_variation', 'unknown')
        score = row.get('score', '0')
        
        try:
            score_float = float(score)
        except (ValueError, TypeError):
            score_float = 0.0
        
        if variation not in variation_stats:
            variation_stats[variation] = {
                'scores': [],
                'count': 0,
                'total_score': 0.0
            }
        
        variation_stats[variation]['scores'].append(score_float)
        variation_stats[variation]['count'] += 1
        variation_stats[variation]['total_score'] += score_float
    
    # Calculate averages and sort by performance
    performance_summary = {}
    
    for variation, stats in variation_stats.items():
        avg_score = stats['total_score'] / stats['count'] if stats['count'] > 0 else 0.0
        performance_summary[variation] = {
            'average_score': avg_score,
            'count': stats['count'],
            'total_score': stats['total_score']
        }
    
    return performance_summary


def print_performance_summary(stats: Dict[str, Any]):
    """Print a formatted performance summary."""
    
    if not stats:
        print("📊 No performance data available")
        return
    
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print(f"{'Prompt Variation':<50} | {'Avg Score':<9} | {'Samples':<7}")
    print(f"{'-'*50} | {'-'*9} | {'-'*7}")
    
    # Sort by average score (descending)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['average_score'], reverse=True)
    
    for variation, data in sorted_stats:
        avg_score = data['average_score']
        count = data['count']
        print(f"{variation:<50} | {avg_score:8.1f} | {count:6d}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate prompt evaluation results")
    parser.add_argument(
        "results_dir", 
        nargs='?', 
        default="meta-optimize-prompt/latest/evaluation_results",
        help="Directory containing evaluation results (default: meta-optimize-prompt/latest/evaluation_results)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: consolidated_results.csv in results directory)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    if not results_dir.is_dir():
        print(f"❌ Path is not a directory: {results_dir}")
        sys.exit(1)
    
    # Find CSV files
    csv_files = find_evaluation_csv_files(results_dir)
    
    if not csv_files:
        print(f"❌ No evaluation CSV files found in: {results_dir}")
        print(f"Looking for files matching pattern: evaluation_results_*.csv")
        sys.exit(1)
    
    print(f"📁 Found {len(csv_files)} evaluation CSV files in: {results_dir}")
    
    # Consolidate files by benchmark
    result = consolidate_csv_files_by_benchmark(csv_files, results_dir)
    
    if result["success"]:
        print(f"\n📊 CONSOLIDATION SUMMARY:")
        print(f"  📊 Benchmarks processed: {result['benchmarks_processed']}")
        print(f"  📋 Total files processed: {result['total_files_processed']}")
        
        # Print details for each benchmark
        for benchmark, benchmark_result in result["benchmark_results"].items():
            if benchmark_result["success"]:
                print(f"\n🎯 {benchmark}:")
                print(f"  📁 Output: {benchmark_result['output_file']}")
                print(f"  📈 Rows: {benchmark_result['total_rows']}")
                
                # Print performance summary for this benchmark
                if benchmark_result.get("performance_stats"):
                    print(f"  🏆 Performance ranking:")
                    stats = benchmark_result["performance_stats"]
                    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['average_score'], reverse=True)
                    
                    for i, (variation, data) in enumerate(sorted_stats, 1):
                        avg_score = data['average_score']
                        count = data['count']
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                        print(f"    {medal} {i:2d}. {variation:<35} | {avg_score:5.1f} | {count:2d} samples")
            else:
                print(f"\n❌ {benchmark}: {benchmark_result.get('error', 'Unknown error')}")
        
        print(f"\n✅ Consolidation completed successfully!")
        
    else:
        print(f"❌ Consolidation failed: No benchmarks processed successfully")
        sys.exit(1)


if __name__ == "__main__":
    main() 