#!/usr/bin/env python3
"""
Script to extract golden examples from key benchmarks and save them as JSON files.
Run this script to pre-generate golden examples for optimization.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import from langProBe
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experimental.one_shot_optimization.optimizer import GoldenExampleExtractor

def main():
    """Extract golden examples from key benchmarks."""
    print("🔍 Extracting Golden Examples for Key Benchmarks")
    print("=" * 60)
    
    # Initialize the extractor
    extractor = GoldenExampleExtractor()
    
    # Extract and save golden examples
    all_examples = extractor.extract_and_save_golden_examples(num_examples=5)
    
    print("\n📋 Summary:")
    print(f"  Total benchmarks processed: {len(all_examples)}")
    
    for benchmark_name, examples in all_examples.items():
        print(f"  {benchmark_name}: {len(examples)} examples")
    
    print(f"\n💾 Golden examples saved to: {extractor.golden_examples_dir}")
    print("✅ Golden examples extraction complete!")

if __name__ == "__main__":
    main() 