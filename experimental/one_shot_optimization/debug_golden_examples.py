#!/usr/bin/env python3
"""
Debug script to show exactly how golden examples are being used in meta-prompt optimization.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experimental.one_shot_optimization.optimizer import MetaPromptOptimizer, DEFAULT_META_TEMPLATES
from experimental.one_shot_optimization.config import OptimizationConfig

def debug_golden_examples_usage(benchmark_name: str = "HeartDisease", strategy: str = "self_written_2"):
    """Debug exactly how golden examples are used in optimization."""
    
    print(f"🔍 DEBUGGING Golden Examples Usage")
    print(f"Benchmark: {benchmark_name} | Strategy: {strategy}")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = MetaPromptOptimizer(OptimizationConfig())
    
    # Load golden examples
    golden_examples = optimizer.golden_extractor.load_golden_examples(benchmark_name)
    
    print(f"📊 Loaded {len(golden_examples)} golden examples:")
    for i, example in enumerate(golden_examples, 1):
        inputs_str = ", ".join(f"{k}: {v}" for k, v in list(example["inputs"].items())[:3])
        outputs_str = ", ".join(f"{k}: {v}" for k, v in example["outputs"].items())
        print(f"  {i}. {inputs_str}... -> {outputs_str}")
    
    print(f"\n🎨 FORMATTED EXAMPLES:")
    print("-" * 40)
    
    # Show formatted examples for meta-prompt
    formatted_meta = optimizer._format_golden_examples(golden_examples[:2])
    print(f"For meta-prompt context:\n{formatted_meta}\n")
    
    # Show formatted examples for few-shot
    formatted_few_shot = optimizer._format_few_shot_examples(golden_examples[:3])
    print(f"For few-shot examples:\n{formatted_few_shot}\n")
    
    print(f"🛠️  TEMPLATE PROCESSING:")
    print("-" * 40)
    
    # Get the original template
    original_template = DEFAULT_META_TEMPLATES[strategy]
    print(f"Original template length: {len(original_template)} chars")
    
    # Process template with golden examples
    processed_template = optimizer._process_template_with_golden_examples(original_template, golden_examples)
    print(f"Processed template length: {len(processed_template)} chars")
    
    # Show the differences
    if "{{#if GOLDEN_EXAMPLES}}" in original_template:
        print("✅ Template has conditional golden examples block")
    else:
        print("❌ Template has no conditional golden examples block")
    
    if "{golden_examples}" in processed_template:
        print("✅ Processed template contains golden examples")
    else:
        print("❌ Processed template does NOT contain golden examples")
    
    print(f"\n📝 FINAL META-PROMPT:")
    print("-" * 40)
    
    # Create the final meta-prompt (what gets sent to LLM)
    sample_original_prompt = "You are an expert medical diagnostician. Analyze patient data to predict heart disease."
    
    final_meta_prompt = processed_template.format(
        original_system_prompt=sample_original_prompt
    )
    
    print(f"Meta-prompt length: {len(final_meta_prompt)} chars")
    print(f"First 500 chars:\n{final_meta_prompt[:500]}...")
    
    # Check for golden examples in final prompt
    example_count = final_meta_prompt.count("Example 1:")
    few_shot_count = final_meta_prompt.count("age: 63")  # First example's age
    
    print(f"\n🔍 VERIFICATION:")
    print(f"  'Example 1:' appearances: {example_count}")
    print(f"  First golden example data appearances: {few_shot_count}")
    
    if example_count > 0 or few_shot_count > 0:
        print("✅ Golden examples ARE included in the meta-prompt!")
    else:
        print("❌ Golden examples are NOT included in the meta-prompt!")
    
    return {
        "golden_examples_loaded": len(golden_examples),
        "template_processed": len(processed_template) > len(original_template),
        "examples_in_final_prompt": example_count + few_shot_count > 0,
        "final_prompt_length": len(final_meta_prompt)
    }

def compare_with_without_golden_examples(benchmark_name: str = "HeartDisease", strategy: str = "self_written_2"):
    """Compare optimization with and without golden examples."""
    
    print(f"\n🔄 COMPARISON: With vs Without Golden Examples")
    print("=" * 60)
    
    optimizer = MetaPromptOptimizer(OptimizationConfig())
    original_template = DEFAULT_META_TEMPLATES[strategy]
    sample_prompt = "You are an expert medical diagnostician."
    
    # With golden examples
    golden_examples = optimizer.golden_extractor.load_golden_examples(benchmark_name)
    with_examples = optimizer._process_template_with_golden_examples(original_template, golden_examples)
    with_final = with_examples.format(original_system_prompt=sample_prompt)
    
    # Without golden examples (empty list)
    without_examples = optimizer._process_template_with_golden_examples(original_template, [])
    without_final = without_examples.format(original_system_prompt=sample_prompt)
    
    print(f"📊 Results:")
    print(f"  With golden examples: {len(with_final)} chars")
    print(f"  Without golden examples: {len(without_final)} chars")
    print(f"  Difference: {len(with_final) - len(without_final)} chars")
    
    # Check for specific golden example content
    has_examples_with = "age: 63" in with_final
    has_examples_without = "age: 63" in without_final
    
    print(f"  Contains golden data (with): {has_examples_with}")
    print(f"  Contains golden data (without): {has_examples_without}")
    
    if has_examples_with and not has_examples_without:
        print("✅ Golden examples are correctly included only when available!")
    else:
        print("❌ Something is wrong with the golden examples logic!")

def main():
    """Main debug function."""
    
    import argparse
    parser = argparse.ArgumentParser(description="Debug golden examples usage")
    parser.add_argument("--benchmark", default="HeartDisease", 
                       choices=["HeartDisease", "hover", "judgebench"])
    parser.add_argument("--strategy", default="self_written_2")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare with vs without golden examples")
    
    args = parser.parse_args()
    
    # Run debug
    results = debug_golden_examples_usage(args.benchmark, args.strategy)
    
    # Run comparison if requested
    if args.compare:
        compare_with_without_golden_examples(args.benchmark, args.strategy)
    
    print(f"\n🎯 SUMMARY:")
    print(f"  Golden examples loaded: {results['golden_examples_loaded']}")
    print(f"  Template processed correctly: {results['template_processed']}")
    print(f"  Examples in final prompt: {results['examples_in_final_prompt']}")
    print(f"  Final prompt length: {results['final_prompt_length']} chars")

if __name__ == "__main__":
    main() 