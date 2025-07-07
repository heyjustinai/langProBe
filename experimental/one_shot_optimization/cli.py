"""
Command-line interface for the Prompt Optimization Pipeline.

This module provides the CLI implementation for the prompt optimization system.
"""

import argparse
import os
import sys
from pathlib import Path

from . import (
    EvaluationOrchestrator,
    ConfigManager,
    PipelineConfig,
    OptimizationConfig
)


def create_quick_config(
    benchmarks: list,
    strategies: list,
    output_dir: str = "meta-optimize-prompt"
) -> PipelineConfig:
    """Create a quick configuration for testing."""
    
    optimization_config = OptimizationConfig(
        strategies=strategies,
        sample_size=10,  # Small sample for quick testing
        max_concurrent=3
    )
    
    return PipelineConfig(
        benchmarks=benchmarks,
        optimization=optimization_config,
        output_dir=output_dir,
        evaluation_config={
            "dataset_mode": "test",
            "lm": "openrouter/meta-llama/llama-3.3-70b-instruct",
            "lm_api_base": "https://openrouter.ai/api/v1",
            "lm_api_key": os.getenv('OPENROUTER_API_KEY')
        }
    )


def cmd_extract(args):
    """Extract default prompts from benchmarks."""
    
    print("Extracting default prompts from benchmarks...")
    
    from .extractor import BenchmarkPromptExtractor
    
    extractor = BenchmarkPromptExtractor()
    
    if args.benchmarks:
        # Extract specific benchmarks
        extracted = {}
        for benchmark in args.benchmarks:
            benchmark_dir = extractor.langprobe_dir / benchmark
            prompt_info = extractor.extract_prompt_from_program_file(benchmark_dir)
            if prompt_info:
                config = extractor._create_benchmark_config(prompt_info)
                if config:
                    extracted[benchmark] = config
    else:
        # Extract all benchmarks
        extracted = extractor.extract_all_benchmark_prompts()
    
    if extracted:
        output_path = extractor.save_extracted_prompts(extracted, args.output_dir)
        print(f"✅ Extracted prompts saved to: {output_path}")
    else:
        print("❌ No prompts extracted")


def cmd_optimize(args):
    """Optimize prompts using meta-prompt strategies."""
    
    print("Optimizing prompts using meta-strategies...")
    
    # Load extracted prompts or use minimal configs
    if args.input_dir:
        # Load from previous extraction
        from .manager import PromptManager
        manager = PromptManager()
        configs = manager.load_prompts_from_version(args.input_dir)
    else:
        # Create minimal configs
        orchestrator = EvaluationOrchestrator()
        configs = orchestrator._create_minimal_configs(args.benchmarks or ["HeartDisease", "hover", "judgebench"])
    
    # Create optimization config
    opt_config = OptimizationConfig(
        strategies=args.strategies,
        sample_size=args.sample_size,
        temperature=args.temperature
    )
    
    # Optimize
    from .optimizer import MetaPromptOptimizer
    optimizer = MetaPromptOptimizer(opt_config)
    optimized = optimizer.optimize_benchmark_prompts(configs)
    
    # Save optimized prompts
    from .manager import PromptManager
    manager = PromptManager(args.output_dir)
    version_dir = manager.save_optimized_prompts(optimized, args.version or "optimized")
    
    print(f"✅ Optimized prompts saved to: {version_dir}")


def cmd_pipeline(args):
    """Run the full optimization pipeline."""
    
    print("Running full optimization pipeline...")
    
    if args.config:
        # Load from configuration file
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {args.config}")
            
            # Check if it's in the pipeline_configs directory
            pipeline_config_path = Path("configs/pipeline_configs") / config_path.name
            if pipeline_config_path.exists():
                print(f"💡 Did you mean: {pipeline_config_path}")
                print("   Try: python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml")
            else:
                print("Available configurations:")
                pipeline_configs_dir = Path("configs/pipeline_configs")
                if pipeline_configs_dir.exists():
                    for config_file in pipeline_configs_dir.glob("*.yaml"):
                        print(f"   - configs/pipeline_configs/{config_file.name}")
                else:
                    print("   No configuration directory found")
            sys.exit(1)
            
        orchestrator = EvaluationOrchestrator.create_from_config_file(args.config)
    else:
        # Create configuration from arguments
        config = create_quick_config(
            benchmarks=args.benchmarks or ["HeartDisease", "hover", "judgebench"],
            strategies=args.strategies or ["openai_v1_research_paper_generated_with_few_data"],
            output_dir=args.output_dir or "meta-optimize-prompt"
        )
        orchestrator = EvaluationOrchestrator(config)
    
    # Run pipeline
    results = orchestrator.run_full_pipeline(
        extract_prompts=not args.skip_extraction,
        optimize_prompts=not args.skip_optimization,
        save_prompts=not args.skip_saving,
        run_evaluation=not args.skip_evaluation,
        selected_benchmarks=args.benchmarks
    )
    
    if results["success"]:
        print("\n🎉 Pipeline completed successfully!")
    else:
        print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


def cmd_quick_test(args):
    """Run a quick test with a single benchmark."""
    
    print(f"⚡ Running quick test with {args.benchmark}...")
    
    config = create_quick_config(
        benchmarks=[args.benchmark],
        strategies=["openai_v1_research_paper_generated_with_few_data"],
        output_dir="meta-optimize-prompt"
    )
    
    orchestrator = EvaluationOrchestrator(config)
    results = orchestrator.quick_test(args.benchmark)
    
    if results["success"]:
        print(f"\n✅ Quick test completed successfully!")
    else:
        print(f"\n❌ Quick test failed: {results.get('error', 'Unknown error')}")


def cmd_list_strategies(args):
    """List available meta-prompt strategies."""
    
    print("Available meta-prompt strategies:")
    
    # Import the optimizer to get available strategies
    from .optimizer import MetaPromptOptimizer
    from .config import OptimizationConfig
    
    # Create a dummy config to initialize the optimizer
    dummy_config = OptimizationConfig(strategies=[])
    optimizer = MetaPromptOptimizer(dummy_config)
    strategies = optimizer.available_strategies
    
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i:2}. {strategy}")
    
    print(f"\nTotal: {len(strategies)} strategies available")


def cmd_create_config(args):
    """Create configuration files."""
    
    print("⚙️  Creating configuration files...")
    
    config_manager = ConfigManager(args.config_dir)
    config_manager.create_default_configs()
    
    print(f"✅ Configuration files created in: {args.config_dir}")
    print("Available configurations:")
    
    configs = config_manager.list_pipeline_configs()
    for config in configs:
        print(f"  - {config}")


def cmd_list_versions(args):
    """List available prompt versions."""
    
    from .manager import PromptManager
    
    manager = PromptManager(args.output_dir)
    versions = manager.list_available_versions()
    
    if not versions:
        print("No prompt versions found")
        return
    
    print("📦 Available prompt versions:")
    
    for version in versions:
        summary = manager.get_version_summary(version)
        if summary:
            print(f"  {version}")
            print(f"      Benchmarks: {summary['total_benchmarks']}")
            print(f"      Variations: {summary['total_variations']}")
            print(f"      Strategies: {', '.join(summary['strategies_used'])}")
        else:
            print(f"  {version} (no metadata)")


def create_parser():
    """Create the CLI argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Prompt Optimization Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with one benchmark
  python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease
  
  # Extract prompts from all benchmarks
  python -m experimental.one_shot_optimization.cli extract
  
  # Run full pipeline with configuration file
  python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml
  
  # Run pipeline with specific benchmarks
  python -m experimental.one_shot_optimization.cli pipeline --benchmarks HeartDisease hover
  
  # Optimize existing prompts
  python -m experimental.one_shot_optimization.cli optimize --benchmarks HeartDisease
  
  # List available strategies
  python -m experimental.one_shot_optimization.cli list-strategies
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract default prompts from benchmarks')
    extract_parser.add_argument('--benchmarks', nargs='+', help='Specific benchmarks to extract')
    extract_parser.add_argument('--output-dir', default='extracted_prompts', help='Output directory')
    extract_parser.set_defaults(func=cmd_extract)
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize prompts using meta-strategies')
    optimize_parser.add_argument('--benchmarks', nargs='+', help='Benchmarks to optimize')
    optimize_parser.add_argument('--strategies', nargs='+', default=['openai_v1_research_paper_generated_with_few_data'], help='Meta-prompt strategies')
    optimize_parser.add_argument('--input-dir', help='Input directory with extracted prompts')
    optimize_parser.add_argument('--output-dir', default='meta-optimize-prompt/generated', help='Output directory')
    optimize_parser.add_argument('--version', help='Version name for this optimization run')
    optimize_parser.add_argument('--sample-size', type=int, default=25, help='Sample size for optimization')
    optimize_parser.add_argument('--temperature', type=float, default=0.3, help='Temperature for meta-prompt optimization')
    optimize_parser.set_defaults(func=cmd_optimize)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full optimization pipeline')
    pipeline_parser.add_argument('--config', help='Configuration file to use')
    pipeline_parser.add_argument('--benchmarks', nargs='+', help='Specific benchmarks to process')
    pipeline_parser.add_argument('--strategies', nargs='+', help='Meta-prompt strategies to use')
    pipeline_parser.add_argument('--output-dir', help='Output directory for prompts')
    pipeline_parser.add_argument('--skip-extraction', action='store_true', help='Skip prompt extraction')
    pipeline_parser.add_argument('--skip-optimization', action='store_true', help='Skip meta-prompt optimization')
    pipeline_parser.add_argument('--skip-saving', action='store_true', help='Skip saving prompts')
    pipeline_parser.add_argument('--skip-evaluation', action='store_true', help='Skip running evaluation')
    pipeline_parser.set_defaults(func=cmd_pipeline)
    
    # Quick test command
    quick_parser = subparsers.add_parser('quick-test', help='Run quick test with single benchmark')
    quick_parser.add_argument('--benchmark', default='HeartDisease', help='Benchmark to test with')
    quick_parser.set_defaults(func=cmd_quick_test)
    
    # List strategies command
    list_strategies_parser = subparsers.add_parser('list-strategies', help='List available meta-prompt strategies')
    list_strategies_parser.set_defaults(func=cmd_list_strategies)
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create default configuration files')
    config_parser.add_argument('--config-dir', default='configs', help='Directory to create configs in')
    config_parser.set_defaults(func=cmd_create_config)
    
    # List versions command
    versions_parser = subparsers.add_parser('list-versions', help='List available prompt versions')
    versions_parser.add_argument('--output-dir', default='meta-optimize-prompt', help='Base output directory')
    versions_parser.set_defaults(func=cmd_list_versions)
    
    return parser


def main():
    """Main CLI entry point."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 