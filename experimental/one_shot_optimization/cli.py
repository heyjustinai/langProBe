"""
Command-line interface for the Prompt Optimization Pipeline.

This module provides the CLI implementation for the prompt optimization system.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

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
    
    # Determine configuration to use
    config_to_use = None
    
    if args.config:
        # Use explicitly specified config
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
        config_to_use = args.config
        
    elif not any([args.benchmarks, args.strategies, args.output_dir]):
        # No arguments provided - try to use default config
        default_config_path = Path("configs/pipeline_configs/default.yaml")
        if default_config_path.exists():
            config_to_use = str(default_config_path)
            print(f"📋 Using default configuration: {default_config_path}")
        else:
            print("⚠️  No default configuration found, using fallback settings")
    
    if config_to_use:
        # Load from configuration file
        orchestrator = EvaluationOrchestrator.create_from_config_file(config_to_use)
    else:
        # Create configuration from arguments (fallback)
        config = create_quick_config(
            benchmarks=args.benchmarks or ["HeartDisease", "judgebench"],
            strategies=args.strategies or ["ce1", "ce2"],  # Use your custom strategies as default
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
    from .config import OptimizationConfig, PipelineConfig
    
    # Try to load from default config to show all available templates
    config_to_use = OptimizationConfig(strategies=[])
    
    default_config_path = Path("configs/pipeline_configs/default.yaml")
    if default_config_path.exists():
        try:
            pipeline_config = PipelineConfig.from_file(default_config_path)
            config_to_use = pipeline_config.optimization
            print(f"📋 Loading strategies from {default_config_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load default config ({e}), showing built-in strategies only")
    
    optimizer = MetaPromptOptimizer(config_to_use)
    strategies = optimizer.available_strategies
    
    # Group strategies by source
    default_strategies = []
    yaml_strategies = []
    custom_strategies = []
    
    for strategy in strategies:
        if strategy in ['test_3', 'test_0', 'test_1', 'openai_v2_research_paper_generated_with_few_data', 
                       'openai_v1_research_paper_generated_with_few_data', 'gemini2.5_research_paper_generated_with_few_data',
                       'metaai_research_paper_generated_with_few_data', 'self_written', 'self_written_2', 'llama_405b']:
            default_strategies.append(strategy)
        elif strategy in config_to_use.custom_templates:
            custom_strategies.append(strategy)
        else:
            yaml_strategies.append(strategy)
    
    # Display strategies by group
    if default_strategies:
        print(f"\n📦 Built-in strategies ({len(default_strategies)}):")
        for i, strategy in enumerate(default_strategies, 1):
            print(f"  {i:2}. {strategy}")
    
    if yaml_strategies:
        print(f"\n📄 YAML template strategies ({len(yaml_strategies)}):")
        for i, strategy in enumerate(yaml_strategies, len(default_strategies) + 1):
            print(f"  {i:2}. {strategy}")
    
    if custom_strategies:
        print(f"\n🔧 Custom config strategies ({len(custom_strategies)}):")
        for i, strategy in enumerate(custom_strategies, len(default_strategies) + len(yaml_strategies) + 1):
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


def cmd_show_logs(args):
    """Show comprehensive logs for a specific optimization run."""
    
    print(f"📊 Showing logs for optimization run: {args.version}")
    
    # Determine version directory
    if args.version.startswith("v"):
        version_dir = Path("meta-optimize-prompt/generated") / args.version
    else:
        version_dir = Path(args.version)
    
    if not version_dir.exists():
        print(f"❌ Version directory not found: {version_dir}")
        return
    
    # Look for log files
    log_dir = version_dir / "llm_logs"
    if not log_dir.exists():
        print(f"❌ No log directory found at: {log_dir}")
        print("   This version may not have comprehensive logging enabled.")
        return
    
    print(f"📁 Log directory: {log_dir}")
    
    # Show summary files
    summary_files = list(log_dir.glob("*_summary_*.json"))
    if summary_files:
        print(f"\n📋 Summary Files:")
        for summary_file in sorted(summary_files):
            print(f"   📄 {summary_file.name}")
    
    # Show detailed log files
    detailed_files = list(log_dir.glob("detailed_log_*.txt"))
    if detailed_files:
        print(f"\n📝 Detailed Log Files:")
        for detailed_file in sorted(detailed_files):
            print(f"   📄 {detailed_file.name}")
    
    # Show raw log files
    raw_files = list(log_dir.glob("llm_calls_*.jsonl"))
    if raw_files:
        print(f"\n🔍 Raw Log Files (JSONL):")
        for raw_file in sorted(raw_files):
            print(f"   📄 {raw_file.name}")
    
    # Combined summary
    combined_file = log_dir / "combined_llm_summary.json"
    if combined_file.exists():
        print(f"\n📊 Combined Summary:")
        try:
            import json
            with open(combined_file) as f:
                summary = json.load(f)
            
            print(f"   Total LLM calls: {summary['total_calls']}")
            print(f"   Total cost: ${summary['total_cost']:.6f}")
            print(f"   Total tokens: {summary['total_tokens']:,}")
            print(f"   Phases logged: {list(summary['phases'].keys())}")
            
            # Show breakdown by phase
            for phase, phase_data in summary['phases'].items():
                print(f"\n   📈 {phase.title()} Phase:")
                print(f"      Calls: {phase_data['total_calls']}")
                print(f"      Cost: ${phase_data['total_cost']:.6f}")
                print(f"      Tokens: {phase_data['total_tokens']:,}")
                
                # Show by benchmark
                if phase_data.get('by_benchmark'):
                    print(f"      By benchmark:")
                    for benchmark, stats in phase_data['by_benchmark'].items():
                        print(f"        {benchmark}: {stats['calls']} calls, ${stats['cost']:.6f}")
                
                # Show by strategy
                if phase_data.get('by_strategy'):
                    print(f"      By strategy:")
                    for strategy, stats in phase_data['by_strategy'].items():
                        print(f"        {strategy}: {stats['calls']} calls, ${stats['cost']:.6f}")
        
        except Exception as e:
            print(f"   ❌ Error reading combined summary: {e}")
    
    # Options for viewing logs
    print(f"\n🔧 View Options:")
    print(f"   📄 View detailed log: cat {log_dir}/detailed_log_*.txt")
    print(f"   📊 View summary: cat {log_dir}/combined_llm_summary.json | jq")
    print(f"   🔍 View raw logs: cat {log_dir}/llm_calls_*.jsonl | jq")
    
    # If user wants to view a specific log
    if args.show_detailed:
        print(f"\n📖 Showing detailed log content...")
        for detailed_file in sorted(detailed_files):
            print(f"\n{'='*80}")
            print(f"FILE: {detailed_file.name}")
            print(f"{'='*80}")
            try:
                with open(detailed_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if args.limit_lines:
                        lines = content.split('\n')
                        if len(lines) > args.limit_lines:
                            content = '\n'.join(lines[:args.limit_lines])
                            content += f"\n\n... (truncated, showing first {args.limit_lines} lines of {len(lines)} total)"
                    print(content)
            except Exception as e:
                print(f"❌ Error reading file: {e}")


def create_parser():
    """Create the CLI argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Prompt Optimization Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration (uses configs/pipeline_configs/default.yaml)
  python -m experimental.one_shot_optimization.cli pipeline
  
  # Quick test with one benchmark
  python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease
  
  # Extract prompts from all benchmarks
  python -m experimental.one_shot_optimization.cli extract
  
  # Run pipeline with specific configuration file
  python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml
  
  # Run pipeline with specific benchmarks (overrides config)
  python -m experimental.one_shot_optimization.cli pipeline --benchmarks HeartDisease hover
  
  # Optimize existing prompts
  python -m experimental.one_shot_optimization.cli optimize --benchmarks HeartDisease
  
  # List available strategies
  python -m experimental.one_shot_optimization.cli list-strategies
  
  # Show logs for a specific run
  python -m experimental.one_shot_optimization.cli show-logs --version v2025_07_11_2103
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
    
    # Show logs command
    show_logs_parser = subparsers.add_parser('show-logs', help='Show comprehensive logs for a specific run')
    show_logs_parser.add_argument('--version', required=True, help='Version to show logs for (e.g., v2025_07_11_2103)')
    show_logs_parser.add_argument('--show-detailed', action='store_true', help='Show detailed log content')
    show_logs_parser.add_argument('--limit-lines', type=int, help='Limit detailed log output to N lines')
    show_logs_parser.set_defaults(func=cmd_show_logs)
    
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