"""
Custom prompt processor for applying user-defined system prompts to benchmarks.
"""

from pathlib import Path
from typing import Dict, List, Any
import json

from .config import CustomPromptConfig, CustomPromptEntry, BenchmarkConfig, PromptConfig
from .manager import PromptManager


class CustomPromptProcessor:
    """Processes custom system prompts and applies them to benchmarks."""
    
    def __init__(self, custom_config: CustomPromptConfig):
        self.config = custom_config
        self.manager = PromptManager(custom_config.output_dir)
    
    def process_custom_prompts(self) -> Dict[str, BenchmarkConfig]:
        """
        Process custom prompts and create benchmark configurations.
        
        Returns:
            Dictionary mapping benchmark names to BenchmarkConfig objects
        """
        print(f"🎯 Processing {len(self.config.custom_prompts)} custom prompts for {len(self.config.benchmarks)} benchmarks")
        
        benchmark_configs = {}
        
        for benchmark_name in self.config.benchmarks:
            print(f"  📋 Processing benchmark: {benchmark_name}")
            
            # Create a base benchmark config
            benchmark_config = BenchmarkConfig(
                name=benchmark_name,
                base_prompt="Custom prompt configuration",
                signature="Input -> Output",
                task_description=f"Custom prompt optimization for {benchmark_name}",
                prompt_variations=[]
            )
            
            # Create prompt variations from each custom prompt
            for prompt_key, prompt_entry in self.config.custom_prompts.items():
                variation = self._create_prompt_variation(
                    benchmark_name, prompt_key, prompt_entry
                )
                benchmark_config.add_variation(variation)
                print(f"    ✨ Created variation: {variation.name}")
            
            benchmark_configs[benchmark_name] = benchmark_config
        
        return benchmark_configs
    
    def _create_prompt_variation(
        self, 
        benchmark_name: str, 
        prompt_key: str, 
        prompt_entry: CustomPromptEntry
    ) -> PromptConfig:
        """Create a PromptConfig from a custom prompt entry."""
        
        variation_name = f"{benchmark_name}_custom_{prompt_key}"
        
        return PromptConfig(
            name=variation_name,
            description=prompt_entry.description or f"Custom prompt: {prompt_entry.name}",
            instructions=prompt_entry.system_prompt,
            task_description=f"Custom {prompt_entry.name} prompt for {benchmark_name}",
            signature="Input -> Output",
            meta_strategy=f"custom_{prompt_key}"
        )
    
    def save_custom_prompts(self, benchmark_configs: Dict[str, BenchmarkConfig]) -> Path:
        """
        Save custom prompt configurations to files.
        
        Returns:
            Path to the version directory where prompts were saved
        """
        print(f"💾 Saving custom prompts to: {self.config.get_versioned_output_dir()}")
        
        version_dir = self.manager.save_optimized_prompts(
            benchmark_configs, 
            self.config.version
        )
        
        # Also save the original custom config for reference
        config_path = version_dir / "custom_prompt_config.yaml"
        import yaml
        
        config_data = {
            'name': self.config.name,
            'description': self.config.description,
            'custom_prompts': {
                key: {
                    'name': entry.name,
                    'description': entry.description,
                    'system_prompt': entry.system_prompt
                }
                for key, entry in self.config.custom_prompts.items()
            },
            'benchmarks': self.config.benchmarks,
            'evaluation': self.config.evaluation
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, indent=2, default_flow_style=False)
        
        print(f"  📄 Saved custom config reference: {config_path}")
        
        return version_dir
    
    def create_evaluation_configs(self, version_dir: Path) -> Dict[str, Any]:
        """
        Create evaluation configurations for the custom prompts.
        
        Returns:
            Dictionary with evaluation configuration
        """
        evaluation_config = {
            'dataset_mode': self.config.evaluation.get('dataset_mode', 'test'),
            'lm': self.config.evaluation.get('model', 'openrouter/meta-llama/llama-3.3-70b-instruct'),
            'lm_api_base': 'https://openrouter.ai/api/v1',
            'num_threads': self.config.evaluation.get('num_threads', 8),
            'max_concurrent_processes': self.config.evaluation.get('max_concurrent', 4)
        }
        
        # Save evaluation config
        eval_config_path = version_dir / "evaluation_config.json"
        with open(eval_config_path, 'w') as f:
            json.dump(evaluation_config, f, indent=2)
        
        print(f"  ⚙️  Saved evaluation config: {eval_config_path}")
        
        return evaluation_config
