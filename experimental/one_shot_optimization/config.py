"""
Configuration management for prompt optimization pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import json
from datetime import datetime


@dataclass
class PromptConfig:
    """Configuration for a single prompt variation."""
    name: str
    description: str
    instructions: str
    task_description: Optional[str] = None
    signature: Optional[str] = None
    meta_strategy: Optional[str] = None
    multi_signature: bool = False
    signature_prompts: Optional[Dict[str, str]] = None
    
    def to_json_file(self, filepath: Path) -> None:
        """Save prompt config as JSON file compatible with evaluation system."""
        if self.multi_signature and self.signature_prompts:
            # For multi-signature benchmarks, save the prompts dict directly
            config_data = {
                "task_description": self.task_description or f"{self.name} prompt variation",
                "signature": self.signature or "Multiple signatures",
                "multi_signature": True,
                "prompts": self.signature_prompts
            }
        else:
            # Standard single-signature format
            config_data = {
                "task_description": self.task_description or f"{self.name} prompt variation",
                "signature": self.signature or "Input -> Output",
                "instructions": self.instructions
            }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)


@dataclass 
class BenchmarkConfig:
    """Configuration for a benchmark and its prompt variations."""
    name: str
    base_prompt: str
    signature: str
    task_description: str
    prompt_variations: List[PromptConfig] = field(default_factory=list)
    
    def add_variation(self, variation: PromptConfig) -> None:
        """Add a prompt variation to this benchmark."""
        self.prompt_variations.append(variation)


@dataclass
class OptimizationConfig:
    """Configuration for meta-prompt optimization."""
    strategies: List[str] = field(default_factory=list)
    model: str = "meta-llama/llama-3.3-70b-instruct"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.3
    sample_size: int = 25
    max_concurrent: int = 5
    template_files: List[str] = field(default_factory=list)  # YAML files containing meta-prompt templates
    custom_templates: Dict[str, str] = field(default_factory=dict)  # Direct template definitions
    
    @classmethod
    def from_file(cls, filepath: Path) -> 'OptimizationConfig':
        """Load optimization config from YAML file."""
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class PipelineConfig:
    """Configuration for the full optimization pipeline."""
    benchmarks: List[str] = field(default_factory=list)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output_dir: str = "meta-optimize-prompt"
    version: Optional[str] = None
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set version if not provided."""
        if self.version is None:
            self.version = datetime.now().strftime("v%Y_%m_%d_%H%M")
    
    @classmethod
    def from_file(cls, filepath: Path) -> 'PipelineConfig':
        """Load pipeline config from YAML file."""
        with open(filepath) as f:
            data = yaml.safe_load(f)
            
        # Handle nested optimization config
        if 'optimization' in data:
            data['optimization'] = OptimizationConfig(**data['optimization'])
            
        return cls(**data)
    
    def get_versioned_output_dir(self) -> Path:
        """Get the versioned output directory path."""
        return Path(self.output_dir) / self.version
    
    def save_to_file(self, filepath: Path) -> None:
        """Save pipeline config to YAML file."""
        # Convert to dict for serialization
        config_dict = {
            'benchmarks': self.benchmarks,
            'optimization': {
                'strategies': self.optimization.strategies,
                'model': self.optimization.model,
                'base_url': self.optimization.base_url,
                'temperature': self.optimization.temperature,
                'sample_size': self.optimization.sample_size,
                'max_concurrent': self.optimization.max_concurrent,
                'template_files': self.optimization.template_files,
                'custom_templates': self.optimization.custom_templates
            },
            'output_dir': self.output_dir,
            'version': self.version,
            'evaluation_config': self.evaluation_config
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, indent=2, default_flow_style=False)


class ConfigManager:
    """Manages configuration files and templates."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.config_dir / "pipeline_configs").mkdir(exist_ok=True)
        (self.config_dir / "benchmark_configs").mkdir(exist_ok=True)
        (self.config_dir / "templates").mkdir(exist_ok=True)
    
    def create_default_configs(self) -> None:
        """Create default configuration files."""
        
        # Default pipeline config
        default_pipeline = PipelineConfig(
            benchmarks=["HeartDisease", "hover", "judgebench"],
            optimization=OptimizationConfig(
                strategies=[
                    "context_aware_expert",
                    "cognitive_framework", 
                    "precision_balanced_expert"
                ]
            ),
            evaluation_config={
                "dataset_mode": "test",
                "lm": "openrouter/meta-llama/llama-3.3-70b-instruct",
                "num_threads": 8
            }
        )
        
        config_path = self.config_dir / "pipeline_configs" / "default.yaml"
        default_pipeline.save_to_file(config_path)
        print(f"Created default pipeline config: {config_path}")
        
        # Quick test config
        quick_test = PipelineConfig(
            benchmarks=["HeartDisease"],
            optimization=OptimizationConfig(
                strategies=["context_aware_expert"],
                sample_size=5
            ),
            evaluation_config={
                "dataset_mode": "test"
            }
        )
        
        test_config_path = self.config_dir / "pipeline_configs" / "quick_test.yaml"
        quick_test.save_to_file(test_config_path)
        print(f"Created quick test config: {test_config_path}")
    
    def load_pipeline_config(self, name: str) -> PipelineConfig:
        """Load a pipeline configuration by name."""
        config_path = self.config_dir / "pipeline_configs" / f"{name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_path}")
        return PipelineConfig.from_file(config_path)
    
    def list_pipeline_configs(self) -> List[str]:
        """List available pipeline configurations."""
        config_dir = self.config_dir / "pipeline_configs"
        return [f.stem for f in config_dir.glob("*.yaml")] 