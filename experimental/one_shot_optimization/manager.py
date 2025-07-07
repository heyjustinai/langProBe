"""
Prompt management system for storing and organizing optimized prompts.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .config import BenchmarkConfig, PromptConfig


class PromptManager:
    """Manages storage and organization of prompts."""
    
    def __init__(self, base_output_dir: str = "meta-optimize-prompt"):
        self.base_output_dir = Path(base_output_dir)
        self.ensure_directory_structure()
        
    def ensure_directory_structure(self):
        """Create the required directory structure."""
        # Main directories
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        (self.base_output_dir / "generated").mkdir(exist_ok=True)
        (self.base_output_dir / "manual").mkdir(exist_ok=True)
        (self.base_output_dir / "templates").mkdir(exist_ok=True)
        
        print(f"Prompt storage initialized at: {self.base_output_dir}")
    
    def save_optimized_prompts(
        self, 
        optimized_configs: Dict[str, BenchmarkConfig],
        version: str,
        include_metadata: bool = True
    ) -> Path:
        """
        Save optimized prompts to versioned directory structure.
        
        Args:
            optimized_configs: Dictionary of optimized benchmark configurations
            version: Version string for this optimization run
            include_metadata: Whether to save metadata files
            
        Returns:
            Path to the created version directory
        """
        
        # Create versioned directory
        version_dir = self.base_output_dir / "generated" / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving optimized prompts to: {version_dir}")
        
        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "benchmarks": {},
            "summary": {
                "total_benchmarks": len(optimized_configs),
                "total_variations": 0,
                "strategies_used": set()
            }
        }
        
        # Save prompts for each benchmark
        for benchmark_name, config in optimized_configs.items():
            benchmark_dir = version_dir / benchmark_name
            benchmark_dir.mkdir(exist_ok=True)
            
            print(f"  Saving {benchmark_name} ({len(config.prompt_variations)} variations)")
            
            benchmark_metadata = {
                "name": benchmark_name,
                "base_prompt_length": len(config.base_prompt),
                "signature": config.signature,
                "task_description": config.task_description,
                "variations": []
            }
            
            # Save each prompt variation
            for variation in config.prompt_variations:
                filename = f"{benchmark_name}_{variation.name}_prompt.json"
                filepath = benchmark_dir / filename
                
                # Save the prompt file
                variation.to_json_file(filepath)
                
                # Track metadata
                variation_metadata = {
                    "name": variation.name,
                    "filename": filename,
                    "description": variation.description,
                    "meta_strategy": variation.meta_strategy,
                    "prompt_length": len(variation.instructions),
                    "created_at": datetime.now().isoformat()
                }
                
                benchmark_metadata["variations"].append(variation_metadata)
                metadata["summary"]["total_variations"] += 1
                
                if variation.meta_strategy:
                    metadata["summary"]["strategies_used"].add(variation.meta_strategy)
                
                print(f"    {filename}")
            
            # Save benchmark-specific metadata
            if include_metadata:
                metadata_file = benchmark_dir / "benchmark_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(benchmark_metadata, f, indent=2)
            
            metadata["benchmarks"][benchmark_name] = benchmark_metadata
        
        # Convert set to list for JSON serialization
        metadata["summary"]["strategies_used"] = list(metadata["summary"]["strategies_used"])
        
        # Save overall metadata
        if include_metadata:
            metadata_file = version_dir / "optimization_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  Saved metadata: optimization_metadata.json")
        
        # Update "latest" symlink
        self._update_latest_symlink(version_dir)
        
        print(f"✅ Saved {metadata['summary']['total_variations']} prompt variations for {len(optimized_configs)} benchmarks")
        return version_dir
    
    def _update_latest_symlink(self, version_dir: Path):
        """Update the 'latest' symlink to point to the most recent version."""
        latest_link = self.base_output_dir / "generated" / "latest"
        
        try:
            # Remove existing symlink if it exists
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            
            # Create new symlink
            latest_link.symlink_to(version_dir.name)
            print(f"  Updated 'latest' symlink to: {version_dir.name}")
            
        except Exception as e:
            print(f"  Could not create 'latest' symlink: {e}")
    
    def create_evaluation_config(
        self, 
        version_dir: Path,
        benchmark_configs: Dict[str, BenchmarkConfig]
    ) -> Dict[str, Any]:
        """
        Create evaluation configuration compatible with run_multi_benchmark_evaluation_parallel.py.
        
        Args:
            version_dir: Path to the version directory containing prompts
            benchmark_configs: Dictionary of benchmark configurations
            
        Returns:
            Configuration dictionary for the evaluation script
        """
        
        evaluation_config = []
        
        for benchmark_name, config in benchmark_configs.items():
            benchmark_config = {
                "benchmark": benchmark_name,
                "prompts": [],
                "description": config.task_description
            }
            
            # Add each prompt variation
            for variation in config.prompt_variations:
                prompt_file = version_dir / benchmark_name / f"{benchmark_name}_{variation.name}_prompt.json"
                
                if prompt_file.exists():
                    # Use absolute path to avoid path resolution issues
                    prompt_config = {
                        "name": variation.name,
                        "file": str(prompt_file.resolve()),
                        "description": variation.description
                    }
                    benchmark_config["prompts"].append(prompt_config)
            
            if benchmark_config["prompts"]:
                evaluation_config.append(benchmark_config)
        
        # Save evaluation config
        config_file = version_dir / "evaluation_config.json"
        with open(config_file, 'w') as f:
            json.dump(evaluation_config, f, indent=2)
        
        print(f"⚙️  Created evaluation config: evaluation_config.json")
        return evaluation_config
    
    def load_prompts_from_version(self, version: str) -> Dict[str, BenchmarkConfig]:
        """Load prompts from a specific version."""
        version_dir = self.base_output_dir / "generated" / version
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Version directory not found: {version_dir}")
        
        # Load metadata
        metadata_file = version_dir / "optimization_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Reconstruct benchmark configurations
        configs = {}
        
        for benchmark_name, benchmark_metadata in metadata["benchmarks"].items():
            benchmark_dir = version_dir / benchmark_name
            
            config = BenchmarkConfig(
                name=benchmark_name,
                base_prompt="",  # Will be filled from variations
                signature=benchmark_metadata["signature"],
                task_description=benchmark_metadata["task_description"]
            )
            
            # Load each variation
            for var_metadata in benchmark_metadata["variations"]:
                prompt_file = benchmark_dir / var_metadata["filename"]
                
                if prompt_file.exists():
                    with open(prompt_file) as f:
                        prompt_data = json.load(f)
                    
                    variation = PromptConfig(
                        name=var_metadata["name"],
                        description=var_metadata["description"],
                        instructions=prompt_data["instructions"],
                        task_description=prompt_data.get("task_description"),
                        signature=prompt_data.get("signature"),
                        meta_strategy=var_metadata.get("meta_strategy")
                    )
                    
                    config.add_variation(variation)
                    
                    # Use first variation as base prompt if not set
                    if not config.base_prompt:
                        config.base_prompt = variation.instructions
            
            configs[benchmark_name] = config
        
        return configs
    
    def list_available_versions(self) -> List[str]:
        """List all available prompt versions."""
        generated_dir = self.base_output_dir / "generated"
        
        if not generated_dir.exists():
            return []
        
        versions = []
        for item in generated_dir.iterdir():
            if item.is_dir() and item.name != "latest":
                versions.append(item.name)
        
        # Sort versions by modification time (newest first)
        versions.sort(key=lambda v: (generated_dir / v).stat().st_mtime, reverse=True)
        return versions
    
    def get_version_summary(self, version: str) -> Optional[Dict[str, Any]]:
        """Get summary information for a specific version."""
        version_dir = self.base_output_dir / "generated" / version
        metadata_file = version_dir / "optimization_metadata.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        return metadata["summary"]
    
    def cleanup_old_versions(self, keep_count: int = 5):
        """Remove old versions, keeping only the most recent ones."""
        versions = self.list_available_versions()
        
        if len(versions) <= keep_count:
            print(f"Only {len(versions)} versions exist, no cleanup needed")
            return
        
        versions_to_remove = versions[keep_count:]
        
        print(f"🧹 Cleaning up {len(versions_to_remove)} old versions...")
        
        for version in versions_to_remove:
            version_dir = self.base_output_dir / "generated" / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
                print(f"🗑️  Removed: {version}")
        
        print(f"✅ Cleanup complete, kept {keep_count} most recent versions")
    
    def export_prompts_for_evaluation(
        self, 
        version: str,
        output_format: str = "multi_benchmark_config"
    ) -> Path:
        """
        Export prompts in format compatible with existing evaluation scripts.
        
        Args:
            version: Version to export
            output_format: Format to export in ('multi_benchmark_config', 'individual_files')
            
        Returns:
            Path to exported files
        """
        configs = self.load_prompts_from_version(version)
        version_dir = self.base_output_dir / "generated" / version
        
        if output_format == "multi_benchmark_config":
            # Export in format compatible with run_multi_benchmark_evaluation_parallel.py
            return self._export_multi_benchmark_config(version_dir, configs)
        elif output_format == "individual_files":
            # Copy individual files to a flat structure
            return self._export_individual_files(version_dir, configs)
        else:
            raise ValueError(f"Unknown export format: {output_format}")
    
    def _export_multi_benchmark_config(
        self, 
        version_dir: Path, 
        configs: Dict[str, BenchmarkConfig]
    ) -> Path:
        """Export as multi-benchmark configuration."""
        
        # This is already done in create_evaluation_config
        eval_config = self.create_evaluation_config(version_dir, configs)
        return version_dir / "evaluation_config.json"
    
    def _export_individual_files(
        self, 
        version_dir: Path, 
        configs: Dict[str, BenchmarkConfig]
    ) -> Path:
        """Export as individual prompt files in a flat structure."""
        
        export_dir = version_dir / "individual_prompts"
        export_dir.mkdir(exist_ok=True)
        
        for benchmark_name, config in configs.items():
            for variation in config.prompt_variations:
                source_file = version_dir / benchmark_name / f"{benchmark_name}_{variation.name}_prompt.json"
                dest_file = export_dir / f"{benchmark_name}_{variation.name}_prompt.json"
                
                if source_file.exists():
                    shutil.copy2(source_file, dest_file)
        
        print(f"📦 Exported individual prompt files to: {export_dir}")
        return export_dir 