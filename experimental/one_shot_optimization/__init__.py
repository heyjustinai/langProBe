"""
One-Shot Optimization Pipeline for LangProBe

This package provides tools for extracting, optimizing, and managing prompts
across multiple benchmarks using meta-prompt optimization techniques.
"""

# Load environment variables from .env file
import os
from pathlib import Path
from dotenv import load_dotenv

# Try to load from .env file in the project root
root_dir = Path(__file__).parent.parent.parent
env_path = root_dir / '.env'
if env_path.exists():
    print(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)
else:
    print(f"No .env file found at {env_path}")

from .extractor import BenchmarkPromptExtractor
from .optimizer import MetaPromptOptimizer  
from .manager import PromptManager
from .orchestrator import EvaluationOrchestrator
from .config import PipelineConfig, OptimizationConfig, ConfigManager

__all__ = [
    'BenchmarkPromptExtractor',
    'MetaPromptOptimizer',
    'PromptManager', 
    'EvaluationOrchestrator',
    'PipelineConfig',
    'OptimizationConfig',
    'ConfigManager'
]

__version__ = "1.0.0" 