"""
One-Shot Optimization Pipeline for LangProBe

This package provides tools for extracting, optimizing, and managing prompts
across multiple benchmarks using meta-prompt optimization techniques.
"""

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