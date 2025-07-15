"""
Comprehensive LLM Logging System for One-Shot Optimization Pipeline.

This module provides comprehensive logging of all LLM interactions during both
the optimization phase (meta-prompt generation) and evaluation phase (benchmark testing).
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import threading
import os
import sys
from contextlib import contextmanager

import dspy


@dataclass
class LLMCall:
    """Represents a single LLM call with input and output."""
    
    # Call metadata
    call_id: str
    timestamp: str
    phase: str  # 'optimization' or 'evaluation'
    benchmark: Optional[str] = None
    prompt_variation: Optional[str] = None
    strategy: Optional[str] = None
    
    # LLM configuration
    model: str = ""
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    
    # Input/Output
    input_prompt: str = ""
    output_text: str = ""
    
    # Token usage and cost
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    
    # Performance metrics
    latency_ms: float = 0.0
    
    # Error information
    error: Optional[str] = None
    
    # Additional context
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


class LLMLogger:
    """Comprehensive LLM logger that captures all interactions."""
    
    def __init__(self, log_dir: Path, phase: str = "unknown"):
        self.log_dir = Path(log_dir)
        self.phase = phase
        self.calls: List[LLMCall] = []
        self.lock = threading.Lock()
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up log files
        self.raw_log_file = self.log_dir / f"llm_calls_{phase}.jsonl"
        self.summary_file = self.log_dir / f"llm_summary_{phase}.json"
        
        # Current context for tracking
        self.current_benchmark = None
        self.current_prompt_variation = None
        self.current_strategy = None
        
        # Hook into DSPy's LM calls
        self.original_lm_call = None
        self.hook_installed = False
        
    def set_context(self, benchmark: str = None, prompt_variation: str = None, strategy: str = None):
        """Set the current context for logging."""
        with self.lock:
            if benchmark is not None:
                self.current_benchmark = benchmark
            if prompt_variation is not None:
                self.current_prompt_variation = prompt_variation
            if strategy is not None:
                self.current_strategy = strategy
    
    def clear_context(self):
        """Clear the current context."""
        with self.lock:
            self.current_benchmark = None
            self.current_prompt_variation = None
            self.current_strategy = None
    
    def log_call(self, call: LLMCall):
        """Log a single LLM call."""
        with self.lock:
            self.calls.append(call)
            
            # Write to JSONL file immediately for real-time monitoring
            with open(self.raw_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(call), ensure_ascii=False) + '\n')
    
    def install_dspy_hook(self):
        """Install a hook to capture DSPy LLM calls."""
        if self.hook_installed:
            return
            
        # Store original method
        if hasattr(dspy.LM, '__call__'):
            self.original_lm_call = dspy.LM.__call__
            
            def logged_lm_call(lm_self, *args, **kwargs):
                start_time = time.time()
                call_id = str(uuid.uuid4())
                
                # Extract input information
                input_prompt = ""
                if args:
                    if isinstance(args[0], str):
                        input_prompt = args[0]
                    elif isinstance(args[0], dict):
                        input_prompt = json.dumps(args[0], indent=2)
                    else:
                        input_prompt = str(args[0])
                
                # Extract model configuration
                model = getattr(lm_self, 'model', 'unknown')
                temperature = getattr(lm_self, 'temperature', 0.0)
                max_tokens = getattr(lm_self, 'max_tokens', None)
                
                # Make the actual call
                error = None
                output_text = ""
                input_tokens = 0
                output_tokens = 0
                cost = 0.0
                
                try:
                    result = self.original_lm_call(lm_self, *args, **kwargs)
                    
                    # Extract output
                    if hasattr(result, 'choices') and result.choices:
                        output_text = result.choices[0].message.content if hasattr(result.choices[0], 'message') else str(result.choices[0])
                    elif isinstance(result, str):
                        output_text = result
                    else:
                        output_text = str(result)
                    
                    # Extract token usage
                    if hasattr(result, 'usage'):
                        usage = result.usage
                        input_tokens = getattr(usage, 'prompt_tokens', 0)
                        output_tokens = getattr(usage, 'completion_tokens', 0)
                    
                    # Extract cost if available
                    if hasattr(result, 'cost'):
                        cost = result.cost
                        
                except Exception as e:
                    error = str(e)
                    result = None
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Create call record
                call = LLMCall(
                    call_id=call_id,
                    timestamp=datetime.now().isoformat(),
                    phase=self.phase,
                    benchmark=self.current_benchmark,
                    prompt_variation=self.current_prompt_variation,
                    strategy=self.current_strategy,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    input_prompt=input_prompt,
                    output_text=output_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    cost=cost,
                    latency_ms=latency_ms,
                    error=error,
                    context={
                        'args': str(args),
                        'kwargs': str(kwargs),
                        'lm_class': lm_self.__class__.__name__
                    }
                )
                
                self.log_call(call)
                
                if error:
                    raise RuntimeError(error)
                    
                return result
            
            # Install the hook
            dspy.LM.__call__ = logged_lm_call
            self.hook_installed = True
            print(f"✅ Installed DSPy LLM logging hook for phase: {self.phase}")
    
    def uninstall_dspy_hook(self):
        """Uninstall the DSPy LLM hook."""
        if self.hook_installed and self.original_lm_call:
            dspy.LM.__call__ = self.original_lm_call
            self.hook_installed = False
            print(f"✅ Uninstalled DSPy LLM logging hook for phase: {self.phase}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged calls."""
        with self.lock:
            if not self.calls:
                return {
                    "phase": self.phase,
                    "total_calls": 0,
                    "total_cost": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "total_latency_ms": 0.0,
                    "average_latency_ms": 0.0,
                    "error_count": 0,
                    "error_rate": 0.0,
                    "by_benchmark": {},
                    "by_strategy": {},
                    "by_model": {}
                }
            
            # Calculate totals
            total_calls = len(self.calls)
            total_cost = sum(call.cost for call in self.calls)
            total_input_tokens = sum(call.input_tokens for call in self.calls)
            total_output_tokens = sum(call.output_tokens for call in self.calls)
            total_latency_ms = sum(call.latency_ms for call in self.calls)
            error_count = sum(1 for call in self.calls if call.error)
            
            # Group by benchmark
            by_benchmark = {}
            for call in self.calls:
                if call.benchmark:
                    if call.benchmark not in by_benchmark:
                        by_benchmark[call.benchmark] = {
                            "calls": 0, "cost": 0.0, "input_tokens": 0, 
                            "output_tokens": 0, "latency_ms": 0.0, "errors": 0
                        }
                    stats = by_benchmark[call.benchmark]
                    stats["calls"] += 1
                    stats["cost"] += call.cost
                    stats["input_tokens"] += call.input_tokens
                    stats["output_tokens"] += call.output_tokens
                    stats["latency_ms"] += call.latency_ms
                    if call.error:
                        stats["errors"] += 1
            
            # Group by strategy
            by_strategy = {}
            for call in self.calls:
                if call.strategy:
                    if call.strategy not in by_strategy:
                        by_strategy[call.strategy] = {
                            "calls": 0, "cost": 0.0, "input_tokens": 0,
                            "output_tokens": 0, "latency_ms": 0.0, "errors": 0
                        }
                    stats = by_strategy[call.strategy]
                    stats["calls"] += 1
                    stats["cost"] += call.cost
                    stats["input_tokens"] += call.input_tokens
                    stats["output_tokens"] += call.output_tokens
                    stats["latency_ms"] += call.latency_ms
                    if call.error:
                        stats["errors"] += 1
            
            # Group by model
            by_model = {}
            for call in self.calls:
                if call.model:
                    if call.model not in by_model:
                        by_model[call.model] = {
                            "calls": 0, "cost": 0.0, "input_tokens": 0,
                            "output_tokens": 0, "latency_ms": 0.0, "errors": 0
                        }
                    stats = by_model[call.model]
                    stats["calls"] += 1
                    stats["cost"] += call.cost
                    stats["input_tokens"] += call.input_tokens
                    stats["output_tokens"] += call.output_tokens
                    stats["latency_ms"] += call.latency_ms
                    if call.error:
                        stats["errors"] += 1
            
            return {
                "phase": self.phase,
                "total_calls": total_calls,
                "total_cost": round(total_cost, 6),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "total_latency_ms": round(total_latency_ms, 2),
                "average_latency_ms": round(total_latency_ms / total_calls, 2) if total_calls > 0 else 0,
                "error_count": error_count,
                "error_rate": round(error_count / total_calls, 4) if total_calls > 0 else 0,
                "by_benchmark": by_benchmark,
                "by_strategy": by_strategy,
                "by_model": by_model
            }
    
    def save_summary(self):
        """Save a summary of all logged calls."""
        summary = self.get_summary()
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved LLM call summary to: {self.summary_file}")
        print(f"   Total calls: {summary['total_calls']}")
        print(f"   Total cost: ${summary['total_cost']:.6f}")
        print(f"   Total tokens: {summary['total_tokens']:,}")
        print(f"   Errors: {summary['error_count']}")
        
        return summary
    
    def export_detailed_log(self, output_file: Optional[Path] = None):
        """Export detailed log in human-readable format."""
        if output_file is None:
            output_file = self.log_dir / f"detailed_log_{self.phase}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== LLM Call Log - Phase: {self.phase} ===\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Calls: {len(self.calls)}\n\n")
            
            for i, call in enumerate(self.calls, 1):
                f.write(f"{'=' * 60}\n")
                f.write(f"Call #{i} - {call.call_id}\n")
                f.write(f"{'=' * 60}\n")
                f.write(f"Timestamp: {call.timestamp}\n")
                f.write(f"Phase: {call.phase}\n")
                f.write(f"Benchmark: {call.benchmark}\n")
                f.write(f"Prompt Variation: {call.prompt_variation}\n")
                f.write(f"Strategy: {call.strategy}\n")
                f.write(f"Model: {call.model}\n")
                f.write(f"Temperature: {call.temperature}\n")
                f.write(f"Tokens: {call.input_tokens} → {call.output_tokens} (total: {call.total_tokens})\n")
                f.write(f"Cost: ${call.cost:.6f}\n")
                f.write(f"Latency: {call.latency_ms:.2f}ms\n")
                if call.error:
                    f.write(f"Error: {call.error}\n")
                f.write(f"\n--- Input Prompt ---\n")
                f.write(f"{call.input_prompt}\n")
                f.write(f"\n--- Output ---\n")
                f.write(f"{call.output_text}\n")
                f.write(f"\n")
        
        print(f"📄 Exported detailed log to: {output_file}")
        return output_file


class LoggingManager:
    """Manages logging across the entire pipeline."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.loggers: Dict[str, LLMLogger] = {}
        self.active_logger: Optional[LLMLogger] = None
        
        # Create main log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def create_logger(self, phase: str) -> LLMLogger:
        """Create a new logger for a specific phase."""
        if phase in self.loggers:
            return self.loggers[phase]
        
        logger = LLMLogger(self.log_dir, phase)
        self.loggers[phase] = logger
        return logger
    
    def get_logger(self, phase: str) -> Optional[LLMLogger]:
        """Get an existing logger for a phase."""
        return self.loggers.get(phase)
    
    @contextmanager
    def logging_context(self, phase: str, benchmark: str = None, 
                        prompt_variation: str = None, strategy: str = None):
        """Context manager for logging with automatic cleanup."""
        logger = self.create_logger(phase)
        
        # Set context
        logger.set_context(benchmark=benchmark, prompt_variation=prompt_variation, strategy=strategy)
        
        # Install hook if not already installed
        logger.install_dspy_hook()
        
        # Store previous active logger
        previous_logger = self.active_logger
        self.active_logger = logger
        
        try:
            yield logger
        finally:
            # Clear context
            logger.clear_context()
            
            # Restore previous logger
            self.active_logger = previous_logger
    
    def finalize_all_loggers(self):
        """Finalize all loggers and generate summaries."""
        combined_summary = {
            "total_calls": 0,
            "total_cost": 0.0,
            "total_tokens": 0,
            "phases": {}
        }
        
        for phase, logger in self.loggers.items():
            print(f"\n Phase: {phase}")
            summary = logger.save_summary()
            logger.export_detailed_log()
            logger.uninstall_dspy_hook()
            
            combined_summary["phases"][phase] = summary
            combined_summary["total_calls"] += summary["total_calls"]
            combined_summary["total_cost"] += summary["total_cost"]
            combined_summary["total_tokens"] += summary["total_tokens"]
        
        # Save combined summary
        combined_file = self.log_dir / "combined_llm_summary.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Combined LLM Logging Summary:")
        print(f"   Total calls across all phases: {combined_summary['total_calls']}")
        print(f"   Total cost: ${combined_summary['total_cost']:.6f}")
        print(f"   Total tokens: {combined_summary['total_tokens']:,}")
        print(f"   Phases logged: {list(combined_summary['phases'].keys())}")
        print(f"   📄 Saved to: {combined_file}")
        
        return combined_summary


# Global logging manager instance
_global_logging_manager: Optional[LoggingManager] = None


def initialize_logging(log_dir: Path) -> LoggingManager:
    """Initialize global logging manager."""
    global _global_logging_manager
    _global_logging_manager = LoggingManager(log_dir)
    return _global_logging_manager


def get_logging_manager() -> Optional[LoggingManager]:
    """Get the global logging manager."""
    return _global_logging_manager


def finalize_logging() -> Optional[Dict[str, Any]]:
    """Finalize all logging and return combined summary."""
    global _global_logging_manager
    if _global_logging_manager:
        return _global_logging_manager.finalize_all_loggers()
    return None 