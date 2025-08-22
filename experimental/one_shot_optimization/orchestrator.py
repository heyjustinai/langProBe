"""
Evaluation orchestrator for the full prompt optimization pipeline.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import PipelineConfig, ConfigManager
from .extractor import BenchmarkPromptExtractor
from .optimizer import MetaPromptOptimizer  
from .manager import PromptManager
from .llm_logger import initialize_logging, get_logging_manager, finalize_logging


class EvaluationOrchestrator:
    """Orchestrates the full prompt optimization and evaluation pipeline."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        if config is None:
            # Create default config
            config = PipelineConfig()
        
        self.config = config
        self.extractor = BenchmarkPromptExtractor()
        self.optimizer = MetaPromptOptimizer(config.optimization)
        self.manager = PromptManager(config.output_dir)
        
    def run_full_pipeline(
        self, 
        extract_prompts: bool = True,
        optimize_prompts: bool = True,
        save_prompts: bool = True,
        run_evaluation: bool = True,
        selected_benchmarks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete optimization pipeline.
        
        Args:
            extract_prompts: Whether to extract prompts from benchmarks
            optimize_prompts: Whether to run meta-prompt optimization
            save_prompts: Whether to save optimized prompts
            run_evaluation: Whether to run multi-benchmark evaluation
            selected_benchmarks: Specific benchmarks to process (None for all)
            
        Returns:
            Dictionary with pipeline results and metadata
        """
        
        print("🚀 Starting Full Prompt Optimization Pipeline")
        print(f"📋 Configuration: {self.config.version}")
        print(f"🎯 Target benchmarks: {selected_benchmarks or self.config.benchmarks or 'All discovered'}")
        
        results = {
            "version": self.config.version,
            "config": self.config,
            "stages": {},
            "success": False
        }
        
        # Initialize comprehensive LLM logging
        log_dir = self.config.get_versioned_output_dir() / "llm_logs"
        logging_manager = initialize_logging(log_dir)
        print(f"📊 Initialized LLM logging at: {log_dir}")
        
        try:
            # Stage 1: Extract prompts from benchmarks
            if extract_prompts:
                print(f"\n{'='*60}")
                print("STAGE 1: Extracting Default Prompts")
                print(f"{'='*60}")
                
                extracted_configs = self._extract_benchmark_prompts(selected_benchmarks)
                results["stages"]["extraction"] = {
                    "success": True,
                    "extracted_benchmarks": list(extracted_configs.keys()),
                    "total_prompts": sum(len(config.prompt_variations) for config in extracted_configs.values())
                }
                
            else:
                # Use existing configurations or create minimal ones
                extracted_configs = self._create_minimal_configs(selected_benchmarks or self.config.benchmarks)
                results["stages"]["extraction"] = {
                    "success": True,
                    "skipped": True,
                    "note": "Using minimal configurations"
                }
            
            # Stage 2: Optimize prompts using meta-strategies
            if optimize_prompts:
                print(f"\n{'='*60}")
                print("STAGE 2: Meta-Prompt Optimization")
                print(f"{'='*60}")
                
                # Enable logging for optimization phase
                with logging_manager.logging_context("optimization"):
                    optimized_configs = self.optimizer.optimize_benchmark_prompts(extracted_configs)
                
                # Validate optimized prompts
                validation_results = self.optimizer.validate_optimized_prompts(optimized_configs)
                
                results["stages"]["optimization"] = {
                    "success": True,
                    "optimized_benchmarks": list(optimized_configs.keys()),
                    "total_variations": sum(len(config.prompt_variations) for config in optimized_configs.values()),
                    "validation_results": validation_results,
                    "summary": self.optimizer.get_optimization_summary(optimized_configs)
                }
                
            else:
                optimized_configs = extracted_configs
                results["stages"]["optimization"] = {
                    "success": True,
                    "skipped": True
                }
            
            # Stage 3: Save optimized prompts
            if save_prompts:
                print(f"\n{'='*60}")
                print("STAGE 3: Saving Optimized Prompts")
                print(f"{'='*60}")
                
                version_dir = self.manager.save_optimized_prompts(
                    optimized_configs, 
                    self.config.version
                )
                
                # Create evaluation configuration
                eval_config = self.manager.create_evaluation_config(version_dir, optimized_configs)
                
                results["stages"]["saving"] = {
                    "success": True,
                    "version_dir": str(version_dir),
                    "evaluation_config_path": str(version_dir / "evaluation_config.json")
                }
                
            else:
                version_dir = None
                results["stages"]["saving"] = {
                    "success": True,
                    "skipped": True
                }
            
            # Stage 4: Run multi-benchmark evaluation
            if run_evaluation and version_dir:
                print(f"\n{'='*60}")
                print("STAGE 4: Multi-Benchmark Evaluation")
                print(f"{'='*60}")
                
                # Enable logging for evaluation phase
                with logging_manager.logging_context("evaluation"):
                    evaluation_results = self._run_multi_benchmark_evaluation(version_dir, optimized_configs)
                
                results["stages"]["evaluation"] = evaluation_results
                
            else:
                results["stages"]["evaluation"] = {
                    "success": True,
                    "skipped": True,
                    "reason": "No evaluation requested or no saved prompts"
                }
            
            # Pipeline completed successfully
            results["success"] = True
            print(f"\n{'='*60}")
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            self._print_pipeline_summary(results)
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            results["error"] = str(e)
            results["success"] = False
            
        finally:
            # Finalize LLM logging and generate summaries
            print(f"\n{'='*60}")
            print("📊 Finalizing LLM Logging")
            print(f"{'='*60}")
            
            logging_summary = finalize_logging()
            if logging_summary:
                results["llm_logging_summary"] = logging_summary
                
        return results
    
    def _extract_benchmark_prompts(self, selected_benchmarks: Optional[List[str]]) -> Dict[str, Any]:
        """Extract prompts from benchmark definitions."""
        
        all_extracted = self.extractor.extract_all_benchmark_prompts()
        
        # Use selected_benchmarks if provided, otherwise fall back to config benchmarks
        target_benchmarks = selected_benchmarks or self.config.benchmarks
        
        if target_benchmarks:
            # Filter to selected benchmarks
            filtered = {name: config for name, config in all_extracted.items() 
                       if name in target_benchmarks}
            
            missing = set(target_benchmarks) - set(filtered.keys())
            if missing:
                print(f"  Could not extract prompts for: {missing}")
            
            print(f" Filtering to configured benchmarks: {target_benchmarks}")
            return filtered
        
        # If no benchmarks specified anywhere, extract all
        print("No benchmarks specified - extracting all available")
        return all_extracted
    
    def _create_minimal_configs(self, benchmark_names: List[str]) -> Dict[str, Any]:
        """Create minimal benchmark configurations when extraction is skipped."""
        from .config import BenchmarkConfig, PromptConfig
        
        minimal_configs = {}
        
        for name in benchmark_names:
            config = BenchmarkConfig(
                name=name,
                base_prompt=f"You are an expert assistant for {name} tasks. Provide accurate and helpful responses.",
                signature="Input -> Output",
                task_description=self.extractor._benchmark_name_to_description(name)
            )
            
            default_prompt = PromptConfig(
                name="minimal_default",
                description=f"Minimal default prompt for {name}",
                instructions=config.base_prompt,
                task_description=config.task_description,
                signature=config.signature
            )
            
            config.add_variation(default_prompt)
            minimal_configs[name] = config
        
        return minimal_configs
    
    def _run_multi_benchmark_evaluation(
        self, 
        version_dir: Path, 
        optimized_configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the multi-benchmark evaluation script."""
        
        # Convert optimized_configs dict to benchmark_configs list format
        benchmark_configs = []
        
        for benchmark_name, config in optimized_configs.items():
            for variation in config.prompt_variations:
                prompt_file_path = version_dir / benchmark_name / f"{benchmark_name}_{variation.name}_prompt.json"
                
                if prompt_file_path.exists():
                    # Use relative path from script location
                    relative_path = prompt_file_path.relative_to(version_dir)
                    
                    benchmark_config = {
                        "benchmark": benchmark_name,
                        "prompt_name": variation.name,
                        "prompt_file": str(relative_path),
                        "description": variation.description,
                        "strategy": variation.name.replace("meta_", "").replace("extracted_", "")
                    }
                    benchmark_configs.append(benchmark_config)
        
        # Create a dynamic evaluation script that uses the optimized prompts
        evaluation_script = self._generate_evaluation_script(version_dir, benchmark_configs)
        
        try:
            print(f" Running evaluation script: {evaluation_script}")
            
            # Set environment variables (convert all values to strings)
            env = os.environ.copy()
            for key, value in self.config.evaluation_config.items():
                env[key] = str(value)
            
            # Force unbuffered output
            env['PYTHONUNBUFFERED'] = '1'
            
            # Set the num_threads environment variable for thread distribution
            if 'num_threads' in self.config.evaluation_config:
                print(f"🧵 Using {self.config.evaluation_config['num_threads']} threads from config")
                env['num_threads'] = str(self.config.evaluation_config['num_threads'])
            
            # Set the max_concurrent_processes environment variable
            if 'max_concurrent_processes' in self.config.evaluation_config:
                print(f"🔄 Using {self.config.evaluation_config['max_concurrent_processes']} max concurrent processes from config")
                env['max_concurrent_processes'] = str(self.config.evaluation_config['max_concurrent_processes'])
            
            # Set other evaluation config environment variables
            if 'dataset_mode' in self.config.evaluation_config:
                print(f"📊 Using dataset_mode: {self.config.evaluation_config['dataset_mode']} from config")
                env['dataset_mode'] = str(self.config.evaluation_config['dataset_mode'])
            
            if 'lm' in self.config.evaluation_config:
                print(f"🤖 Using LM: {self.config.evaluation_config['lm']} from config")
                env['lm'] = str(self.config.evaluation_config['lm'])
            
            if 'lm_api_base' in self.config.evaluation_config:
                print(f"🌐 Using API base: {self.config.evaluation_config['lm_api_base']} from config")
                env['lm_api_base'] = str(self.config.evaluation_config['lm_api_base'])
            
            # Run the evaluation with real-time output
            process = subprocess.Popen(
                ["python", "-u", str(evaluation_script)],  # -u for unbuffered
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr with stdout
                universal_newlines=True,
                env=env,
                bufsize=0  # Unbuffered
            )
            
            # Read output in real-time using iter() approach
            print("    📋 Starting evaluation...")
            
            full_output = []
            for line in iter(process.stdout.readline, ''):
                full_output.append(line)
                # Print the line in real-time with proper indentation
                print(f"    {line.rstrip()}")
            
            # Wait for process to finish
            process.wait()
            stdout_text = ''.join(full_output)
            
            if process.returncode == 0:
                # Check if evaluations actually succeeded by parsing the output
                successful_count = 0
                total_count = 0
                
                # Parse the output to extract success/failure counts
                for line in stdout_text.split('\n'):
                    if "✅ Successful:" in line:
                        try:
                            # Extract "X/Y" from "✅ Successful: X/Y"
                            parts = line.split("✅ Successful: ")[1].split("/")
                            successful_count = int(parts[0])
                            total_count = int(parts[1])
                            break
                        except (IndexError, ValueError):
                            pass
                
                if successful_count > 0:
                    print(f"\n✅ Evaluation completed successfully - {successful_count}/{total_count} evaluations succeeded")
                    
                    return {
                        "success": True,
                        "output": stdout_text,
                        "evaluation_script": str(evaluation_script),
                        "successful_evaluations": successful_count,
                        "total_evaluations": total_count
                    }
                else:
                    print(f"\n❌ Evaluation script ran but all evaluations failed - {successful_count}/{total_count} succeeded")
                    return {
                        "success": False,
                        "error": f"All {total_count} evaluations failed - see output above for details",
                        "output": stdout_text,
                        "evaluation_script": str(evaluation_script),
                        "successful_evaluations": successful_count,
                        "total_evaluations": total_count
                    }
            else:
                print(f"\n❌ Evaluation failed with return code: {process.returncode}")
                return {
                    "success": False,
                    "error": "See output above for details",
                    "return_code": process.returncode
                }
                
        except Exception as e:
            print(f"❌ Error running evaluation: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_evaluation_script(
        self, 
        version_dir: Path, 
        benchmark_configs: List[Dict[str, Any]]
    ) -> Path:
        """Generate evaluation script with all prompt variations."""
        
        # Group configs by benchmark for the template
        from collections import defaultdict
        configs_by_benchmark = defaultdict(list)
        for config in benchmark_configs:
            configs_by_benchmark[config["benchmark"]].append({
                "name": config["prompt_name"],
                "file": config["prompt_file"],
                "description": config.get("description", f"Optimized prompt using {config.get('strategy', 'unknown')} strategy")
            })
        
        # Convert to the format expected by the template
        template_configs = []
        for benchmark, prompts in configs_by_benchmark.items():
            template_configs.append({
                "benchmark": benchmark,
                "prompts": prompts,
                "description": f"{benchmark.title()} benchmark evaluation"
            })
        
        evaluation_script = f'''#!/usr/bin/env python
"""
Auto-generated evaluation script for prompt optimization version: {version_dir.name}
"""

import subprocess
import sys
import os
import time
from pathlib import Path

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Add current directory to path for logging integration
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Configuration for benchmark-prompt mappings
BENCHMARK_CONFIGS = {template_configs}

# Initialize LLM logging for evaluation phase
def initialize_evaluation_logging():
    """Initialize LLM logging for the evaluation phase."""
    try:
        from experimental.one_shot_optimization.llm_logger import initialize_logging
        log_dir = Path(__file__).parent / "llm_logs"
        logging_manager = initialize_logging(log_dir)
        return logging_manager
    except ImportError as e:
        print(f"⚠️  LLM logging not available: {{e}}")
        return None

def flatten_benchmark_prompt_configs(benchmark_configs):
    """Flatten the nested benchmark-prompt structure into individual configurations."""
    flattened_configs = []
    
    for benchmark_config in benchmark_configs:
        benchmark = benchmark_config["benchmark"]
        benchmark_desc = benchmark_config["description"]
        
        for prompt_config in benchmark_config["prompts"]:
            flattened_config = {{
                "benchmark": benchmark,
                "benchmark_description": benchmark_desc,
                "prompt_name": prompt_config["name"],
                "prompt_file": prompt_config["file"],
                "prompt_description": prompt_config["description"],
                "run_id": f"{{benchmark}}_{{prompt_config['name']}}"
            }}
            flattened_configs.append(flattened_config)
    
    return flattened_configs

def create_output_directory(base_path, benchmark):
    """Create and return the output directory path for a benchmark."""
    output_dir = Path(base_path) / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)

def convert_txt_to_csv(benchmark_dir: Path, prompt_name: str) -> bool:
    """
    Convert .txt result files to CSV format when the expected CSV doesn't exist.
    
    Args:
        benchmark_dir: Directory containing the benchmark results
        prompt_name: Name of the prompt variation
        
    Returns:
        True if conversion was successful, False otherwise
    """
    import csv
    
    # Look for .txt files that might contain results for this prompt
    txt_files = list(benchmark_dir.glob("*.txt"))
    
    if not txt_files:
        return False
    
    csv_rows = []
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r') as f:
                content = f.read().strip()
                lines = content.split('\\n')
                
                if len(lines) < 2:
                    continue
                    
                header_line = lines[0]
                
                # Check if this looks like a CSV header
                if ',' in header_line and 'score' in header_line.lower():
                    headers = [h.strip() for h in header_line.split(',')]
                    
                    for i, line in enumerate(lines[1:], 2):
                        if line.strip():
                            # Use proper CSV parsing to handle complex values
                            import csv
                            import io
                            
                            try:
                                # Parse the line using CSV reader to handle quoted values and commas
                                csv_reader = csv.reader(io.StringIO(line))
                                values = next(csv_reader)
                                values = [v.strip() for v in values]
                            except Exception as csv_error:
                                # Fallback to simple split if CSV parsing fails
                                print(f"    ⚠️  CSV parsing failed for line {{i}}, using simple split: {{csv_error}}")
                                values = [v.strip() for v in line.split(',')]
                            
                            # Clean values: remove Python object representations
                            cleaned_values = []
                            for value in values:
                                # Remove common Python object patterns
                                if value.startswith('EvaluationResult(') or value.startswith('results=<'):
                                    # Extract numeric score if possible
                                    import re
                                    score_match = re.search(r'score=([0-9.]+)', value)
                                    if score_match:
                                        cleaned_values.append(score_match.group(1))
                                    else:
                                        cleaned_values.append('')
                                elif value.startswith('<') and value.endswith('>'):
                                    # Skip object representations
                                    cleaned_values.append('')
                                else:
                                    cleaned_values.append(value)
                            
                            # Create a row dict, padding missing values
                            row = {{}}
                            for j, header in enumerate(headers):
                                if j < len(cleaned_values):
                                    row[header] = cleaned_values[j]
                                else:
                                    row[header] = ''
                            
                            # Add metadata
                            row['prompt_variation'] = prompt_name
                            row['file_name'] = txt_file.stem
                            row['source_file'] = str(txt_file.name)
                            
                            csv_rows.append(row)
                            
        except Exception as e:
            print(f"    ⚠️  Error reading {{txt_file.name}}: {{e}}")
            sys.stdout.flush()
            continue
    
    if csv_rows:
        # Save as CSV file
        csv_path = benchmark_dir / f"evaluation_results_{{prompt_name}}.csv"
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if csv_rows:
                    # Two-pass approach: First collect all possible fieldnames
                    all_fieldnames = set()
                    for row in csv_rows:
                        all_fieldnames.update(row.keys())
                    
                    # Remove empty fieldnames if they exist
                    all_fieldnames.discard('')
                    all_fieldnames.discard(None)
                    
                    # Sort for consistent ordering
                    fieldnames = sorted(all_fieldnames)
                    
                    # Second pass: Normalize all rows to have the same fields
                    normalized_rows = []
                    for row in csv_rows:
                        # Clean the row: remove empty/None keys and ensure all fields exist
                        cleaned_row = {{}}
                        for field in fieldnames:
                            if field in row and row[field] is not None:
                                cleaned_row[field] = row[field]
                            else:
                                cleaned_row[field] = ''
                        normalized_rows.append(cleaned_row)
                    
                    # Write CSV with normalized data
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(normalized_rows)
            
            print(f"    Converted {{len(txt_files)}} TXT files to CSV: {{csv_path.name}}")
            print(f"    Generated {{len(csv_rows)}} result rows with {{len(fieldnames)}} fields")
            sys.stdout.flush()
            return True
            
        except Exception as e:
            print(f"    ❌ Error writing CSV file: {{e}}")
            sys.stdout.flush()
            return False
    
    return False

def main():
    """Run evaluation with optimized prompts."""
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Save the current directory for later reference
    version_dir = script_dir
    
    # Initialize LLM logging for evaluation phase
    logging_manager = initialize_evaluation_logging()
    if logging_manager:
        print("📊 Initialized LLM logging for evaluation phase")
    
    # Flatten configurations
    all_configs = flatten_benchmark_prompt_configs(BENCHMARK_CONFIGS)
    
    # Convert relative paths to absolute paths
    for config in all_configs:
        config["prompt_file"] = str(version_dir / config["prompt_file"])
    
    # Change to a temporary directory to avoid dataset loading conflicts
    # (HuggingFace datasets can get confused by local directories with dataset names)
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="langprobe_eval_")
    os.chdir(temp_dir)
    print(f"📁 Working directory: {{temp_dir}}")
    sys.stdout.flush()
    
    # Calculate optimal thread distribution based on concurrent processes
    total_threads = int(os.getenv('num_threads', '8'))
    
    # Group configs by benchmark to manage rate limiting better
    from collections import defaultdict
    configs_by_benchmark = defaultdict(list)
    for config in all_configs:
        configs_by_benchmark[config["benchmark"]].append(config)
    
    # Determine max concurrent processes per benchmark from environment
    max_concurrent_per_benchmark = int(os.getenv('max_concurrent_processes', '2'))
    
    # Calculate threads per process to avoid oversubscription
    threads_per_process = max(1, total_threads // max_concurrent_per_benchmark)
    print(f"🧵 Thread allocation: {{total_threads}} total threads ÷ {{max_concurrent_per_benchmark}} concurrent processes = {{threads_per_process}} threads per process")
    sys.stdout.flush()
    
    # Base arguments using config values
    dataset_mode = os.getenv('dataset_mode', 'test')
    lm_model = os.getenv('lm', 'openrouter/meta-llama/llama-3.3-70b-instruct')
    lm_api_base = os.getenv('lm_api_base', 'https://openrouter.ai/api/v1')
    
    base_args = [
        f"--dataset_mode={{dataset_mode}}",
        f"--lm={{lm_model}}",
        f"--lm_api_base={{lm_api_base}}",
        f"--lm_api_key={{os.getenv('OPENROUTER_API_KEY')}}",
        f"--num_threads={{threads_per_process}}"
    ]
    
    print(f"🎯 Running evaluation for {{len(all_configs)}} prompt variations")
    sys.stdout.flush()
    
    print(f" Benchmarks to evaluate: {{list(configs_by_benchmark.keys())}}")
    sys.stdout.flush()
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output for subprocesses
    all_results = []
    
    # Process benchmarks sequentially to avoid overwhelming the API
    for benchmark, benchmark_configs in configs_by_benchmark.items():
        print(f"\\n Starting {{benchmark}} evaluation ({{len(benchmark_configs)}} variations)")
        sys.stdout.flush()
        
        processes = []
        start_time = time.time()
        successful = 0
        
        # Start processes with rate limiting (max concurrent per benchmark)
        # Read from environment to ensure consistency with thread calculation
        max_concurrent_per_benchmark = int(os.getenv('max_concurrent_processes', '2'))
        max_concurrent = min(max_concurrent_per_benchmark, len(benchmark_configs))
        
        for i, config in enumerate(benchmark_configs):
            prompt_name = config["prompt_name"]
            prompt_file = config["prompt_file"]
            run_id = config["run_id"]
            
            # Check if prompt file exists
            if not Path(prompt_file).exists():
                print(f"❌ Error: Prompt file {{prompt_file}} not found for {{run_id}}!")
                sys.stdout.flush()
                all_results.append({{**config, 'success': False, 'duration': 0, 'error': 'Prompt file not found'}})
                continue
            
            # Create output directory (use absolute path since we're in temp dir)
            output_path = create_output_directory(str(version_dir / "evaluation_results"), benchmark)
            
            # Set unique cache directory for each run to prevent conflicts
            timestamp = str(int(time.time() * 1000))  # millisecond timestamp
            unique_id = f"{{prompt_name}}_{{timestamp}}_{{os.getpid()}}_{{i}}"
            
            run_env = env.copy()
            run_env["DSPY_CACHEDIR"] = f"{{output_path}}/.dspy_cache_{{unique_id}}"
            
            # Clear any conflicting environment variables
            run_env.pop("DSPY_CACHE_SUFFIX", None)
            run_env.pop("DSPY_OUTPUT_PREFIX", None)
            
            # Build command with explicit output path control (use absolute path)
            # Use python -u for unbuffered output
            cmd = [
                "python", "-u", str(version_dir / "../../../evaluate_custom_prompt.py"),
                f"--prompt_file={{prompt_file}}",
                f"--benchmarks={{benchmark}}",
                f"--file_path={{output_path}}",
                f"--output_suffix={{prompt_name}}",
                *base_args
            ]
            
            # Enable subprocess logging by setting environment variables
            run_env["ENABLE_LLM_LOGGING"] = "1"
            run_env["LLM_LOG_DIR"] = str(version_dir / "llm_logs")  # Use absolute path
            run_env["LLM_LOG_PHASE"] = "evaluation"
            run_env["LLM_LOG_BENCHMARK"] = benchmark
            run_env["LLM_LOG_PROMPT_VARIATION"] = prompt_name
            run_env["LLM_LOG_STRATEGY"] = f"{{prompt_name}}_{{unique_id}}"  # Make unique per subprocess
            
            # Add project root to PYTHONPATH so subprocess can import logging module
            project_root = str(version_dir.parent.parent.parent)  # Go up from generated/vXXXX to project root
            if "PYTHONPATH" in run_env:
                run_env["PYTHONPATH"] = f"{{project_root}}:{{run_env['PYTHONPATH']}}"
            else:
                run_env["PYTHONPATH"] = project_root
            
            print(f"🔧 Starting [{{run_id}}] (batch {{i+1}}/{{len(benchmark_configs)}})")
            sys.stdout.flush()
            
            # Set logging context for this specific benchmark and prompt variation
            try:
                process = subprocess.Popen(
                    cmd, 
                    env=run_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                processes.append({{
                    'process': process,
                    'run_id': run_id,
                    'start_time': time.time(),
                    'config': config
                }})
                
            except Exception as e:
                print(f"❌ Error starting process [{{run_id}}]: {{e}}")
                sys.stdout.flush()
                all_results.append({{**config, 'success': False, 'duration': 0, 'error': str(e)}})
                continue
                
            # Wait for process slots if we've reached the concurrent limit
            while len(processes) >= max_concurrent:
                for j, proc_info in enumerate(processes[:]):
                    process = proc_info['process']
                    run_id = proc_info['run_id']
                    duration = time.time() - proc_info['start_time']
                    
                    # Check for timeout (4 hours per evaluation for full datasets)
                    timeout_limit = 14400  # 4 hours in seconds
                    
                    if duration > timeout_limit:
                        print(f"⏰ [{{run_id}}] TIMEOUT after {{duration:.1f}}s - Terminating process")
                        sys.stdout.flush()
                        try:
                            process.terminate()
                            time.sleep(2)  # Give it a moment to terminate gracefully
                            if process.poll() is None:
                                process.kill()  # Force kill if terminate didn't work
                            all_results.append({{**proc_info['config'], 'success': False, 'duration': duration, 'error': 'Timeout'}})
                        except Exception as e:
                            print(f"    ⚠️  Error terminating process: {{e}}")
                            sys.stdout.flush()
                            all_results.append({{**proc_info['config'], 'success': False, 'duration': duration, 'error': f'Timeout + termination error: {{e}}'}})
                        processes.pop(j)
                        break
                    
                    if process.poll() is not None:
                        # Capture output for better error reporting
                        try:
                            stdout, stderr = process.communicate(timeout=1)
                        except:
                            stdout, stderr = "", ""
                        
                        if process.returncode == 0:
                            # Validate that files were actually created
                            output_path = version_dir / "evaluation_results" / proc_info['config']['benchmark']
                            prompt_name = proc_info['config']['prompt_name']
                            
                            # Check for expected output files
                            csv_file = output_path / f"evaluation_results_{{prompt_name}}.csv"
                            txt_files = list(output_path.glob("*.txt"))
                            
                            if csv_file.exists() or txt_files:
                                print(f"✅ [{{run_id}}] SUCCESS ({{duration:.1f}}s) - Files generated")
                                sys.stdout.flush()
                                successful += 1
                                all_results.append({{**proc_info['config'], 'success': True, 'duration': duration}})
                            else:
                                print(f"⚠️  [{{run_id}}] SUCCESS but no output files ({{duration:.1f}}s)")
                                print(f"    Expected: {{csv_file}}")
                                sys.stdout.flush()
                                all_results.append({{**proc_info['config'], 'success': False, 'duration': duration}})
                        else:
                            print(f"❌ [{{run_id}}] FAILED ({{duration:.1f}}s)")
                            if stderr:
                                # Show last part of error message
                                error_lines = stderr.strip().split('\\n')
                                print(f"    Error: {{error_lines[-1] if error_lines else 'Unknown error'}}")
                            sys.stdout.flush()
                            all_results.append({{**proc_info['config'], 'success': False, 'duration': duration}})
                        
                        processes.pop(j)
                        break
                
                if len(processes) >= max_concurrent:
                    # Show status of running processes every 30 seconds
                    if int(time.time()) % 30 == 0:
                        for proc_info in processes:
                            runtime = time.time() - proc_info['start_time']
                            print(f"    [{{proc_info['run_id']}}] Running for {{runtime:.1f}}s...")
                        sys.stdout.flush()
                    time.sleep(1)  # Brief pause to avoid busy waiting
            
            # Small delay between starting processes to avoid overwhelming the API
            time.sleep(0.5)
        
        # Wait for remaining processes in this benchmark to complete
        while processes:
            for i, proc_info in enumerate(processes[:]):
                process = proc_info['process']
                run_id = proc_info['run_id']
                duration = time.time() - proc_info['start_time']
                
                # Check for timeout (4 hours per evaluation for full datasets)
                timeout_limit = 14400  # 4 hours in seconds
                
                if duration > timeout_limit:
                    print(f"⏰ [{{run_id}}] TIMEOUT after {{duration:.1f}}s - Terminating process")
                    sys.stdout.flush()
                    try:
                        process.terminate()
                        time.sleep(2)  # Give it a moment to terminate gracefully
                        if process.poll() is None:
                            process.kill()  # Force kill if terminate didn't work
                        all_results.append({{**proc_info['config'], 'success': False, 'duration': duration, 'error': 'Timeout'}})
                    except Exception as e:
                        print(f"    ⚠️  Error terminating process: {{e}}")
                        sys.stdout.flush()
                        all_results.append({{**proc_info['config'], 'success': False, 'duration': duration, 'error': f'Timeout + termination error: {{e}}'}})
                    processes.pop(i)
                    break
                
                if process.poll() is not None:
                    # Capture output for better error reporting
                    try:
                        stdout, stderr = process.communicate(timeout=1)
                    except:
                        stdout, stderr = "", ""
                    
                    if process.returncode == 0:
                        # Validate that files were actually created
                        output_path = version_dir / "evaluation_results" / proc_info['config']['benchmark']
                        prompt_name = proc_info['config']['prompt_name']
                        
                        # Check for expected output files
                        csv_file = output_path / f"evaluation_results_{{prompt_name}}.csv"
                        txt_files = list(output_path.glob("*.txt"))
                        
                        if csv_file.exists() or txt_files:
                            print(f"✅ [{{run_id}}] SUCCESS ({{duration:.1f}}s) - Files generated")
                            sys.stdout.flush()
                            successful += 1
                            all_results.append({{**proc_info['config'], 'success': True, 'duration': duration}})
                        else:
                            print(f"⚠️  [{{run_id}}] SUCCESS but no output files ({{duration:.1f}}s)")
                            print(f"    Expected: {{csv_file}}")
                            sys.stdout.flush()
                            all_results.append({{**proc_info['config'], 'success': False, 'duration': duration}})
                    else:
                        print(f"❌ [{{run_id}}] FAILED ({{duration:.1f}}s)")
                        if stderr:
                            # Show last part of error message
                            error_lines = stderr.strip().split('\\n')
                            print(f"    Error: {{error_lines[-1] if error_lines else 'Unknown error'}}")
                        sys.stdout.flush()
                        all_results.append({{**proc_info['config'], 'success': False, 'duration': duration}})
                    
                    processes.pop(i)
                    break
            
            # Show status of running processes every 30 seconds
            if len(processes) > 0 and int(time.time()) % 30 == 0:
                for proc_info in processes:
                    runtime = time.time() - proc_info['start_time']
                    print(f"    [{{proc_info['run_id']}}] Running for {{runtime:.1f}}s...")
                sys.stdout.flush()
            
            time.sleep(2)
        
        benchmark_time = time.time() - start_time
        success_rate = successful / len(benchmark_configs) * 100
        
        print(f"{{benchmark}} completed: {{successful}}/{{len(benchmark_configs)}} ({{success_rate:.1f}}%) in {{benchmark_time:.1f}}s")
        sys.stdout.flush()
        
        # Brief pause between benchmarks
        if len(configs_by_benchmark) > 1:
            print(f"⏸️  Pausing 3 seconds before next benchmark...")
            sys.stdout.flush()
            time.sleep(3)
    
    total_successful = sum(1 for r in all_results if r['success'])
    total_time = sum(r['duration'] for r in all_results)
    
    print(f"\\n============================================================")
    print(f"FINAL EVALUATION SUMMARY")
    print(f"============================================================")
    print(f"✅ Successful: {{total_successful}}/{{len(all_configs)}}")
    print(f"❌ Failed: {{len(all_configs) - total_successful}}/{{len(all_configs)}}")
    print(f" Total time: {{total_time:.1f}} seconds")
    print(f"Results saved to: evaluation_results/")
    sys.stdout.flush()
    
    # Show breakdown by benchmark
    for benchmark in configs_by_benchmark.keys():
        benchmark_results = [r for r in all_results if r['benchmark'] == benchmark]
        benchmark_successful = sum(1 for r in benchmark_results if r['success'])
        print(f"  {{benchmark}}: {{benchmark_successful}}/{{len(benchmark_results)}} successful")
    sys.stdout.flush()
    
    # Consolidate all individual CSV files into one master file
    if total_successful > 0:
        consolidate_results(all_configs, version_dir)

def consolidate_results(configs, version_dir):
    """Consolidate individual CSV files into per-benchmark master files."""
    import csv
    from collections import defaultdict
    
    print(f"\\n Consolidating results per benchmark...")
    sys.stdout.flush()
    
    # Group configs by benchmark
    benchmark_configs = defaultdict(list)
    for config in configs:
        benchmark_configs[config["benchmark"]].append(config)
    
    total_benchmarks_processed = 0
    
    for benchmark, benchmark_configs_list in benchmark_configs.items():
        print(f"\\n📊 Processing benchmark: {{benchmark}}")
        sys.stdout.flush()
        
        # Try to get test set size from stats file
        test_set_size = None
        benchmark_dir = version_dir / "evaluation_results" / benchmark
        stat_files = list(benchmark_dir.glob("*.stat"))
        if stat_files:
            try:
                with open(stat_files[0], 'r') as f:
                    for line in f:
                        if line.startswith('test_set_size:'):
                            test_set_size = int(line.split(':')[1].strip())
                            break
            except:
                pass
        
        benchmark_rows = []
        benchmark_stats = {{}}
        
        for config in benchmark_configs_list:
            prompt_name = config["prompt_name"]
            
            # Look for the individual result file
            result_file = version_dir / "evaluation_results" / benchmark / f"evaluation_results_{{prompt_name}}.csv"
            
            if result_file.exists():
                try:
                    with open(result_file, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        
                    if rows:
                        benchmark_rows.extend(rows)
                        print(f"  ✅ Merged {{prompt_name}} ({{len(rows)}} rows)")
                        sys.stdout.flush()
                        
                        # Calculate stats for this variation
                        for row in rows:
                            variation = row.get('prompt_variation', prompt_name)
                            score_str = row.get('score', '0')
                            try:
                                score = float(score_str)
                            except (ValueError, TypeError):
                                score = 0.0
                            
                            if variation not in benchmark_stats:
                                benchmark_stats[variation] = {{'scores': [], 'total': 0.0, 'count': 0}}
                            
                            benchmark_stats[variation]['scores'].append(score)
                            benchmark_stats[variation]['total'] += score
                            benchmark_stats[variation]['count'] += 1
                    else:
                        print(f"  ⚠️  {{prompt_name}} file is empty")
                        sys.stdout.flush()
                except Exception as e:
                    print(f"  ❌ Error reading {{prompt_name}}: {{e}}")
                    sys.stdout.flush()
            else:
                print(f"  ❌ Missing result file: {{result_file}}")
                sys.stdout.flush()
                
                # Try to convert from .txt files if CSV doesn't exist
                print(f"    Attempting to convert from .txt files...")
                sys.stdout.flush()
                benchmark_dir = version_dir / "evaluation_results" / benchmark
                if convert_txt_to_csv(benchmark_dir, prompt_name):
                    # Now try to read the newly created CSV file
                    if result_file.exists():
                        try:
                            with open(result_file, 'r', newline='', encoding='utf-8') as f:
                                reader = csv.DictReader(f)
                                rows = list(reader)
                                
                            if rows:
                                benchmark_rows.extend(rows)
                                print(f"  ✅ Merged converted {{prompt_name}} ({{len(rows)}} rows)")
                                sys.stdout.flush()
                                
                                # Calculate stats for this variation
                                for row in rows:
                                    variation = row.get('prompt_variation', prompt_name)
                                    score_str = row.get('score', '0')
                                    try:
                                        score = float(score_str)
                                    except (ValueError, TypeError):
                                        score = 0.0
                                    
                                    if variation not in benchmark_stats:
                                        benchmark_stats[variation] = {{'scores': [], 'total': 0.0, 'count': 0}}
                                    
                                    benchmark_stats[variation]['scores'].append(score)
                                    benchmark_stats[variation]['total'] += score
                                    benchmark_stats[variation]['count'] += 1
                            else:
                                print(f"    ⚠️  Converted {{prompt_name}} file is empty")
                                sys.stdout.flush()
                        except Exception as e:
                            print(f"    ❌ Error reading converted {{prompt_name}}: {{e}}")
                            sys.stdout.flush()
                else:
                    print(f"    ❌ Could not convert .txt files for {{prompt_name}}")
                    sys.stdout.flush()
        
        if benchmark_rows:
            # Sort rows by score (highest to lowest)
            def get_score(row):
                try:
                    return float(row.get('score', 0))
                except (ValueError, TypeError):
                    return 0.0
            
            sorted_rows = sorted(benchmark_rows, key=get_score, reverse=True)
            
            # Save benchmark-specific consolidated CSV
            benchmark_dir = version_dir / "evaluation_results" / benchmark
            consolidated_file = benchmark_dir / "consolidated_results.csv"
            
            try:
                with open(consolidated_file, 'w', newline='', encoding='utf-8') as f:
                    if sorted_rows:
                        fieldnames = sorted_rows[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(sorted_rows)
                
                print(f"  Saved: {{consolidated_file}}")
                print(f"  Rows: {{len(sorted_rows)}} (sorted by score)")
                print(f"  Variations: {{len(benchmark_stats)}}")
                if test_set_size:
                    print(f"  📊 Test samples: {{test_set_size}}")
                sys.stdout.flush()
                
                # Show performance summary for this benchmark
                if benchmark_stats:
                    if test_set_size:
                        print(f"  🏆 PERFORMANCE RANKING (evaluated on {{test_set_size}} samples):")
                    else:
                        print(f"  🏆 PERFORMANCE RANKING:")
                    sys.stdout.flush()
                    
                    # Sort by average score (descending)
                    sorted_variations = sorted(
                        benchmark_stats.items(), 
                        key=lambda x: x[1]['total'] / x[1]['count'] if x[1]['count'] > 0 else 0, 
                        reverse=True
                    )
                    
                    for i, (variation, stats) in enumerate(sorted_variations, 1):
                        avg_score = stats['total'] / stats['count'] if stats['count'] > 0 else 0.0
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                        # Truncate long variation names to ensure alignment
                        display_name = variation[:35] + "..." if len(variation) > 35 else variation
                        print(f"    {{medal}} {{i:2d}}. {{display_name:<38}} | {{avg_score:5.1f}}")
                        sys.stdout.flush()
                        
                total_benchmarks_processed += 1
                        
            except Exception as e:
                print(f"  ❌ Error writing consolidated file: {{e}}")
                sys.stdout.flush()
        else:
            print(f"  ❌ No valid result files found for {{benchmark}}")
            sys.stdout.flush()
    
    print(f"\\n✅ Consolidated results for {{total_benchmarks_processed}} benchmarks")
    print(f"Each benchmark has its own consolidated_results.csv file")
    sys.stdout.flush()
    
    # Finalize and consolidate subprocess logging
    try:
        if 'logging_manager' in locals() and logging_manager:
            from experimental.one_shot_optimization.llm_logger import finalize_logging
            print("\\n📊 Finalizing subprocess LLM logging...")
            finalize_logging()
            print("✅ Subprocess LLM logging finalized")
    except Exception as e:
        print(f"⚠️  Error finalizing subprocess logging: {{e}}")

if __name__ == "__main__":
    main()
'''
        
        script_path = version_dir / "run_evaluation.py"
        script_path.write_text(evaluation_script)
        script_path.chmod(0o755)  # Make executable
        
        return script_path
    
    def _print_pipeline_summary(self, results: Dict[str, Any]):
        """Print a summary of the pipeline execution."""
        
        print(f"\nPipeline Summary:")
        print(f"Version: {results['version']}")
        
        for stage_name, stage_results in results["stages"].items():
            if stage_results.get("skipped"):
                print(f"  {stage_name.title()}: ⏭️  SKIPPED")
            elif stage_results.get("success"):
                if stage_name == "evaluation" and "successful_evaluations" in stage_results:
                    successful = stage_results.get("successful_evaluations", 0)
                    total = stage_results.get("total_evaluations", 0)
                    print(f"  {stage_name.title()}: ✅ SUCCESS ({successful}/{total} evaluations)")
                else:
                    print(f"  {stage_name.title()}: ✅ SUCCESS")
            else:
                print(f"  {stage_name.title()}: ❌ FAILED")
        
        # Print optimization summary if available
        opt_summary = results["stages"].get("optimization", {}).get("summary")
        if opt_summary:
            print(f"\n Optimization Summary:")
            print(f"  Benchmarks: {opt_summary['total_benchmarks']}")
            print(f"  Total variations: {opt_summary['total_variations']}")
            print(f"  Strategies used: {opt_summary['strategies_used']}")
        
        # Print file locations
        saving_results = results["stages"].get("saving", {})
        if saving_results.get("version_dir"):
            print(f"\n Output Locations:")
            print(f"  Prompts: {saving_results['version_dir']}")
            if saving_results.get("evaluation_config_path"):
                print(f"  Config: {saving_results['evaluation_config_path']}")
    
    @classmethod
    def create_from_config_file(cls, config_file: str) -> 'EvaluationOrchestrator':
        """Create orchestrator from configuration file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config = PipelineConfig.from_file(config_path)
        return cls(config)
    
    def quick_test(self, benchmark: str = "HeartDisease") -> Dict[str, Any]:
        """Run a quick test with a single benchmark."""
        
        # Create quick test configuration
        quick_config = PipelineConfig(
            benchmarks=[benchmark],
            optimization=self.config.optimization,
            version=f"quick_test_{benchmark}_{self.config.version}"
        )
        
        # Update configuration
        original_config = self.config
        self.config = quick_config
        
        try:
            return self.run_full_pipeline(
                extract_prompts=True,
                optimize_prompts=True,
                save_prompts=True,
                run_evaluation=False,  # Skip evaluation for quick test
                selected_benchmarks=[benchmark]
            )
        finally:
            # Restore original configuration
            self.config = original_config 