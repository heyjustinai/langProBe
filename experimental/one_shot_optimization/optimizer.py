"""
Meta-prompt optimization engine.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from .config import BenchmarkConfig, PromptConfig, OptimizationConfig

# Embedded meta-prompt strategy templates
DEFAULT_META_TEMPLATES = {
    "openai_v1_research_paper_generated_with_few_data": """
You are an expert prompt optimization specialist. Your task is to improve the following system prompt to make it more effective.

ORIGINAL SYSTEM PROMPT:
```
{original_system_prompt}
```

Create an enhanced version that:
1. Establishes clear role definition and expertise
2. Provides structured approach to the task
3. Includes step-by-step reasoning guidance
4. Emphasizes accuracy and thoroughness
5. Maintains clarity and actionability

Return ONLY the improved system prompt text, without any explanations.
""",

    "openai_v2_research_paper_generated_with_few_data": """
You are a prompt engineering expert specializing in analytical tasks. Optimize the following system prompt:

ORIGINAL SYSTEM PROMPT:
```
{original_system_prompt}
```

Create an improved version that:
1. Positions the AI as a capable analytical expert
2. Provides clear methodology for approaching the task
3. Includes guidance for handling complex information
4. Emphasizes evidence-based reasoning
5. Ensures comprehensive yet focused responses

Return ONLY the optimized system prompt.
""",

    "gemini2.5_research_paper_generated_with_few_data": """
You are a prompt optimization specialist. Enhance the following system prompt for better performance:

ORIGINAL SYSTEM PROMPT:
```
{original_system_prompt}
```

Create an enhanced version that:
1. Establishes expert-level capability and authority
2. Provides systematic approach to problem-solving
3. Includes multi-layered analysis guidance
4. Emphasizes reasoning transparency
5. Balances depth with clarity

Return ONLY the enhanced system prompt.
""",

    "metaai_research_paper_generated_with_few_data": """
You are an expert in prompt engineering. Improve the following system prompt for optimal performance:

ORIGINAL SYSTEM PROMPT:
```
{original_system_prompt}
```

Create a better version that:
1. Defines clear expert identity and capabilities
2. Provides structured analytical framework
3. Includes comprehensive reasoning guidance
4. Emphasizes accuracy and logical consistency
5. Ensures clear communication of insights

Return ONLY the improved system prompt.
""",

    "llama_405b": """
You are a prompt optimization expert. Enhance this system prompt for better task performance:

ORIGINAL SYSTEM PROMPT:
```
{original_system_prompt}
```

Create an optimized version that:
1. Establishes strong expert-level authority
2. Provides advanced analytical methodology
3. Includes sophisticated reasoning techniques
4. Emphasizes systematic verification
5. Ensures detailed yet clear communication

Return ONLY the enhanced system prompt.
""",

    "self_written": """
You are a prompt engineering specialist. Improve the following system prompt to be more effective:

ORIGINAL SYSTEM PROMPT:
```
{original_system_prompt}
```

Create a better version that:
1. Clearly defines the AI's role and expertise
2. Provides structured approach to the task
3. Includes clear step-by-step guidance
4. Emphasizes accuracy and completeness
5. Maintains focus on core objectives

Return ONLY the improved system prompt.
""",

    "self_written_2": """
You are an expert in creating effective system prompts. Optimize the following prompt:

ORIGINAL SYSTEM PROMPT:
```
{original_system_prompt}
```

Create an enhanced version that:
1. Establishes clear expert identity
2. Provides systematic methodology
3. Includes detailed reasoning guidance
4. Emphasizes quality and accuracy
5. Ensures practical applicability

Return ONLY the optimized system prompt.
"""
}


class MetaPromptOptimizer:
    """Optimizes prompts using meta-prompt strategies."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.available_strategies = list(DEFAULT_META_TEMPLATES.keys())
        
        # Initialize OpenAI client for optimization
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=config.base_url,
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
        except ImportError:
            print("Warning: OpenAI library not available. Meta-prompt optimization will use fallback.")
            self.client = None
        
    def _get_available_strategies(self) -> List[str]:
        """Get list of available meta-prompt strategies."""
        return self.available_strategies
    
    def optimize_benchmark_prompts(
        self, 
        benchmark_configs: Dict[str, BenchmarkConfig],
        selected_strategies: Optional[List[str]] = None
    ) -> Dict[str, BenchmarkConfig]:
        """
        Optimize prompts for multiple benchmarks using meta-prompt strategies.
        
        Args:
            benchmark_configs: Dictionary of benchmark configurations
            selected_strategies: List of meta-prompt strategies to use
            
        Returns:
            Updated benchmark configurations with optimized prompt variations
        """
        
        if selected_strategies is None:
            selected_strategies = self.config.strategies
            
        print(f"🚀 Starting meta-prompt optimization")
        print(f"📊 Benchmarks: {list(benchmark_configs.keys())}")
        print(f"🎯 Strategies: {selected_strategies}")
        
        optimized_configs = {}
        
        for benchmark_name, config in benchmark_configs.items():
            print(f"\n📝 Optimizing prompts for {benchmark_name}...")
            
            optimized_config = self._optimize_single_benchmark(
                config, 
                selected_strategies
            )
            
            optimized_configs[benchmark_name] = optimized_config
            
            print(f"  ✅ Generated {len(optimized_config.prompt_variations)} prompt variations")
        
        return optimized_configs
    
    def _optimize_single_benchmark(
        self, 
        benchmark_config: BenchmarkConfig,
        strategies: List[str]
    ) -> BenchmarkConfig:
        """Optimize prompts for a single benchmark using parallel strategy execution."""
        
        # Start with existing prompt variations
        optimized_config = BenchmarkConfig(
            name=benchmark_config.name,
            base_prompt=benchmark_config.base_prompt,
            signature=benchmark_config.signature,
            task_description=benchmark_config.task_description,
            prompt_variations=benchmark_config.prompt_variations.copy()
        )
        
        # Thread-safe lock for updating the config
        config_lock = threading.Lock()
        
        # Generate optimized variations for each strategy in parallel
        print(f"  🚀 Running {len(strategies)} strategies in parallel (max_concurrent: {self.config.max_concurrent})")
        
        def optimize_strategy(strategy: str):
            """Optimize a single strategy and return results."""
            start_time = time.time()
            try:
                optimized_prompt = self._apply_meta_prompt_strategy(
                    benchmark_config.base_prompt,
                    benchmark_config.signature,
                    strategy
                )
                
                if optimized_prompt and len(optimized_prompt.strip()) > 50:
                    variation = PromptConfig(
                        name=f"meta_{strategy}",
                        description=f"Meta-optimized prompt using {strategy} strategy",
                        instructions=optimized_prompt.strip(),
                        task_description=benchmark_config.task_description,
                        signature=benchmark_config.signature,
                        meta_strategy=strategy
                    )
                    duration = time.time() - start_time
                    return strategy, variation, None
                else:
                    duration = time.time() - start_time
                    return strategy, None, "Failed to generate valid optimized prompt"
                    
            except Exception as e:
                duration = time.time() - start_time
                return strategy, None, str(e)
        
        # Execute strategies in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            # Submit all strategy optimization tasks
            future_to_strategy = {
                executor.submit(optimize_strategy, strategy): strategy 
                for strategy in strategies
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                
                try:
                    strategy_name, variation, error = future.result()
                    
                    if variation:
                        with config_lock:
                            optimized_config.add_variation(variation)
                        print(f"  ✅ [{strategy_name}] Generated optimized prompt ({len(variation.instructions)} chars)")
                    else:
                        print(f"  ❌ [{strategy_name}] {error}")
                        
                except Exception as e:
                    print(f"  ❌ [{strategy}] Unexpected error: {e}")
        
        return optimized_config
    
    def _apply_meta_prompt_strategy(
        self,
        original_prompt: str,
        signature: str,
        strategy: str
    ) -> Optional[str]:
        """Apply a specific meta-prompt strategy to optimize a prompt."""
        
        if strategy not in DEFAULT_META_TEMPLATES:
            print(f"Warning: Strategy '{strategy}' not found in available templates")
            return self._fallback_optimization(original_prompt, strategy)
        
        meta_template = DEFAULT_META_TEMPLATES[strategy]
        
        # Fill in the template with the original prompt
        meta_prompt = meta_template.format(
            original_system_prompt=original_prompt
        )
        
        # Use LLM to optimize the prompt
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "user", "content": meta_prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=2000
                )
                
                optimized_prompt = response.choices[0].message.content.strip()
                return optimized_prompt
                
            except Exception as e:
                print(f"Error during LLM optimization: {e}")
                return self._fallback_optimization(original_prompt, strategy)
        else:
            # Fallback if no LLM client available
            return self._fallback_optimization(original_prompt, strategy)
    
    def _fallback_optimization(self, original_prompt: str, strategy: str) -> str:
        """Fallback optimization when LLM optimization is not available."""
        
        # Simple template-based improvements based on strategy
        strategy_improvements = {
            "openai_v1_research_paper_generated_with_few_data": """
You are an expert assistant with deep analytical capabilities. Your approach:
1. Carefully analyze all provided information and context
2. Apply systematic reasoning to break down complex problems
3. Provide clear, well-structured responses with supporting evidence
4. Focus on accuracy and precision in your analysis
5. Consider multiple perspectives when evaluating information

Original task: """,
            
            "openai_v2_research_paper_generated_with_few_data": """
You are a highly capable analytical expert. Your methodology:
1. Thoroughly examine all available data and context
2. Apply rigorous analytical frameworks to understand patterns
3. Synthesize information to generate comprehensive insights
4. Maintain objectivity while providing detailed explanations
5. Ensure conclusions are well-supported by evidence

Task context: """,
            
            "gemini2.5_research_paper_generated_with_few_data": """
You are an advanced reasoning specialist with expertise in complex analysis. Your process:
1. Systematically evaluate all input data and contextual factors
2. Apply multi-layered analysis to identify key patterns and relationships
3. Generate well-reasoned conclusions based on comprehensive evaluation
4. Provide detailed explanations that demonstrate your reasoning process
5. Balance depth of analysis with clarity of communication

Primary objective: """,
            
            "metaai_research_paper_generated_with_few_data": """
You are a sophisticated analytical assistant designed for complex reasoning tasks. Your approach:
1. Conduct thorough analysis of all available information
2. Apply advanced reasoning techniques to identify insights
3. Generate comprehensive responses that address all aspects of the task
4. Maintain high standards of accuracy and logical consistency
5. Provide clear explanations that demonstrate your analytical process

Core task: """,
            
            "llama_405b": """
You are an expert-level assistant with advanced analytical capabilities. Your methodology:
1. Perform comprehensive analysis of all provided information
2. Apply sophisticated reasoning to understand complex relationships
3. Generate detailed, well-structured responses with clear logic
4. Ensure accuracy through systematic verification of conclusions
5. Communicate insights effectively with appropriate level of detail

Assignment: """,
            
            "self_written": """
You are a domain expert with specialized knowledge relevant to this task.
Your approach:
1. Apply your expertise to understand the nuanced requirements
2. Consider all relevant factors and contextual information
3. Provide clear, actionable insights based on your knowledge
4. Maintain accuracy while being comprehensive in your analysis
5. Ensure your response directly addresses the specific needs

Task: """,
            
            "self_written_2": """
You are an analytical specialist focused on delivering high-quality results.
Your process:
1. Carefully examine all aspects of the given task
2. Apply systematic analysis to identify key components
3. Generate well-reasoned responses with clear supporting logic
4. Ensure completeness while maintaining focus on core objectives
5. Provide insights that are both accurate and practically useful

Objective: """
        }
        
        improvement = strategy_improvements.get(strategy, f"""
You are an expert assistant specialized in this type of task.
Your approach:
1. Carefully analyze the requirements and context
2. Apply systematic reasoning to understand the problem
3. Generate well-structured, accurate responses
4. Ensure your analysis is thorough and well-supported

Task: """)
        
        return improvement + original_prompt
    
    def validate_optimized_prompts(
        self, 
        optimized_configs: Dict[str, BenchmarkConfig]
    ) -> Dict[str, Dict[str, bool]]:
        """
        Validate that optimized prompts meet basic quality criteria.
        
        Returns:
            Dictionary mapping benchmark -> variation -> validation_passed
        """
        validation_results = {}
        
        for benchmark_name, config in optimized_configs.items():
            validation_results[benchmark_name] = {}
            
            for variation in config.prompt_variations:
                is_valid = self._validate_single_prompt(variation)
                validation_results[benchmark_name][variation.name] = is_valid
                
                if not is_valid:
                    print(f"⚠️  Validation failed for {benchmark_name}.{variation.name}")
        
        return validation_results
    
    def _validate_single_prompt(self, prompt_config: PromptConfig) -> bool:
        """Validate a single prompt configuration."""
        
        # Basic validation criteria
        if not prompt_config.instructions or len(prompt_config.instructions.strip()) < 50:
            return False
            
        # Check for common prompt engineering elements
        instructions_lower = prompt_config.instructions.lower()
        has_role_definition = any(term in instructions_lower for term in 
                                ['you are', 'act as', 'your role', 'as an expert'])
        
        has_task_clarity = any(term in instructions_lower for term in 
                             ['task', 'goal', 'objective', 'purpose'])
        
        # At least one of these should be present for a good prompt
        return has_role_definition or has_task_clarity or len(prompt_config.instructions) > 200
    
    def get_optimization_summary(
        self, 
        optimized_configs: Dict[str, BenchmarkConfig]
    ) -> Dict[str, Any]:
        """Generate a summary of the optimization process."""
        
        total_variations = 0
        strategies_used = set()
        
        for config in optimized_configs.values():
            total_variations += len(config.prompt_variations)
            for variation in config.prompt_variations:
                if variation.meta_strategy:
                    strategies_used.add(variation.meta_strategy)
                else:
                    strategies_used.add("extracted")
        
        return {
            "total_benchmarks": len(optimized_configs),
            "total_variations": total_variations,
            "strategies_used": list(strategies_used),
            "avg_variations_per_benchmark": total_variations / len(optimized_configs) if optimized_configs else 0
        } 