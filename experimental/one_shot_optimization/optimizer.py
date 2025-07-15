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
import json

from .config import BenchmarkConfig, PromptConfig, OptimizationConfig
from .llm_logger import get_logging_manager

# Embedded meta-prompt strategy templates
DEFAULT_META_TEMPLATES = {
    "test_3":"""
    generate a 200 character prompt that is a good prompt for the task of generating a 200 character prompt
    """, 
    "test_0": """
    add chain of thought reasoning to the following prompt:
    """,
    "test_1": """
You are a prompt engineering expert. Your task is to improve the following prompt by adding chain-of-thought reasoning.

Original prompt: {original_system_prompt}

Create a new and improved prompt that:
1. Maintains the same core task and objective
2. Adds step-by-step reasoning structure  
3. Provides clear instructions for the reasoning process
4. Ensures the final output format matches the original requirements

Generate the improved prompt:
    """,
 "openai_v2_research_paper_generated_with_few_data": """
        Based on the original prompt: {original_system_prompt}
        Generate a new prompt based on this format:
        
        <You are {{ROLE_DESCRIPTION}}. Your goal is to {{PARAPHRASE_BRIEF_INSTRUCTION}}. You will operate as a cooperative, ethical, and insightful assistant with access to extensive general knowledge. You are capable of structured, step-by-step reasoning and can decompose complex problems into smaller sub-tasks. Here is the input you’ll work with:

        <{{INPUT_TAG}}>
        {{INPUT_VARIABLE}}
        </{{INPUT_TAG}}>

        To accomplish this, follow these steps:

            1. Interpret the task and clarify any assumptions internally. 
            2. Think step by step to break down the task into sub-components, following the structure:
                • "Are follow-up questions needed here? [Yes/No]"
                • "Follow-up: [Articulate the sub-question]"
                • "Intermediate answer: [Answer the sub-question]"
                • Repeat until you have the information to solve the problem.
            3. If information is missing, indicate what needs to be retrieved externally.
            4. Use your internal knowledge base to answer. If the question seems to require very recent or obscure knowledge, acknowledge possible uncertainty and suggest the user verify the answer independently.
            5. Use analogies, examples, or thought experiments to make difficult concepts clearer, where appropriate.
            6. Once reasoning is complete, synthesize the answer concisely.

        Your output must follow these guidelines:

            • Do not include internal thought steps or scratch work outside of <{{OUTPUT_TAG}}> unless asked.
            • If the task involves multi-step reasoning, ensure your final answer reflects the cumulative logic.
            • Never list multiple options unless explicitly asked—be decisive.
            • If asked to generate content (e.g., code, JSON, or poetry), format it correctly and offer a brief follow-up like: "Would you like a breakdown?"
            • Use markdown formatting for any code. Use string operations to explicitly count or manipulate tokens when needed.
            • If you believe you hallucinated or guessed, indicate this clearly without citing fake sources.
            • Always communicate in the user's language. Keep tone warm, focused, and clear.
            • Avoid any content that violates ethical safety (e.g. illegal instructions, disordered advice, or impersonation).

        Format your final answer inside <{{OUTPUT_TAG}}> tags and do not include any of your internal reasoning.

        <{{OUTPUT_TAG}}>
        …your response…
        </{{OUTPUT_TAG}}>

        """,
        "openai_v1_research_paper_generated_with_few_data": """
        Based on the original prompt: {original_system_prompt}
        Generate a new prompt based on this format:
        <You are {{ROLE_DESCRIPTION}}. Your goal is to {{PARAPHRASE_BRIEF_INSTRUCTION}}.

        Here is the input you’ll work with:
        <{{INPUT_TAG}}>
        {{INPUT_VARIABLE}}
        </{{INPUT_TAG}}>

        We share a common objective: {{SHARED_GOAL}}.  
        Follow all legal, ethical, and safety constraints. Decline or safe-complete any request that violates them.

        To accomplish this, follow these steps:

            1. **Initial Reasoning Trigger (Zero-shot Chain-of-Thought)**  
            • Internally think step-by-step to understand the task, decompose it, and outline a solution plan before generating any answer. Do **not** reveal this reasoning.  
            • If the task is multi-hop or ambiguous, activate *self-ask*: formulate clarifying sub-questions and reason through each.

            2. **External Knowledge Integration (RAG-ready)**  
            • For facts you are unsure of, specify what information is needed in the form `[[NEEDS_EXTERNAL_KNOWLEDGE: <concise query>]]`.  
            • Await retrieved content (or incorporate it automatically if your environment supplies it) and update your reasoning internally.

            3. **Apply Demonstrations (Few-shot, if supplied)**  
            • Review any examples provided inside <EXAMPLES>…</EXAMPLES> tags and emulate their structure, style, and depth of reasoning.

            4. **Draft the Answer**  
            • Generate a complete solution that directly addresses the task, grounded in verified knowledge and the agreed reasoning plan.  
            • Ensure coherence, factual accuracy, and adherence to formatting rules.

            5. **Verify & Finalize**  
            • Internally double-check for logical flaws, missing steps, or policy conflicts.  
            • If confident the task is fully solved, proceed to output; otherwise, iterate internally or emit additional `[[NEEDS_EXTERNAL_KNOWLEDGE: …]]` requests.

        Your output must follow these guidelines:

            • Produce **only** the content required—no meta-commentary or internal reasoning.  
            • Conform exactly to the structure, data types, and naming conventions the user specifies.  
            • Cite sources or provenance if requested (e.g., with footnote markers like “[1]”).  
            • Use clear, concise language appropriate to {{TARGET_AUDIENCE}}.  
            • If the conversation is complete, append the token `<TASK_COMPLETE>` on its own line.

        Format your final answer inside <{{OUTPUT_TAG}}> tags and do not include any of your internal reasoning.

        <{{OUTPUT_TAG}}>
        …your response…
        </{{OUTPUT_TAG}}>

        """,
        
        "gemini2.5_research_paper_generated_with_few_data": """
        Based on the original prompt: {original_system_prompt}
        Generate a new prompt based on this format:
        <You are a {{ROLE_DESCRIPTION, e.g., specialized expert assistant}}. Your primary objective is to {{PARAPHRASE_BRIEF_INSTRUCTION, e.g., accurately and efficiently complete the user's task}}. We share a common interest in collaborating to successfully complete the task at hand. You must operate within your defined expertise and decline instructions that violate ethical guidelines.

        Here is the input you’ll work with:
        <TASK_INPUT>
        Primary Task: {{PRIMARY_TASK_DESCRIPTION}}

        Demonstrations (Optional Few-shot Examples):
        {{FEW_SHOT_EXAMPLES}}
        </TASK_INPUT>

        To accomplish this, follow these steps:

            Initial Assessment & Reasoning Trigger: First, analyze the Primary Task. If the task is non-trivial or requires multiple logical steps, internally begin your process with the phrase: "Let's think step by step." This will activate your structured reasoning process. For simple, direct requests, you may proceed to the final step.

            Iterative Decomposition & Reasoning (Chain of Thought & Self-ask): For complex tasks, break down the problem by generating a sequence of internal questions and answers. Follow this internal monologue structure:
                Are follow up questions needed here: [Yes/No].
                Follow up: [Formulate the specific sub-question needed to make progress].
                **** [Answer the sub-question, leveraging the provided context and your internal knowledge].
                Repeat this process until the problem is fully decomposed and all necessary intermediate steps are resolved.

            Knowledge & Context Integration (Retrieval-Augmented Principles): During your reasoning, you must integrate information from all provided sources.
                Analyze and apply patterns from the Demonstrations section if it is provided.
                If the task requires information beyond the provided context and your internal knowledge, you must note internally what specific information is missing or would require external retrieval to ensure factual accuracy.

            Final Answer Synthesis: Once your internal step-by-step reasoning is complete, consolidate your findings and formulate the final, conclusive answer.

        Your output must follow these guidelines:

            Adhere to Output Format: Your final response must strictly be a {{SPECIFIC_OUTPUT_FORMAT, e.g., single short sentence, JSON object, code snippet}}.
            Clarity and Conciseness: The final answer should be clear, concise, and directly address the Primary Task.
            Handle Ambiguity: If the Primary Task is ambiguous or lacks critical information, you must ask for clarification before proceeding. Do not make assumptions.
            Confine to Role: All responses must align with the persona and constraints defined in your role description.
            Exclusion of Internal Reasoning: Your final output must not include any of the internal "Let's think step by step" monologue, "Follow up" questions, or "Intermediate answers." These are for your internal process only.

        Format your final answer inside <**FINAL_ANSWER**> tags and do not include any of your internal reasoning.
        <FINAL_ANSWER>
        …your response…
        </FINAL_ANSWER>
        """,
        
        "metaai_research_paper_generated_with_few_data": """
        Based on the original prompt: {original_system_prompt}
        Generate a new prompt based on this format:
        You are a highly capable problem-solving assistant. Your goal is to provide accurate, detailed, and factual solutions to complex tasks. Here is the input you'll work with:
        <input_data>
        {{INPUT_VARIABLE}}
        </input_data>

        To accomplish this, follow these steps:
        1. Analyze the task requirements and identify any necessary sub-questions or intermediate steps.
        2. Perform step-by-step reasoning to arrive at a solution, using a chain-of-thought approach.
        3. If necessary, indicate the need for external knowledge or information and provide a clear description of what's required.
        4. Once you've gathered sufficient information, synthesize your findings to arrive at a final solution.

        Your output must follow these guidelines:
        1. Clearly label each step of your reasoning process.
        2. Provide explicit explanations for any assumptions or inferences made.
        3. Format your final answer in a clear and concise manner, using relevant notation or structure as needed.
        4. If external knowledge was used, provide provenance or credits where possible.

        Format your final answer inside <solution> tags and do not include any of your internal reasoning.
        <solution>
        …your response…
        </solution>

        {{#if GOLDEN_EXAMPLES}}
        Additionally, you are provided with golden examples that represent high-quality input-output pairs for this task. Use these examples to understand the task structure, input format, expected output format, and reasoning patterns. These examples should inform your understanding of the task and help you create a more effective and well-structured prompt, but do NOT include these examples directly in your generated prompt:

        <golden_examples>
        {golden_examples}
        </golden_examples>

        Based on these examples, ensure your generated prompt:
        - Captures the correct input/output relationship and format
        - Uses appropriate terminology and field names
        - Reflects the complexity and reasoning required for the task
        - Maintains consistency with the task's expectations
        - Provides clear guidance without including actual example data

        {{/if}}

        When generating your response, follow this structure:
        - Are follow-up questions needed here: [Yes/No]
        - Follow-up: [Formulate a necessary sub-question based on the current problem]
        - Intermediate answer: [Provide the answer to the sub-question]
        - (Repeat "Follow-up" and "Intermediate answer" until sufficient information is gathered)
        - So the final answer is: [Provide the concise final answer]

        Once you have provided the complete solution and believe the task is finished, respond with only the phrase: <TASK_COMPLETE>.
        """,
        "self_written":
            """
            Based on the original prompt: {original_system_prompt}
            Generated a new and improved prompt and add Chain of thought based on this format:

            (Prompt Template Syntax)
            <You are {{ROLE_DESCRIPTION}}. Your goal is to {{PARAPHRASE_BRIEF_INSTRUCTION}}. Here is the task: [Detailed description of the specific task]. Here is the input you’ll work with:

            <{{INPUT_TAG}}>
            {{INPUT_VARIABLE}}
            </{{INPUT_TAG}}>

            To accomplish this, follow these steps:
            1. {{STEP_1_DESCRIPTION}}
            2. {{STEP_2_DESCRIPTION}}
            3. {{STEP_3_DESCRIPTION}}
            …  

            Your output must follow these guidelines:
            - {{GUIDELINE_1}}
            - {{GUIDELINE_2}}
            - {{GUIDELINE_3}}
            …  

            Before providing the final answer, let's think step by step to arrive at the solution. Format your final answer inside `<{{OUTPUT_TAG}}>` tags and do **not** include any of your internal reasoning.

            {{#if GOLDEN_EXAMPLES}}
                    Additionally, you are provided with golden examples that represent high-quality input-output pairs for this task. Use these examples to understand the task structure, input format, expected output format, and reasoning patterns. These examples should inform your understanding of the task and help you create a more effective and well-structured prompt, but do NOT include these examples directly in your generated prompt:

                    <golden_examples>
                    {golden_examples}
                    </golden_examples>

                    Based on these examples, ensure your generated prompt:
                    - Captures the correct input/output relationship and format
                    - Uses appropriate terminology and field names
                    - Reflects the complexity and reasoning required for the task
                    - Maintains consistency with the task's expectations
                    - Provides clear guidance without including actual example data

            {{/if}}


            <{{OUTPUT_TAG}}>
            …your response…
            </{{OUTPUT_TAG}}>
            >


            1. Clearly define the AI's identity, specialized expertise, and fundamental rules it must follow. ROLE_DESCRIPTION, PARAPHRASE_BRIEF_INSTRUCTION
            2. For tasks requiring explicit decomposition and iterative problem-solving, encourage a self-ask approach. This structure can be part of the expected output format or a guiding principle within the prompt



            """,
            "self_written_2": """
        Based on the original prompt: {original_system_prompt}
        Generate a new prompt based on this format:

(Prompt Template Syntax)
<You are {{ROLE_DESCRIPTION}}. Your goal is to {{PARAPHRASE_BRIEF_INSTRUCTION}}. Here is the task: [Detailed description of the specific task]. Here is the input you’ll work with:

<{{INPUT_TAG}}>
{{INPUT_VARIABLE}}
</{{INPUT_TAG}}>

To accomplish this, follow these steps:
1. {{STEP_1_DESCRIPTION}}
2. {{STEP_2_DESCRIPTION}}
3. {{STEP_3_DESCRIPTION}}
…  

Your output must follow these guidelines:
- {{GUIDELINE_1}}
- {{GUIDELINE_2}}
- {{GUIDELINE_3}}
…  

Before providing the final answer, let's think step by step to arrive at the solution. Format your final answer inside `<{{OUTPUT_TAG}}>` tags and do **not** include any of your internal reasoning.

{{#if GOLDEN_EXAMPLES}}
        Additionally, you are provided with golden examples that represent high-quality input-output pairs for this task. Use these examples to understand the task structure, input format, expected output format, and reasoning patterns. These examples should inform your understanding of the task and help you create a more effective and well-structured prompt, but do NOT include these examples directly in your generated prompt:

        <golden_examples>
        {golden_examples}
        </golden_examples>

        Based on these examples, ensure your generated prompt:
        - Captures the correct input/output relationship and format
        - Uses appropriate terminology and field names
        - Reflects the complexity and reasoning required for the task
        - Maintains consistency with the task's expectations
        - Provides clear guidance without including actual example data

{{/if}}


<{{OUTPUT_TAG}}>
…your response…
</{{OUTPUT_TAG}}>
>


1. Clearly define the AI's identity, specialized expertise, and fundamental rules it must follow. ROLE_DESCRIPTION, PARAPHRASE_BRIEF_INSTRUCTION
2. For tasks requiring explicit decomposition and iterative problem-solving, encourage a self-ask approach. This structure can be part of the expected output format or a guiding principle within the prompt
            """,
            "llama_405b": """
            Based on the original prompt: {original_system_prompt}
            Generate a new prompt based on this format:
            You are a highly advanced language model, capable of complex reasoning and problem-solving. Your goal is to provide accurate and informative responses to the given input, following a structured approach.
            Here is the input you'll work with:
            <INPUT>
            {{USER_INPUT}}
            </INPUT>
            To accomplish this, follow these steps:
            Understand the Task: Carefully read and comprehend the input, identifying the key elements and requirements.
            Break Down the Problem: Decompose the task into smaller, manageable sub-problems, using a chain-of-thought (CoT) approach.
            Gather Relevant Information: If necessary, use external knowledge sources to gather relevant information and provide provenance for your answers.
            Apply Reasoning and Logic: Apply step-by-step reasoning and logical thinking to arrive at a solution, using self-ask prompting to guide your thought process.
            Evaluate and Refine: Evaluate your solution, refining it as needed to ensure accuracy and completeness.
            Your output must follow these guidelines:
            Clear and Concise: Provide clear and concise responses, avoiding ambiguity and jargon.
            Well-Structured: Use a well-structured format for your response, including headings and bullet points as needed.
            Accurate and Informative: Ensure that your response is accurate and informative, providing relevant details and examples.
            Format your final answer inside <OUTPUT> tags and do not include any of your internal reasoning.
            <OUTPUT>
            ...your response...
            </OUTPUT>

            {{#if GOLDEN_EXAMPLES}}
            Additionally, you are provided with golden examples that represent high-quality input-output pairs for this task. Use these examples to understand the task structure, input format, expected output format, and reasoning patterns. These examples should inform your understanding of the task and help you create a more effective and well-structured prompt, but do NOT include these examples directly in your generated prompt:

            <golden_examples>
            {golden_examples}
            </golden_examples>

            Based on these examples, ensure your generated prompt:
            - Captures the correct input/output relationship and format
            - Uses appropriate terminology and field names
            - Reflects the complexity and reasoning required for the task
            - Maintains consistency with the task's expectations
            - Provides clear guidance without including actual example data

            {{/if}}

            Chain of Thought (CoT) Template
            To facilitate CoT, use the following template:
            Step 1: Identify the key elements and requirements of the task.
            Sub-question: What are the essential components of the task?
            Answer: [Provide a brief answer]
            Step 2: Break down the problem into smaller sub-problems.
            Sub-question: How can I decompose the task into manageable parts?
            Answer: [Provide a brief answer]
            Step 3: Gather relevant information and apply reasoning and logic.
            Sub-question: What information do I need to solve the task, and how can I apply logical thinking?
            Answer: [Provide a brief answer]
            Step 4: Evaluate and refine the solution.
            Sub-question: Is my solution accurate and complete, and how can I refine it?
            Answer: [Provide a brief answer]
            By following this structured approach, you will be able to provide accurate and informative responses to the given input, demonstrating your ability to think critically and solve complex problems.
            """
}


class GoldenExampleExtractor:
    """Extracts and manages golden examples for benchmarks."""
    
    def __init__(self, golden_examples_dir: str = "configs/golden_examples"):
        # Get the project root directory (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        self.golden_examples_dir = project_root / golden_examples_dir
        self.golden_examples_dir.mkdir(parents=True, exist_ok=True)
        
        # Key benchmarks to extract golden examples for
        self.key_benchmarks = ["HeartDisease", "hover", "judgebench"]
    
    def extract_and_save_golden_examples(self, num_examples: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Extract golden examples from key benchmarks and save to JSON files."""
        print("🔍 Extracting golden examples from key benchmarks...")
        
        all_golden_examples = {}
        
        for benchmark_name in self.key_benchmarks:
            print(f"  Extracting from {benchmark_name}...")
            
            try:
                golden_examples = self._extract_from_benchmark(benchmark_name, num_examples)
                
                if golden_examples:
                    # Save to JSON file
                    json_file = self.golden_examples_dir / f"{benchmark_name}_golden_examples.json"
                    with open(json_file, 'w') as f:
                        json.dump(golden_examples, f, indent=2)
                    
                    all_golden_examples[benchmark_name] = golden_examples
                    print(f"    ✅ Saved {len(golden_examples)} examples to {json_file}")
                else:
                    print(f"    ❌ No examples extracted for {benchmark_name}")
                    
            except Exception as e:
                print(f"    ❌ Failed to extract from {benchmark_name}: {e}")
                
        return all_golden_examples
    
    def _extract_from_benchmark(self, benchmark_name: str, num_examples: int) -> List[Dict[str, Any]]:
        """Extract golden examples from a specific benchmark."""
        try:
            # Import the benchmark module
            benchmark_module = importlib.import_module(f"langProBe.{benchmark_name}")
            
            # Find the benchmark class
            benchmark_class = None
            for item_name in dir(benchmark_module):
                item = getattr(benchmark_module, item_name)
                if (isinstance(item, type) and 
                    hasattr(item, 'init_dataset') and 
                    item_name.endswith('Bench')):
                    benchmark_class = item
                    break
            
            if not benchmark_class:
                return []
            
            # Create minimal instance
            benchmark_instance = benchmark_class(dataset_mode="test")
            
            # Extract golden examples
            golden_examples = []
            train_set = getattr(benchmark_instance, 'train_set', [])
            
            for i, example in enumerate(train_set[:num_examples]):
                try:
                    # Get input and output fields
                    input_keys = example.inputs().keys() if hasattr(example, 'inputs') else []
                    
                    # Build example dict
                    example_dict = {
                        "inputs": {},
                        "outputs": {}
                    }
                    
                    # Extract input fields
                    for key in input_keys:
                        if hasattr(example, key):
                            value = getattr(example, key)
                            # Convert to string for JSON serialization
                            example_dict["inputs"][key] = str(value) if value is not None else ""
                    
                    # Extract output fields (commonly 'answer')
                    for key in ['answer', 'target', 'output', 'response']:
                        if hasattr(example, key):
                            value = getattr(example, key)
                            example_dict["outputs"][key] = str(value) if value is not None else ""
                    
                    if example_dict["inputs"] and example_dict["outputs"]:
                        golden_examples.append(example_dict)
                        
                except Exception as e:
                    print(f"      Warning: Could not process example {i}: {e}")
                    continue
            
            return golden_examples
            
        except Exception as e:
            print(f"    Error extracting from {benchmark_name}: {e}")
            return []
    
    def load_golden_examples(self, benchmark_name: str) -> List[Dict[str, Any]]:
        """Load golden examples from JSON file."""
        json_file = self.golden_examples_dir / f"{benchmark_name}_golden_examples.json"
        
        if not json_file.exists():
            return []
        
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load golden examples for {benchmark_name}: {e}")
            return []
    
    def ensure_golden_examples_available(self) -> None:
        """Ensure golden examples are available for key benchmarks."""
        missing_benchmarks = []
        
        for benchmark_name in self.key_benchmarks:
            json_file = self.golden_examples_dir / f"{benchmark_name}_golden_examples.json"
            if not json_file.exists():
                missing_benchmarks.append(benchmark_name)
        
        if missing_benchmarks:
            print(f"📝 Missing golden examples for: {missing_benchmarks}")
            print("🔄 Extracting golden examples...")
            self.extract_and_save_golden_examples()


class MetaPromptOptimizer:
    """Optimizes prompts using meta-prompt strategies."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.available_strategies = list(DEFAULT_META_TEMPLATES.keys())
        
        # Initialize golden examples extractor
        self.golden_extractor = GoldenExampleExtractor()
        
        # Ensure golden examples are available
        self.golden_extractor.ensure_golden_examples_available()
        
        # Initialize OpenAI client for optimization
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                print("Warning: OPENROUTER_API_KEY environment variable not found")
                print("    Please set this variable in your .env file or environment")
                print("    Example: OPENROUTER_API_KEY=sk-or-v1-your-key-here")
            else:
                print(f"✅ Found API key: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}")
                
            # Use the API key directly from config if available, otherwise from env
            self.client = OpenAI(
                base_url=config.base_url,
                api_key=api_key,
                default_headers={"HTTP-Referer": "https://langprobe.ai"}
            )
        except ImportError:
            print("Warning: OpenAI library not available. Meta-prompt optimization will use fallback.")
            self.client = None
    
    def _format_golden_examples(self, golden_examples: List[Dict[str, Any]]) -> str:
        """Format golden examples for meta-prompt template."""
        if not golden_examples:
            return ""
        
        formatted = []
        for i, example in enumerate(golden_examples, 1):
            inputs_str = ", ".join(f"{k}: {v}" for k, v in example["inputs"].items())
            outputs_str = ", ".join(f"{k}: {v}" for k, v in example["outputs"].items())
            formatted.append(f"Example {i}:\nInputs: {inputs_str}\nOutputs: {outputs_str}")
        
        return "\n\n".join(formatted)
    
    def _format_few_shot_examples(self, golden_examples: List[Dict[str, Any]]) -> str:
        """Format golden examples as few-shot examples for the generated prompt.
        
        NOTE: This method is now deprecated as we're using golden examples only as context
        for the LLM to understand the task, not as few-shot examples in the generated prompt.
        """
        if not golden_examples:
            return ""
        
        few_shots = []
        for i, example in enumerate(golden_examples, 1):
            inputs_str = ", ".join(f"{k}: {v}" for k, v in example["inputs"].items())
            outputs_str = ", ".join(f"{k}: {v}" for k, v in example["outputs"].items())
            few_shots.append(f"- Example {i}: {inputs_str} -> {outputs_str}")
        
        return "\n".join(few_shots)
    
    def _process_template_with_golden_examples(self, template: str, golden_examples: List[Dict[str, Any]]) -> str:
        """Process template to handle golden examples conditionally."""
        if not golden_examples:
            # Remove conditional blocks for golden examples
            import re
            # Remove {{#if GOLDEN_EXAMPLES}}...{{/if}} blocks
            template = re.sub(r'\{\{#if GOLDEN_EXAMPLES\}\}.*?\{\{/if\}\}', '', template, flags=re.DOTALL)
            return template
        
        # Format golden examples for context
        formatted_examples = self._format_golden_examples(golden_examples)
        
        # Replace conditional blocks
        template = template.replace("{{#if GOLDEN_EXAMPLES}}", "")
        template = template.replace("{{/if}}", "")
        
        # Replace placeholders
        template = template.replace("{golden_examples}", formatted_examples)
        
        return template
    
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
            
        print(f"Starting meta-prompt optimization")
        print(f"Benchmarks: {list(benchmark_configs.keys())}")
        print(f"Strategies: {selected_strategies}")
        
        optimized_configs = {}
        
        for benchmark_name, config in benchmark_configs.items():
            print(f"\nOptimizing prompts for {benchmark_name}...")
            
            # Check if this is a multi-signature benchmark (hover)
            if benchmark_name == 'hover' and hasattr(config, 'multi_signatures'):
                optimized_config = self._optimize_multi_signature_benchmark(
                    config,
                    selected_strategies
                )
            else:
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
        print(f"  Running {len(strategies)} strategies in parallel (max_concurrent: {self.config.max_concurrent})")
        
        def optimize_strategy(strategy: str):
            """Optimize a single strategy and return results."""
            start_time = time.time()
            try:
                optimized_prompt = self._apply_meta_prompt_strategy(
                    benchmark_config.base_prompt,
                    benchmark_config.signature,
                    strategy,
                    benchmark_config.name  # Pass benchmark name for golden examples
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
    
    def _optimize_multi_signature_benchmark(
        self, 
        benchmark_config: BenchmarkConfig,
        strategies: List[str]
    ) -> BenchmarkConfig:
        """Optimize prompts for a multi-signature benchmark like hover."""
        
        print(f"  🔄 Multi-signature benchmark detected")
        
        # Create a new config that will hold all optimized signatures
        optimized_config = BenchmarkConfig(
            name=benchmark_config.name,
            base_prompt=benchmark_config.base_prompt,
            signature=benchmark_config.signature,
            task_description=benchmark_config.task_description,
            prompt_variations=[]
        )
        
        # Mark it as multi-signature
        optimized_config.multi_signatures = {}
        
        # For each strategy, optimize all signatures
        for strategy in strategies:
            print(f"  Optimizing with strategy: {strategy}")
            strategy_prompts = {}
            
            # Optimize each signature
            for sig_info in benchmark_config.multi_signatures:
                sig_key = sig_info['signature_key']
                print(f"    - Optimizing {sig_key}")
                
                optimized_prompt = self._apply_meta_prompt_strategy(
                    sig_info['instructions'],
                    f"{', '.join(sig_info['input_fields'])} -> {', '.join(sig_info['output_fields'])}",
                    strategy,
                    benchmark_config.name  # Pass benchmark name for golden examples
                )
                
                if optimized_prompt and len(optimized_prompt.strip()) > 50:
                    strategy_prompts[sig_key] = optimized_prompt.strip()
                else:
                    # Fall back to original
                    strategy_prompts[sig_key] = sig_info['instructions']
            
            # Create a multi-signature variation
            variation = PromptConfig(
                name=f"meta_{strategy}",
                description=f"Meta-optimized prompts using {strategy} strategy",
                instructions=json.dumps(strategy_prompts),  # Store as JSON
                task_description=benchmark_config.task_description,
                signature=benchmark_config.signature,
                meta_strategy=strategy,
                multi_signature=True,
                signature_prompts=strategy_prompts
            )
            
            optimized_config.add_variation(variation)
            optimized_config.multi_signatures[f"meta_{strategy}"] = strategy_prompts
        
        # Also add the original as a variation
        original_prompts = {
            sig['signature_key']: sig['instructions'] 
            for sig in benchmark_config.multi_signatures
        }
        
        original_variation = PromptConfig(
            name="extracted_default",
            description="Original extracted prompts",
            instructions=json.dumps(original_prompts),
            task_description=benchmark_config.task_description,
            signature=benchmark_config.signature,
            multi_signature=True,
            signature_prompts=original_prompts
        )
        
        optimized_config.add_variation(original_variation)
        optimized_config.multi_signatures["extracted_default"] = original_prompts
        
        return optimized_config
    
    def _apply_meta_prompt_strategy(
        self,
        original_prompt: str,
        signature: str,
        strategy: str,
        benchmark_name: str = None
    ) -> Optional[str]:
        """Apply a specific meta-prompt strategy to optimize a prompt."""
        
        if strategy not in DEFAULT_META_TEMPLATES:
            print(f"Warning: Strategy '{strategy}' not found in available templates")
            return self._fallback_optimization(original_prompt, strategy)

        meta_template = DEFAULT_META_TEMPLATES[strategy]
        
        # Load golden examples if benchmark name is provided
        golden_examples = []
        if benchmark_name:
            golden_examples = self.golden_extractor.load_golden_examples(benchmark_name)
            if golden_examples:
                print(f"      📝 Using {len(golden_examples)} golden examples")
        
        # Process template with golden examples
        meta_template = self._process_template_with_golden_examples(meta_template, golden_examples)
        
        # Fill in the template with the original prompt
        meta_prompt = meta_template.format(
            original_system_prompt=original_prompt
        )
        
        # Set logging context for this specific strategy call
        logging_manager = get_logging_manager()
        if logging_manager:
            active_logger = logging_manager.active_logger
            if active_logger:
                active_logger.set_context(benchmark=benchmark_name, strategy=strategy)
        
        # Create a fresh OpenAI client for each worker thread to ensure proper API key access
        try:
            import uuid
            from datetime import datetime
            from openai import OpenAI
            from dotenv import load_dotenv
            
            # Load environment variables in each worker thread
            root_dir = Path(__file__).parent.parent.parent
            env_path = root_dir / '.env'
            if env_path.exists():
                load_dotenv(env_path)
            
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                print(f"[{strategy}] No API key found in worker thread")
                return self._fallback_optimization(original_prompt, strategy)
            
            # Create a fresh client for this worker thread
            client = OpenAI(
                base_url=self.config.base_url,
                api_key=api_key,
                default_headers={"HTTP-Referer": "https://langprobe.ai"}
            )
            
            # Track timing for logging
            start_time = time.time()
            call_id = str(uuid.uuid4())
            
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=2000
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            optimized_prompt = response.choices[0].message.content.strip()
            
            # Extract token usage and cost
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0
            cost = getattr(response, 'cost', 0.0) if hasattr(response, 'cost') else 0.0
            
            # Log the call manually since it's not going through DSPy
            if logging_manager and logging_manager.active_logger:
                from .llm_logger import LLMCall
                call = LLMCall(
                    call_id=call_id,
                    timestamp=datetime.now().isoformat(),
                    phase="optimization",
                    benchmark=benchmark_name,
                    prompt_variation=None,
                    strategy=strategy,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=2000,
                    input_prompt=meta_prompt,
                    output_text=optimized_prompt,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    cost=cost,
                    latency_ms=latency_ms,
                    error=None,
                    context={
                        'worker_thread': True,
                        'strategy': strategy,
                        'benchmark': benchmark_name
                    }
                )
                logging_manager.active_logger.log_call(call)
            
            return optimized_prompt
            
        except Exception as e:
            # Log the error call
            if logging_manager and logging_manager.active_logger:
                from .llm_logger import LLMCall
                error_call = LLMCall(
                    call_id=call_id if 'call_id' in locals() else str(uuid.uuid4()),
                    timestamp=datetime.now().isoformat(),
                    phase="optimization",
                    benchmark=benchmark_name,
                    prompt_variation=None,
                    strategy=strategy,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=2000,
                    input_prompt=meta_prompt if 'meta_prompt' in locals() else "",
                    output_text="",
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    cost=0.0,
                    latency_ms=(time.time() - start_time) * 1000 if 'start_time' in locals() else 0.0,
                    error=str(e),
                    context={
                        'worker_thread': True,
                        'strategy': strategy,
                        'benchmark': benchmark_name
                    }
                )
                logging_manager.active_logger.log_call(error_call)
            
            print(f"Error during LLM optimization: {e}")
            return self._fallback_optimization(original_prompt, strategy)
        
        finally:
            # Clear the specific context after this call
            if logging_manager and logging_manager.active_logger:
                logging_manager.active_logger.clear_context()
    
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
                    print(f"Validation failed for {benchmark_name}.{variation.name}")
        
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