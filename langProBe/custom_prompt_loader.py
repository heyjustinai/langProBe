import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import dspy
from dspy.teleprompt import Teleprompter
from langProBe.optimizers import OptimizerConfig


class CustomPromptLoader(Teleprompter):
    """
    A custom optimizer that loads a pre-generated optimized prompt from a file
    and applies it to a program without running the optimization process again.
    """

    def __init__(self, prompt_file_path: str, metric=None, **kwargs):
        """
        Initialize the CustomPromptLoader.
        
        Args:
            prompt_file_path: Path to the JSON file containing the optimized prompt
            metric: Evaluation metric (stored but not passed to parent)
        """
        # Initialize the parent class without passing metric
        super().__init__(**kwargs)
        self.metric = metric  # Store metric separately
        self.prompt_file_path = prompt_file_path
        self.name = f"CustomPrompt-{os.path.basename(prompt_file_path)}"

    def compile(self, program, trainset=None, valset=None):
        """
        Load the optimized program from the file and return it.
        
        Args:
            program: The original program to optimize
            trainset: Training dataset (not used)
            valset: Validation dataset (not used)
            
        Returns:
            The optimized program with the loaded prompt
        """
        print(f"🚀 CustomPromptLoader.compile() starting for: {self.prompt_file_path}")
        
        # Store original instructions for comparison
        original_instructions = []
        if hasattr(program, 'predictors'):
            for i, predictor in enumerate(program.predictors()):
                if hasattr(predictor, "signature") and hasattr(predictor.signature, "instructions"):
                    original_instructions.append((i, predictor.signature.instructions))
        
        print(f"📋 Original program has {len(original_instructions)} predictors with instructions")
        for i, instr in original_instructions:
            print(f"  Predictor {i}: '{instr[:100]}...'")
        
        # Create a copy of the program to avoid modifying the original
        # Since dspy doesn't have a copy_module function, we'll use a different approach
        # For DSPy modules, we can create a new instance with the same configuration
        if hasattr(program, '__class__'):
            try:
                # Try to create a new instance of the same class
                optimized_program = program.__class__()
                # Copy over attributes that don't start with _
                for attr in dir(program):
                    if not attr.startswith('_') and hasattr(program, attr) and not callable(getattr(program, attr)):
                        setattr(optimized_program, attr, getattr(program, attr))
                print(f"✅ Successfully created copy of program class: {program.__class__.__name__}")
            except Exception as e:
                # If copying fails, use the original program
                print(f"❌ SILENT FALLBACK #1: Program copying failed: {e}")
                print(f"🔄 Using original program instead of copy")
                optimized_program = program
        else:
            # If it's not a class instance, use the original
            print(f"⚠️  SILENT FALLBACK #2: Program is not a class instance, using original")
            optimized_program = program
        
        # Load the optimized prompt from the file
        try:
            with open(self.prompt_file_path, 'r') as f:
                optimized_data = json.load(f)
                
            # Apply the optimized prompt to the program
            if isinstance(optimized_data, dict):
                # Check if this is our structured prompt format
                if 'instructions' in optimized_data:
                    # Handle our custom JSON format: {"task_description": ..., "signature": ..., "instructions": ...}
                    instructions = optimized_data['instructions']
                    
                    print(f"🔧 CustomPromptLoader: Applying custom instructions from {self.prompt_file_path}")
                    print(f"📝 Instructions length: {len(instructions)} characters")
                    print(f"📝 Instructions preview: {instructions[:150]}...")
                    
                    # Apply instructions to all predictors that have a signature with instructions
                    applied_count = 0
                    for predictor in optimized_program.predictors():
                        if hasattr(predictor, "signature") and hasattr(predictor.signature, "instructions"):
                            old_instructions = predictor.signature.instructions
                            predictor.signature.instructions = instructions
                            applied_count += 1
                            print(f"✅ Applied custom instructions to {type(predictor).__name__}")
                            print(f"   📝 Original: '{old_instructions[:60]}...'")
                            print(f"   📝 New:      '{instructions[:60]}...'")
                    
                    print(f"📊 Applied instructions to {applied_count} predictors")
                
                elif 'multi_signature' in optimized_data and optimized_data['multi_signature']:
                    # Handle multi-signature prompts (e.g., for hover)
                    prompts = optimized_data.get('prompts', {})
                    print(f"🔧 CustomPromptLoader: Applying multi-signature prompts from {self.prompt_file_path}")
                    print(f"📝 Found prompts for: {list(prompts.keys())}")
                    
                    # Special handling for hover benchmark
                    if hasattr(optimized_program, 'summarize1') and 'summarize1' in prompts:
                        optimized_program.summarize1.signature.instructions = prompts['summarize1']
                        print(f"✅ Applied custom instructions to summarize1")
                    
                    if hasattr(optimized_program, 'summarize2') and 'summarize2' in prompts:
                        optimized_program.summarize2.signature.instructions = prompts['summarize2']
                        print(f"✅ Applied custom instructions to summarize2")
                    
                    if hasattr(optimized_program, 'create_query_hop2') and 'query_hop2' in prompts:
                        optimized_program.create_query_hop2.signature.instructions = prompts['query_hop2']
                        print(f"✅ Applied custom instructions to create_query_hop2")
                    
                    if hasattr(optimized_program, 'create_query_hop3') and 'query_hop3' in prompts:
                        optimized_program.create_query_hop3.signature.instructions = prompts['query_hop3']
                        print(f"✅ Applied custom instructions to create_query_hop3")
                    
                    # Also check if the program accepts optimized_prompts parameter
                    if hasattr(optimized_program, '__init__'):
                        try:
                            # Try to create a new instance with optimized prompts
                            optimized_program = optimized_program.__class__(optimized_prompts=prompts)
                            print(f"✅ Created new program instance with optimized prompts")
                        except Exception as e:
                            print(f"⚠️  Could not create new instance with prompts: {e}")
                
                elif 'traces' in optimized_data or 'signature' in optimized_data:
                    # This looks like a dspy.Module serialization
                    optimized_program = dspy.Module.load(optimized_data)
                    print(f"🔧 CustomPromptLoader: Loaded DSPy module from {self.prompt_file_path}")
                
                else:
                    print(f"❌ SILENT FALLBACK #3: Unknown dict format in {self.prompt_file_path}")
                    print(f"🔄 Available keys: {list(optimized_data.keys())}")
                    print(f"🔄 Returning original program without modification")
                    return program
                    
            else:
                # If the file contains just the instructions string
                # Apply to all predictors that have a signature with instructions
                print(f"🔧 CustomPromptLoader: Applying string instructions from {self.prompt_file_path}")
                for predictor in optimized_program.predictors():
                    if hasattr(predictor, "signature") and hasattr(predictor.signature, "instructions"):
                        predictor.signature.instructions = optimized_data
                        print(f"✅ Applied string instructions to {type(predictor).__name__}")
                        
        except Exception as e:
            print(f"❌ SILENT FALLBACK #4: Exception loading optimized prompt: {e}")
            print(f"🔄 Returning original program due to exception")
            import traceback
            traceback.print_exc()
            # Return the original program if loading fails
            return program
        
        # FINAL VALIDATION: Check if instructions were actually changed
        print(f"\n🔍 FINAL VALIDATION:")
        
        final_instructions = []
        if hasattr(optimized_program, 'predictors'):
            for i, predictor in enumerate(optimized_program.predictors()):
                if hasattr(predictor, "signature") and hasattr(predictor.signature, "instructions"):
                    final_instructions.append((i, predictor.signature.instructions))
        
        print(f"📋 Final program has {len(final_instructions)} predictors with instructions")
        
        # Compare original vs final instructions
        instructions_changed = False
        if len(original_instructions) == len(final_instructions):
            for (orig_i, orig_instr), (final_i, final_instr) in zip(original_instructions, final_instructions):
                if orig_instr != final_instr:
                    instructions_changed = True
                    print(f"✅ Predictor {final_i}: Instructions CHANGED")
                    print(f"   📝 Length change: {len(orig_instr)} → {len(final_instr)}")
                else:
                    print(f"❌ Predictor {final_i}: Instructions UNCHANGED (potential silent fallback)")
        else:
            print(f"⚠️  Number of predictors changed: {len(original_instructions)} → {len(final_instructions)}")
        
        if not instructions_changed:
            print(f"🚨 CRITICAL: NO INSTRUCTIONS WERE CHANGED!")
            print(f"🚨 This indicates a SILENT FALLBACK occurred somewhere!")
            print(f"🚨 The evaluation will use original instructions, not optimized ones!")
        else:
            print(f"✅ SUCCESS: Instructions were successfully modified")
            
        print(f"🏁 CustomPromptLoader.compile() completed\n")
        return optimized_program


def create_custom_prompt_optimizer(prompt_file_path: str) -> OptimizerConfig:
    """
    Create an OptimizerConfig for the CustomPromptLoader.
    
    Args:
        prompt_file_path: Path to the JSON file containing the optimized prompt
        
    Returns:
        An OptimizerConfig for the CustomPromptLoader
    """
    from langProBe.optimizers import OptimizerConfig
    
    return OptimizerConfig(
        optimizer=CustomPromptLoader,
        init_args=dict(prompt_file_path=prompt_file_path),
        compile_args=dict(),
        langProBe_configs=dict(use_valset=False, save_candidate_score=False),
        name=f"CustomPrompt-{os.path.basename(prompt_file_path)}"
    )
