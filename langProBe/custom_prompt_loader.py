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
            except Exception:
                # If copying fails, use the original program
                optimized_program = program
        else:
            # If it's not a class instance, use the original
            optimized_program = program
        
        # Load the optimized prompt from the file
        try:
            with open(self.prompt_file_path, 'r') as f:
                optimized_data = json.load(f)
                
            # Apply the optimized prompt to the program
            if isinstance(optimized_data, dict):
                # If the file contains a dspy.Module serialization
                optimized_program = dspy.Module.load(optimized_data)
            else:
                # If the file contains just the instructions
                # Apply to all predictors that have a signature with instructions
                for predictor in optimized_program.predictors():
                    if hasattr(predictor, "signature") and hasattr(predictor.signature, "instructions"):
                        predictor.signature.instructions = optimized_data
                        
        except Exception as e:
            print(f"Error loading optimized prompt: {e}")
            # Return the original program if loading fails
            return program
            
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
