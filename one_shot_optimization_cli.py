#!/usr/bin/env python3
"""
Entry point for the One-Shot Optimization Pipeline CLI.

This script provides easy access to the one-shot optimization pipeline from the project root.
For the full implementation, see experimental.one_shot_optimization.cli
"""

import sys
from pathlib import Path

# Load environment variables from .env file
import os
from dotenv import load_dotenv

# Try to load from .env file in the project root
root_dir = Path(__file__).parent
env_path = root_dir / '.env'
if env_path.exists():
    print(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)
else:
    print(f"No .env file found at {env_path}")

# Add current directory to path for experimental modules
sys.path.append(str(Path(__file__).parent))

def main():
    """Entry point that delegates to the module-based CLI."""
    try:
        from experimental.one_shot_optimization.cli import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"❌ Error importing one-shot optimization CLI: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 