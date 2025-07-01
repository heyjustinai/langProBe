#!/usr/bin/env python3
"""
Entry point for the One-Shot Optimization Pipeline CLI.

This script provides easy access to the one-shot optimization pipeline from the project root.
For the full implementation, see experimental.one_shot_optimization.cli
"""

import sys
from pathlib import Path

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