#!/bin/bash

# 🚀 LangProBe One-Shot Optimization Setup Script
# Automated setup for the one-shot optimization system

set -e  # Exit on any error

echo "🚀 =================================================="
echo "   LangProBe One-Shot Optimization Setup"
echo "   Setting up your environment..."
echo "=================================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -d "langProBe" ] || [ ! -d "experimental" ]; then
    print_error "Please run this script from the LangProBe project root directory"
    print_info "Expected to find 'langProBe/' and 'experimental/' folders in current directory"
    print_info "Current directory: $(pwd)"
    print_info "Contents: $(ls -la | grep '^d' | awk '{print $9}' | tr '\n' ' ')"
    exit 1
fi

print_success "Found LangProBe project directory"

# Check Python version
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ -z "$python_version" ]; then
    print_error "Python 3 is not installed or not in PATH"
    print_info "Please install Python 3.10+ first:"
    print_info "  - macOS: brew install python@3.10"
    print_info "  - Ubuntu: sudo apt update && sudo apt install python3.10"
    print_info "  - Or download from: https://www.python.org/downloads/"
    exit 1
fi

# Convert version to comparable number (e.g., 3.10 -> 310)
version_num=$(echo $python_version | sed 's/\.//')
if [ "$version_num" -lt 310 ]; then
    print_error "Python 3.10+ required, found Python $python_version"
    print_info "Please install Python 3.10+ and try again"
    exit 1
fi

print_success "Found Python $python_version"

# Detect environment manager preference
use_conda=false
if command -v conda &> /dev/null; then
    print_success "Found conda installation"
    echo
    read -p "Use conda for environment management? (recommended) [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Using Python venv instead of conda"
    else
        use_conda=true
        print_info "Using conda for environment management"
    fi
elif command -v python3 &> /dev/null; then
    print_info "Conda not found, using Python venv"
else
    print_error "Neither conda nor python3 found in PATH"
    exit 1
fi

# Environment setup
env_name="langprobe"
if [ "$use_conda" = true ]; then
    # Conda environment setup
    if conda env list | grep -q "^$env_name "; then
        print_warning "Environment '$env_name' already exists"
        read -p "Recreate it? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing environment..."
            conda env remove -n $env_name -y
        else
            print_info "Using existing environment"
            eval "$(conda shell.bash hook)"
            conda activate $env_name
            print_success "Activated existing conda environment '$env_name'"
        fi
    fi
    
    if ! conda env list | grep -q "^$env_name "; then
        print_info "Creating conda environment '$env_name' with Python 3.10..."
        conda create -n $env_name python=3.10 -y
    fi
    
    print_info "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate $env_name
    print_success "Activated conda environment '$env_name'"
    
else
    # Python venv setup
    venv_dir="${env_name}-env"
    if [ -d "$venv_dir" ]; then
        print_warning "Virtual environment '$venv_dir' already exists"
        read -p "Recreate it? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf "$venv_dir"
        else
            print_info "Using existing virtual environment"
            source "$venv_dir/bin/activate"
            print_success "Activated existing virtual environment"
        fi
    fi
    
    if [ ! -d "$venv_dir" ]; then
        print_info "Creating Python virtual environment '$venv_dir'..."
        python3 -m venv "$venv_dir"
    fi
    
    print_info "Activating virtual environment..."
    source "$venv_dir/bin/activate"
    print_success "Activated virtual environment '$venv_dir'"
fi

# Verify we're in the right environment
print_info "Current Python: $(which python)"
print_info "Python version: $(python --version)"

# Install dependencies
print_info "Installing Python dependencies..."

# Upgrade pip first
python -m pip install --upgrade pip

# Install base dependencies
if [ -f "requirements.txt" ]; then
    print_info "Installing from requirements.txt..."
    pip install -r requirements.txt
else
    print_warning "requirements.txt not found, installing core dependencies..."
    pip install dspy-ai>=2.6 requests shortuuid seaborn langchain
    pip install black sentence-transformers torch numpy
    pip install huggingface-hub langchain_community
fi

# Install one-shot optimization specific dependencies
print_info "Installing one-shot optimization dependencies..."
pip install openai python-dotenv pyyaml

print_success "All Python dependencies installed"

# Check for API key
api_key_configured=false
if [ -f ".env" ] && grep -q "OPENROUTER_API_KEY" ".env"; then
    api_key=$(grep "OPENROUTER_API_KEY" ".env" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    if [ -n "$api_key" ] && [ "$api_key" != "your-api-key-here" ] && [ "$api_key" != "sk-or-v1-your-actual-key-here" ]; then
        print_success "Found OpenRouter API key in .env file"
        api_key_configured=true
    fi
elif [ -n "$OPENROUTER_API_KEY" ]; then
    print_success "Found OpenRouter API key in environment"
    api_key_configured=true
fi

if [ "$api_key_configured" = false ]; then
    echo
    print_warning "OpenRouter API key not configured"
    print_info "You'll need to set up your OpenRouter API key to use the optimization features"
    echo
    read -p "Do you have an OpenRouter API key? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo
        print_info "Please enter your OpenRouter API key (starts with sk-or-v1-):"
        read -s api_key
        echo
        if [ -n "$api_key" ]; then
            echo "OPENROUTER_API_KEY=$api_key" > .env
            print_success "API key saved to .env file"
            api_key_configured=true
        else
            print_warning "No API key entered, you'll need to set it up manually later"
        fi
    else
        print_info "No problem! You can get an API key later from:"
        print_info "  1. Visit https://openrouter.ai/"
        print_info "  2. Create an account"
        print_info "  3. Generate an API key"
        print_info "  4. Add it to .env file: OPENROUTER_API_KEY=sk-or-v1-your-key"
    fi
fi

# Verify installation
echo
print_info "Verifying installation..."

# Test 1: CLI import
if python -c "import experimental.one_shot_optimization.cli" 2>/dev/null; then
    print_success "CLI module imports successfully"
else
    print_error "Failed to import CLI module"
    print_info "This might indicate a dependency issue"
fi

# Test 2: Basic CLI command
if python -m experimental.one_shot_optimization.cli --help >/dev/null 2>&1; then
    print_success "CLI command works"
else
    print_error "CLI command failed"
fi

# Test 3: List strategies (requires API key)
if [ "$api_key_configured" = true ]; then
    if python -m experimental.one_shot_optimization.cli list-strategies >/dev/null 2>&1; then
        print_success "Strategy listing works (API connection OK)"
    else
        print_warning "Strategy listing failed (check API key or connection)"
    fi
else
    print_info "Skipping API tests (no API key configured)"
fi

# Create basic directory structure if needed
print_info "Ensuring directory structure..."
mkdir -p meta-optimize-prompt
mkdir -p configs/pipeline_configs
mkdir -p configs/custom_prompts

# Final instructions
echo
print_success "=================================================="
print_success "   Setup Complete!"
print_success "=================================================="
echo

if [ "$use_conda" = true ]; then
    print_info "📝 To use the system in future sessions:"
    echo "   conda activate $env_name"
else
    print_info "📝 To use the system in future sessions:"
    echo "   source $venv_dir/bin/activate"
fi

echo
print_info "🚀 Quick Test Commands:"
echo "   # Test basic functionality"
echo "   python -m experimental.one_shot_optimization.cli --help"
echo
echo "   # List available strategies"
echo "   python -m experimental.one_shot_optimization.cli list-strategies"
echo

if [ "$api_key_configured" = true ]; then
    echo "   # Run a quick optimization test"
    echo "   python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease"
    echo
    echo "   # Run full pipeline"
    echo "   python -m experimental.one_shot_optimization.cli pipeline"
else
    print_warning "⚠️  API Key Required:"
    echo "   Before running optimizations, set up your OpenRouter API key:"
    echo "   1. Visit: https://openrouter.ai/"
    echo "   2. Get your API key (starts with sk-or-v1-)"
    echo "   3. Add to .env file: OPENROUTER_API_KEY=sk-or-v1-your-key"
    echo "   4. Then run: python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease"
fi

echo
print_info "📚 Documentation:"
echo "   - Setup Guide: experimental/one_shot_optimization/SETUP.md"
echo "   - Usage Guide: experimental/one_shot_optimization/README.md"
echo "   - Configuration: configs/pipeline_configs/"

echo
print_success "🎉 Ready to optimize your prompts!"
