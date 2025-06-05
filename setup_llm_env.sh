#!/bin/bash
set -e

# Step 1: Install Python 3.11 if not already installed (Ubuntu/debian)
if ! python3.11 --version &>/dev/null; then
  echo "Installing Python 3.11..."
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3.11-distutils python3-pip
fi

# Step 2: Create and activate a virtual environment
python3.11 -m venv llm_env
source llm_env/bin/activate

# Step 3: Upgrade pip, wheel, setuptools
pip install --upgrade pip wheel setuptools

# Step 4: Install all Python packages (requirements_llm.txt must be present)
pip install -r requirements.txt

echo "✅ LLM environment is ready. To activate: source llm_env/bin/activate"
