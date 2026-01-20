#!/bin/bash

echo "Running HPC worker"

echo ""

VENV_DIR="LUCID"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' exists... Activating..."
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment '$VENV_DIR' does not exist. Creating new venv..."
    if ! python3.11 -m venv "$VENV_DIR"; then
        echo "Error: Failed to create the virtual environment '$VENV_DIR'." >&2
        exit 1
    fi
    echo "Virtual environment '$VENV_DIR' created successfully. Activating..."
    source "$VENV_DIR/bin/activate"
    echo ""
    echo "Installing venv packages"
    echo ""
    pip install --upgrade pip
    echo "Updated pip"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    echo "Installed Pytorch"
    pip install -r requirements.txt
    echo ""
    echo "Installed requirements.txt"
    echo ""
    echo "Installation complete"
fi

mkdir -p SAR/models
mkdir -p Logs

echo "complete"