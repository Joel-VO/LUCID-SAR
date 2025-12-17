#!/bin/bash

echo "Running HPC worker"

echo ""

VENV_DIR="PULSAR"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' exists... Activating..."
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment '$VENV_DIR' does not exist. Creating new venv..."
    if ! python3 -m venv "$VENV_DIR"; then
        echo "Error: Failed to create the virtual environment '$VENV_DIR'." >&2
        exit 1
    fi
    echo "Virtual environment '$VENV_DIR' created successfully. Activating..."
    source "$VENV_DIR/bin/activate"
    echo ""
    echo "Installing venv packages"
    echo ""
    pip install --upgrade pip -q
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 -q
    pip install -r requirements.txt -q
    echo ""
    echo "Installation complete"
fi

echo ""


echo "████████ ████████     ███    ████ ██    ██ ████ ██    ██  ██████      ██     ██  ███████  ████████  ████████ ██       
   ██    ██     ██   ██ ██    ██  ███   ██  ██  ███   ██ ██    ██     ███   ███ ██     ██ ██     ██ ██       ██       
   ██    ██     ██  ██   ██   ██  ████  ██  ██  ████  ██ ██           ████ ████ ██     ██ ██     ██ ██       ██       
   ██    ████████  ██     ██  ██  ██ ██ ██  ██  ██ ██ ██ ██   ████    ██ ███ ██ ██     ██ ██     ██ ██████   ██       
   ██    ██   ██   █████████  ██  ██  ████  ██  ██  ████ ██    ██     ██     ██ ██     ██ ██     ██ ██       ██       
   ██    ██    ██  ██     ██  ██  ██   ███  ██  ██   ███ ██    ██     ██     ██ ██     ██ ██     ██ ██       ██       
   ██    ██     ██ ██     ██ ████ ██    ██ ████ ██    ██  ██████      ██     ██  ███████  ████████  ████████ ████████ 
"


echo ""
echo ""

echo "Running ID-CNN training loop"
python3 SAR/Denoiser/ID-CNN.py

echo ""
echo ""

echo "████████ ████████     ███    ████ ██    ██ ████ ██    ██  ██████      ████████   ███████  ██    ██ ████████ 
   ██    ██     ██   ██ ██    ██  ███   ██  ██  ███   ██ ██    ██     ██     ██ ██     ██ ███   ██ ██       
   ██    ██     ██  ██   ██   ██  ████  ██  ██  ████  ██ ██           ██     ██ ██     ██ ████  ██ ██       
   ██    ████████  ██     ██  ██  ██ ██ ██  ██  ██ ██ ██ ██   ████    ██     ██ ██     ██ ██ ██ ██ ██████   
   ██    ██   ██   █████████  ██  ██  ████  ██  ██  ████ ██    ██     ██     ██ ██     ██ ██  ████ ██       
   ██    ██    ██  ██     ██  ██  ██   ███  ██  ██   ███ ██    ██     ██     ██ ██     ██ ██   ███ ██       
   ██    ██     ██ ██     ██ ████ ██    ██ ████ ██    ██  ██████      ████████   ███████  ██    ██ ████████ 
"