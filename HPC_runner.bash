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
    pip install --upgrade pip -q
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 -q
    pip install -r requirements.txt -q
    echo ""
    echo "Installation complete"
fi

mkdir -p SAR/models
mkdir -p Logs

echo ""


echo "
████████ ████████     ███    ████ ██    ██ ████ ██    ██  ██████      ██     ██  ███████  ████████  ████████ ██       
   ██    ██     ██   ██ ██    ██  ███   ██  ██  ███   ██ ██    ██     ███   ███ ██     ██ ██     ██ ██       ██       
   ██    ██     ██  ██   ██   ██  ████  ██  ██  ████  ██ ██           ████ ████ ██     ██ ██     ██ ██       ██       
   ██    ████████  ██     ██  ██  ██ ██ ██  ██  ██ ██ ██ ██   ████    ██ ███ ██ ██     ██ ██     ██ ██████   ██       
   ██    ██   ██   █████████  ██  ██  ████  ██  ██  ████ ██    ██     ██     ██ ██     ██ ██     ██ ██       ██       
   ██    ██    ██  ██     ██  ██  ██   ███  ██  ██   ███ ██    ██     ██     ██ ██     ██ ██     ██ ██       ██       
   ██    ██     ██ ██     ██ ████ ██    ██ ████ ██    ██  ██████      ██     ██  ███████  ████████  ████████ ████████ 
"


echo ""
echo ""

echo "Running ID-CNN Branch"

echo ""
echo ""

touch Logs/ID-CNN_Base
echo "Running ID-CNN Base Model"
python3 SAR/Denoiser/ID-CNN_Base.py >> Logs/ID-CNN_Base
echo "Ouput saved to Logs/ID-CNN_Base"


echo ""

touch Logs/ID-CNN_Inception_Full
echo "Running ID-CNN Full Inception Model"
python3 SAR/Denoiser/ID-CNN_Inception_Full.py >> Logs/ID-CNN_Inception_Full
echo "Ouput saved to Logs/ID-CNN_Inception_Full"

echo ""

touch Logs/ID-CNN_Inception_reduced
echo "Running ID-CNN Reduced Inception Model"
python3 SAR/Denoiser/ID-CNN_Inception_reduced.py >> Logs/ID-CNN_Inception_reduced
echo "Ouput saved to Logs/ID-CNN_Inception_reduced"

echo ""
echo ""

echo "Running SAR-Colorizer Branch"

echo ""

touch Logs/U-Net_Resnet
echo "U-Net Colorizer"
python3 SAR/Colorizer/U-Net_Resnet.py >> Logs/U-Net_Resnet
echo "Ouput saved to Logs/U-Net_Resnet"

echo ""

# touch Logs/U-Net_Resnet
# echo "U-Net Colorizer"
# python3 SAR/Colorizer/U-Net_Resnet.py >> Logs/U-Net_Resnet
# echo "Ouput saved to Logs/U-Net_Resnet"

echo ""

echo "
████████ ████████     ███    ████ ██    ██ ████ ██    ██  ██████       ██████   ███████  ██     ██ ████████  ██       ████████ ████████ ████████ 
   ██    ██     ██   ██ ██    ██  ███   ██  ██  ███   ██ ██    ██     ██    ██ ██     ██ ███   ███ ██     ██ ██       ██          ██    ██       
   ██    ██     ██  ██   ██   ██  ████  ██  ██  ████  ██ ██           ██       ██     ██ ████ ████ ██     ██ ██       ██          ██    ██       
   ██    ████████  ██     ██  ██  ██ ██ ██  ██  ██ ██ ██ ██   ████    ██       ██     ██ ██ ███ ██ ████████  ██       ██████      ██    ██████   
   ██    ██   ██   █████████  ██  ██  ████  ██  ██  ████ ██    ██     ██       ██     ██ ██     ██ ██        ██       ██          ██    ██       
   ██    ██    ██  ██     ██  ██  ██   ███  ██  ██   ███ ██    ██     ██    ██ ██     ██ ██     ██ ██        ██       ██          ██    ██       
   ██    ██     ██ ██     ██ ████ ██    ██ ████ ██    ██  ██████       ██████   ███████  ██     ██ ██        ████████ ████████    ██    ████████ 
"