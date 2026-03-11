#!/bin/bash

# WhisperFlow Lightweight Build Script
# This script bundles the frontend and backend into a single executable.

set -e

echo "--- Step 1: Building Frontend ---"
cd frontend
if [ -f "pnpm-lock.yaml" ]; then
    echo "Detected pnpm-lock.yaml, using pnpm..."
    pnpm install || npm install --legacy-peer-deps
else
    npm install --legacy-peer-deps
fi
npm run build
cd ..

echo "--- Step 2: Preparing Backend ---"
# Create a temporary directory for PyInstaller to collect files
rm -rf build dist
mkdir -p build/frontend/dist
cp -r frontend/dist/* build/frontend/dist/

echo "--- Step 3: Packaging with PyInstaller ---"

# Check for pyinstaller, install if missing
if ! python3 -m PyInstaller --version > /dev/null 2>&1; then
    echo "PyInstaller not found. Installing..."
    python3 -m pip install pyinstaller
fi

python3 -m PyInstaller --onedir \
    --name WhisperFlow \
    --add-data "frontend/dist:frontend/dist" \
    --add-data "ai/denoise/weights:ai/denoise/weights" \
    --collect-all faster_whisper \
    --collect-all onnxruntime \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.loops \
    --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols \
    --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.protocols.websockets \
    --hidden-import uvicorn.protocols.websockets.auto \
    --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    backend/main.py

echo "--- Build Complete! ---"
echo "The executable 'WhisperFlow' is located in the 'dist' folder."
echo "Users can double-click it to start the application."
echo "Note: All data (models, etc.) will be stored in a 'data' folder next to the executable."
echo "To uninstall, simply delete the folder containing WhisperFlow."
