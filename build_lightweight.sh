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

echo "--- Step 3: Packaging Bootstrap Launcher ---"

# Check for pyinstaller, install if missing
if ! python3 -m PyInstaller --version > /dev/null 2>&1; then
    echo "PyInstaller not found. Installing..."
    python3 -m pip install pyinstaller
fi

# We only package launcher.py. Heavy libs like torch will be downloaded by it.
python3 -m PyInstaller --onefile \
    --name WhisperFlow \
    --add-data "frontend/dist:frontend/dist" \
    --add-data "backend:backend" \
    --add-data "ai:ai" \
    launcher.py

echo "--- Build Complete! ---"
echo "The tiny executable 'WhisperFlow' is located in the 'dist' folder."
echo "Note: On first run, it will download about 500MB of runtime components."
