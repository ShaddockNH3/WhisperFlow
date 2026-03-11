@echo off
echo --- Step 1: Building Frontend ---
cd frontend
if exist "pnpm-lock.yaml" (
    echo Detected pnpm-lock.yaml, using pnpm...
    call pnpm install
) else (
    call npm install --legacy-peer-deps
)
call npm run build
cd ..

echo --- Step 2: Preparing Backend ---
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
mkdir build\frontend\dist
xcopy /e /y frontend\dist build\frontend\dist

echo --- Step 3: Packaging with PyInstaller ---
python -m pip install pyinstaller

python -m PyInstaller --onedir ^
    --name WhisperFlow ^
    --add-data "frontend/dist;frontend/dist" ^
    --add-data "ai/denoise/weights;ai/denoise/weights" ^
    --collect-all faster_whisper ^
    --collect-all onnxruntime ^
    --hidden-import uvicorn.logging ^
    --hidden-import uvicorn.loops ^
    --hidden-import uvicorn.loops.auto ^
    --hidden-import uvicorn.protocols ^
    --hidden-import uvicorn.protocols.http ^
    --hidden-import uvicorn.protocols.http.auto ^
    --hidden-import uvicorn.protocols.websockets ^
    --hidden-import uvicorn.protocols.websockets.auto ^
    --hidden-import uvicorn.lifespan ^
    --hidden-import uvicorn.lifespan.on ^
    backend/main.py

echo --- Build Complete! ---
echo The executable 'WhisperFlow.exe' is located in the 'dist' folder.
echo Note: All data (models, etc.) will be stored in a 'data' folder next to the executable.
pause
