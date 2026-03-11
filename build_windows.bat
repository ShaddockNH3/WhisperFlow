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

echo --- Step 3: Packaging Bootstrap Launcher ---
python -m pip install pyinstaller

python -m PyInstaller --onefile ^
    --name WhisperFlow ^
    --add-data "frontend/dist;frontend/dist" ^
    --add-data "backend;backend" ^
    --add-data "ai;ai" ^
    launcher.py

echo --- Build Complete! ---
echo The tiny executable 'WhisperFlow.exe' is located in the 'dist' folder.
echo Note: On first run, it will download necessary components to the 'data' folder.
pause
