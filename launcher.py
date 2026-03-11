import os
import sys
import subprocess
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

# Configuration
APP_NAME = "WhisperFlow"
VERSION = "v1.0.0"
# Increment this whenever the deps list below changes, to force re-install on existing runtimes.
DEPS_VERSION = "2"
DATA_DIR = Path(sys.executable).parent / "data"
RUNTIME_DIR = DATA_DIR / "runtime"
MODELS_DIR = DATA_DIR / "models"
# Specific paths for Python 3.12.3
PYTHON_EXE = RUNTIME_DIR / ("python.exe" if sys.platform == "win32" else "bin/python3")

# URL for portable Python 3.12.3
PYTHON_URLS = {
    "win32": "https://www.python.org/ftp/python/3.12.3/python-3.12.3-embed-amd64.zip",
    "linux": "https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.12.3+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz"
}

def report_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading Runtime (Python 3.12.3)... {percent}%")
    sys.stdout.flush()

def _install_deps():
    """Install / upgrade all Python dependencies into the portable runtime."""
    print("Installing AI libraries using uv (this may take a while on first run)...")
    uv_cache = DATA_DIR / "uv_cache"
    uv_cache.mkdir(parents=True, exist_ok=True)

    deps = [
        "fastapi", "uvicorn", "python-multipart", "pydantic", "aiofiles",
        "torch", "torchaudio",
        "faster-whisper", "onnxruntime", "python-dotenv", "websockets",
        "opencc-python-reimplemented", "soundfile", "librosa", "numpy<2.0.0",
        "zhipuai>=2.1.0",
    ]

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(uv_cache)

    cmd = [
        str(PYTHON_EXE), "-m", "uv", "pip", "install",
        "--index-url", "https://pypi.org/simple",
        "--extra-index-url", "https://download.pytorch.org/whl/cpu"
    ] + deps

    subprocess.run(cmd, env=env, check=True)

    # Record the deps version so we can skip this step on the next launch.
    (DATA_DIR / "deps_version.txt").write_text(DEPS_VERSION)
    print("\n--- Dependencies up to date. ---")


def setup_runtime():
    deps_version_file = DATA_DIR / "deps_version.txt"
    needs_python = not PYTHON_EXE.exists()
    needs_deps = needs_python or not deps_version_file.exists() or deps_version_file.read_text().strip() != DEPS_VERSION

    if not needs_python and not needs_deps:
        return True

    if needs_python:
        print(f"--- {APP_NAME} First Run Setup ---")
        print(f"Installing isolated Python 3.12.3 environment into {RUNTIME_DIR}...")

        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        url = PYTHON_URLS.get(sys.platform if sys.platform == "win32" else "linux")
        if not url:
            print(f"Unsupported platform: {sys.platform}")
            return False

        temp_file = DATA_DIR / ("python_runtime.zip" if sys.platform == "win32" else "python_runtime.tar.gz")

        try:
            urllib.request.urlretrieve(url, temp_file, reporthook=report_progress)
            print("\nExtracting environment...")

            if sys.platform == "win32":
                with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                    zip_ref.extractall(RUNTIME_DIR)

                # Windows Embeddable Python Fix: Enable site-packages
                pth_file = RUNTIME_DIR / "python312._pth"
                if pth_file.exists():
                    with open(pth_file, "w") as f:
                        # Overwrite to ensure standard imports work
                        f.write(".\n")
                        f.write("python312.zip\n")
                        f.write("Lib/site-packages\n")
                        f.write("import site\n")
                (RUNTIME_DIR / "Lib/site-packages").mkdir(parents=True, exist_ok=True)
            else:
                with tarfile.open(temp_file, "r:gz") as tar_ref:
                    tar_ref.extractall(DATA_DIR)
                    extracted_dir = DATA_DIR / "python"
                    if extracted_dir.exists():
                        if RUNTIME_DIR.exists():
                            shutil.rmtree(RUNTIME_DIR)
                        shutil.move(extracted_dir, RUNTIME_DIR)

            temp_file.unlink()

            print("Installing package manager (pip & uv for speed)...")
            pip_script = DATA_DIR / "get-pip.py"
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", pip_script)
            subprocess.run([str(PYTHON_EXE), str(pip_script), "--no-warn-script-location"], check=True)
            pip_script.unlink()

            subprocess.run([str(PYTHON_EXE), "-m", "pip", "install", "uv", "--no-warn-script-location"], check=True)

        except Exception as e:
            print(f"\nError during runtime setup: {e}")
            return False

    try:
        if needs_deps:
            print(f"--- Updating dependencies (version {DEPS_VERSION}) ---")
            _install_deps()
    except Exception as e:
        print(f"\nError during dependency installation: {e}")
        return False

    print("\n--- Setup Complete! WhisperFlow is ready. ---")
    return True

def launch_backend():
    print(f"Starting {APP_NAME}...")
    # Since launcher is bundled, __file__ points to the temp extraction dir
    temp_dir = Path(__file__).parent
    backend_script = temp_dir / "backend" / "main.py"
    
    # The real app folder where the .exe sits
    real_app_root = Path(sys.executable).parent
    
    # Set environment variables for portability within the subprocess
    env = os.environ.copy()
    env["WHISPERFLOW_APP_ROOT"] = str(real_app_root)
    env["HF_HOME"] = str(MODELS_DIR / "huggingface")
    env["XDG_CACHE_HOME"] = str(MODELS_DIR / "xdg")
    env["TORCH_HOME"] = str(MODELS_DIR / "torch")
    
    # Ensure backend folder is in PYTHONPATH so imports like 'from router import ...' work
    env["PYTHONPATH"] = str(temp_dir / "backend") + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        # Run the backend using the portable python
        subprocess.run([str(PYTHON_EXE), str(backend_script)], env=env)
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    if setup_runtime():
        launch_backend()
    else:
        print("Failed to initialize environment.")
        input("Press Enter to exit...")
