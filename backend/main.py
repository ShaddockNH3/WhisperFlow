import os
import sys
import time
import webbrowser
import multiprocessing
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import router  # Fixed: Use direct import, launcher adds folder to PYTHONPATH

# Load .env from 'data' folder next to executable for portability
# We prefer the environment variable passed by our launcher
app_root = os.environ.get("WHISPERFLOW_APP_ROOT")
if not app_root:
    if getattr(sys, 'frozen', False):
        app_root = os.path.dirname(sys.executable)
    else:
        app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env_path = os.path.join(app_root, "data", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

app = FastAPI(
    title="WhisperFlow",
    description="ASR System",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router.ws_router, prefix="/api")

# Resolve static files path for portable build
if getattr(sys, 'frozen', False):
    static_dir = os.path.join(sys._MEIPASS, "frontend/dist")
else:
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend/dist")

# Serve frontend if it exists
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "WhisperFlow API is running. Frontend not found."}

def open_browser():
    # Delay slightly to ensure server is up
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    # Required for Windows PyInstaller with multiprocessing
    multiprocessing.freeze_support()
    # Start browser opener in a separate process
    multiprocessing.Process(target=open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
