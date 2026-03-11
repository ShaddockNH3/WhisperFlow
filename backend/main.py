from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ws_api import router as ws_router
from dotenv import load_dotenv

load_dotenv()  # 加载项目根目录或 backend/ 目录下的 .env 文件

app = FastAPI(
    title="WhisperFlow API",
    description="A pseudo-streaming ASR pipeline",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to WhisperFlow API"}
