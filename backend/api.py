from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict, Any
import uuid
from services.ai_service import process_audio

router = APIRouter()

# In-memory storage for tasks
mock_db: Dict[str, Any] = {}

@router.post("/recognize")
async def recognize_audio(file: UploadFile = File(...)):
    """
    Receives an audio file, processes it (mocked), and returns the transcription.
    """
    if not file.content_type.startswith("audio/"):
         return {"warning": f"Content type {file.content_type} might not be audio, but proceeding."}
    
    try:
        audio_bytes = await file.read()
        
        # Call the AI service (currently mocked)
        transcription = await process_audio(audio_bytes)
        
        task_id = str(uuid.uuid4())
        record = {
            "task_id": task_id,
            "filename": file.filename,
            "transcription": transcription,
            "status": "completed"
        }
        
        # Store in mock db
        mock_db[task_id] = record
        
        return record
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/recognize/{task_id}")
async def delete_record(task_id: str):
    """
    Deletes a previously recognized audio record.
    """
    if task_id not in mock_db:
        raise HTTPException(status_code=404, detail="Record not found")
        
    del mock_db[task_id]
    return {"message": "Record deleted successfully", "task_id": task_id}
