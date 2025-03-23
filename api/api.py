# api.py
import sys
import os

# Ajoutez le chemin du répertoire parent au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import json
import uuid
import tempfile
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from services.chat import EducationalAssistant
from services.audio_transcribe import WhisperTranscriber

# Configuration
DOCUMENTS_DIR = "D:/bureau/BD&AI 1/ci2/S2/th_info/cour"
QDRANT_PATH = "D:/bureau/BD&AI 1/ci2/S2/tec_veille/mini_projet/local_qdrant_storage"
COLLECTION_NAME = "asr_docs"
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "assistant_educatif"

# Initialize the assistant
assistant = EducationalAssistant(
    documents_dir=DOCUMENTS_DIR,
    qdrant_path=QDRANT_PATH,
    collection_name=COLLECTION_NAME,
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    experiment_name=EXPERIMENT_NAME
)

# Initialize the transcriber
transcriber = WhisperTranscriber(model_size="small", language="fr")
transcriber.load_model()  # Pre-load the model when API starts

app = FastAPI(title="Assistant Éducatif API")

# Data models
class MessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class MessageResponse(BaseModel):
    response: str
    session_id: str

class SessionsResponse(BaseModel):
    sessions: List[str]

class AnalyticsResponse(BaseModel):
    analytics: Dict[str, Any]

# Store active sessions
active_sessions = {}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribes an uploaded audio file and returns the text."""
    try:
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        
        # Save the uploaded file
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Transcribe the audio
        text = transcriber.transcribe(temp_path)
        
        # Clean up the temporary file
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        if text:
            return {"text": text, "success": True}
        else:
            raise HTTPException(
                status_code=500,
                detail="Transcription failed"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during transcription: {str(e)}"
        )

@app.post("/message", response_model=MessageResponse)
async def process_message(request: MessageRequest):
    """Traite un message et retourne une réponse de l'assistant éducatif."""
    try:
        # Utiliser le session_id existant ou en créer un nouveau
        session_id = request.session_id or str(uuid.uuid4())
        
        # Enregistrer la session comme active
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                "created_at": str(uuid.uuid1()),
                "message_count": 0
            }
        
        active_sessions[session_id]["message_count"] += 1
        
        # Traiter la requête
        response = assistant.process_query(request.message, session_id)
        
        return MessageResponse(
            response=response,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement du message: {str(e)}"
        )

@app.get("/sessions", response_model=SessionsResponse)
async def get_sessions():
    """Récupère la liste des sessions actives."""
    return SessionsResponse(sessions=list(active_sessions.keys()))

@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Récupère les statistiques d'utilisation de l'assistant."""
    try:
        analytics = assistant.get_analytics_summary()
        return AnalyticsResponse(analytics=analytics)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des statistiques: {str(e)}"
        )

@app.on_event("shutdown")
def shutdown_event():
    """Termine proprement la session MLflow avant de fermer l'application."""
    assistant.finish_session()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
