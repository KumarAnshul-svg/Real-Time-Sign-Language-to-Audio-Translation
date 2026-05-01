# app.py (Advanced Version)

import os
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from spellchecker import SpellChecker

app = FastAPI()
spell = SpellChecker()

# =========================
# MODELS
# =========================

class CorrectionRequest(BaseModel):
    text: str

# =========================
# ENDPOINTS
# =========================

@app.post("/correct")
async def correct_text(req: CorrectionRequest):
    """
    Refines the predicted text using a spell checker.
    In a real-world app, this could be a call to Gemini or GPT.
    """
    words = req.text.split()
    corrected_words = []
    
    for word in words:
        # If the word is more than 1 char, try correcting it
        if len(word) > 1:
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
        else:
            corrected_words.append(word)
            
    return {"original": req.text, "corrected": " ".join(corrected_words)}

@app.get("/labels")
async def get_labels():
    """Serves the label list for the frontend."""
    try:
        with open("web/labels.json", "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
app.mount("/models", StaticFiles(directory="models"), name="models")
app.mount("/labels", StaticFiles(directory="labels"), name="labels")
app.mount("/", StaticFiles(directory="web", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
