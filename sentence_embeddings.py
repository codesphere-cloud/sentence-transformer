from fastapi import FastAPI, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import os

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

class TextData(BaseModel):
    texts: List[str]

@app.post("/embeddings")
async def create_embeddings(text_data: TextData):
    try:
        embeddings = model.encode(text_data.texts)
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)