from fastapi import FastAPI, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import os

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

class TextData(BaseModel):
    texts: List[str]

def verify_api_key(api_key: Optional[str] = Query(None, alias='key')):
    expected_api_key = os.getenv("API_KEY")
    if api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

@app.post("/embeddings")
async def create_embeddings(text_data: TextData, api_key: str = Depends(verify_api_key)):
    try:
        embeddings = model.encode(text_data.texts)
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)