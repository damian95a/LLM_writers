from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from fastapi.middleware.cors import CORSMiddleware

from .env import QDRANT_CLOUD_URL, QDRANT_API_KEY
from .plotting import (
    get_filtered_data_after_tsne,
    get_tsne_of_embeddings,
    get_tsne_of_authors_avg,
)
from .proj import OLLAMA_MODEL_NAME, QDRANT_COLLECTION_NAME

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/tsne")
def tsne_filtered(
    authors: Optional[str] = Query(default=None),
    model_name: str = Query(default=OLLAMA_MODEL_NAME),
):
    try:
        qdrant_client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)
        tsne_data = get_filtered_data_after_tsne(
            qdrant_client,
            collection_name=QDRANT_COLLECTION_NAME,
            model_name=model_name,
            authors_filter=authors,
        )
        return JSONResponse(content=tsne_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/tsne-books")
def get_tsne_books(
    # authors: Optional[str] = Query(default=None),
    model_name: str = Query(default=OLLAMA_MODEL_NAME),
):
    try:
        qdrant_client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)
        tsne_data = get_tsne_of_embeddings(
            qdrant_client, collection_name=QDRANT_COLLECTION_NAME, model_name=model_name
        )
        return JSONResponse(content=tsne_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/author-average-tsne")
def get_author_average_tsne(
    # authors: Optional[str] = Query(default=None),
    model_name: str = Query(default=OLLAMA_MODEL_NAME),
):
    try:
        qdrant_client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)
        tsne_data = get_tsne_of_authors_avg(
            qdrant_client, collection_name=QDRANT_COLLECTION_NAME, model_name=model_name
        )
        return JSONResponse(content=tsne_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
