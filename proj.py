import numpy as np
import os
import sys
import pickle
import json
import uuid
import dotenv
dotenv.load_dotenv()

from typing import List, Dict, Tuple
from qdrant_client import QdrantClient, models

from database import *
from plotting import *
from searching import *

try:
    from qdrant_client.http.models import ScrollRequest, ScrollResponse
except ImportError:

    class ScrollRequest: pass # Placeholder
    class ScrollResponse: # Placeholder
        def __init__(self, points, next_page_offset):
            self.points = points
            self.next_page_offset = next_page_offset

QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


DATA_DIR = "data/raw_data"
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 30
QDRANT_BATCH_SIZE = 64
QDRANT_COLLECTION_NAME = "literary_embeddings"


source_texts = []
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".txt"):
            # Store as (relative_path_from_DATA_DIR, child_folder_name)
            rel_path = os.path.relpath(os.path.join(root, file), DATA_DIR)
            # Get the immediate child folder name (if any)
            author_folder = os.path.relpath(root, DATA_DIR).split(os.sep)[0] if os.path.relpath(root, DATA_DIR) != '.' else ''
            source_texts.append((rel_path, author_folder, file))

VECTOR_DIMENSION = None

def do_everything(current_ollama_model_name: str):
    if not QDRANT_CLOUD_URL or not QDRANT_API_KEY:
        print("Error: QDRANT_CLOUD_URL and QDRANT_API_KEY environment variables must be set.")
        exit(1)

    # Check Ollama, determine VECTOR_DIMENSION based on the current_ollama_model_name
    check_ollama_status_and_get_dim(current_ollama_model_name)
    if VECTOR_DIMENSION is None:
        print(f"Error: VECTOR_DIMENSION was not set for model {current_ollama_model_name}. Cannot proceed.")
        exit(1)

    print(f"Initializing Qdrant client for URL: {QDRANT_CLOUD_URL[:40]}...")
    qdrant_client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY, timeout=30)

    setup_qdrant_collection(qdrant_client, QDRANT_COLLECTION_NAME, VECTOR_DIMENSION, current_ollama_model_name, recreate_if_needed=False)

    total_points_upserted_session = 0

    for source_text_filename, author_folder, file_name in source_texts:
        filepath = os.path.join(DATA_DIR, source_text_filename)
        if not os.path.exists(filepath):
            print(f"Warning: Source text file not found: {filepath}. Skipping.")
            continue

        print(f"\nProcessing file: {source_text_filename} using model: {current_ollama_model_name}")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + CHUNK_SIZE_WORDS]
            chunks.append(" ".join(chunk_words))
            if i + CHUNK_SIZE_WORDS >= len(words): break
            i += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

        print(f"Generating embeddings for {len(chunks)} chunks...")

        points_batch = []
        for idx, chunk_text in enumerate(chunks):
            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(chunks) -1 :
                 print(f"  Processing chunk {idx + 1}/{len(chunks)} for {source_text_filename}...")

            embedding_array_2d = get_ollama_embeddings(chunk_text, current_ollama_model_name, OLLAMA_EMBED_API_URL)

            if embedding_array_2d is not None and embedding_array_2d.shape == (1, VECTOR_DIMENSION):
                embedding_vector_1d = embedding_array_2d[0].tolist()
                point_id = str(uuid.uuid4()) # Or generate a deterministic ID if you might re-embed same chunk with same model

                point = models.PointStruct(
                    id=point_id,
                    vector=embedding_vector_1d,
                    payload={
                        "source_file": file_name,
                        "chunk_text": chunk_text,
                        "chunk_index": idx,
                        "model_name": current_ollama_model_name,
                        "author": author_folder # <-- ADDED MODEL NAME HERE
                    }
                )
                points_batch.append(point)

                if len(points_batch) >= QDRANT_BATCH_SIZE:
                    try:
                        qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points_batch, wait=True)
                        print(f"    Upserted batch of {len(points_batch)} points to Qdrant.")
                        total_points_upserted_session += len(points_batch)
                        points_batch = []
                    except Exception as e:
                        print(f"    Error upserting batch to Qdrant: {e}")
            else:
                print(f"    Failed to get valid embedding for chunk {idx+1}. Skipping.")
                if embedding_array_2d is not None:
                    print(f"    Received embedding shape: {embedding_array_2d.shape}, expected: (1, {VECTOR_DIMENSION})")

        if points_batch: # Upsert remaining
            try:
                qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points_batch, wait=True)
                print(f"    Upserted final batch of {len(points_batch)} points for {source_text_filename} to Qdrant.")
                total_points_upserted_session += len(points_batch)
            except Exception as e:
                print(f"    Error upserting final batch for {source_text_filename} to Qdrant: {e}")

    print(f"\nFinished processing. Total points upserted in this session for model '{current_ollama_model_name}': {total_points_upserted_session}")
    try:
        collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Collection '{QDRANT_COLLECTION_NAME}' now has {collection_info.points_count} total points.")
    except Exception as e:
        print(f"Could not retrieve final collection info: {e}")

def normalize_vector_np(vec: np.ndarray) -> np.ndarray:
    if vec.ndim > 1: vec = vec.flatten()
    norm = np.linalg.norm(vec)
    if norm == 0: return vec
    return vec / norm

############
def process_texts(ollama_model):
    #--- Run the process with the chosen OLLAMA_MODEL_NAME ---
    print(f"Starting embedding generation and upload to Qdrant Cloud using model: {ollama_model}")
    print("Ensure QDRANT_CLOUD_URL, and QDRANT_API_KEY are correctly set.")
    print(f"Embeddings will be stored in Qdrant collection: {QDRANT_COLLECTION_NAME}")

    do_everything(current_ollama_model_name=ollama_model)

    print("\nProcessing complete.")

    print(f"Initializing Qdrant client for search (URL: {QDRANT_CLOUD_URL[:40]}...).")
    try:
        q_client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY, timeout=30)
        # Verify collection exists (optional, but good practice)
        q_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Successfully connected to Qdrant and collection '{QDRANT_COLLECTION_NAME}' is accessible.")
    except Exception as e:
        print(f"Failed to initialize Qdrant client or access collection '{QDRANT_COLLECTION_NAME}': {e}")
        print("Please ensure Qdrant is running, accessible, and the collection exists with the correct embeddings.")
        exit(1)

def process_query(user_input_query, ollama_model):
    if user_input_query.strip():
        # --- 1. Chunk-level search ---
        q_client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY, timeout=30)
        chunk_results = search_similar_embeddings(
            user_query=user_input_query,
            qdrant_client=q_client,
            collection_name=QDRANT_COLLECTION_NAME,
            ollama_model_for_query=ollama_model,
            top_k=3
        )
        if chunk_results:
            print(f"\n--- Top {len(chunk_results)} SIMILAR CHUNKS for model '{ollama_model}' ---")
            for i, hit in enumerate(chunk_results): # hit is ScoredPoint
                print(f"\nChunk Result {i+1}:")
                print(f"  ID: {hit.id}, Score: {hit.score:.4f}")
                if hit.payload:
                    print(f"  Source File: {hit.payload.get('source_file', 'N/A')}")
                    print(f"  Model Stored: {hit.payload.get('model_name', 'N/A')}")
                    chunk_text = hit.payload.get('chunk_text', '')
                    print(f"  Text: \"{chunk_text[:150].strip()}...\"")
        else:
            print("No similar chunks found or an error occurred during chunk search.")

        # --- 2. File-level search (averaged chunks) ---
        file_results_avg = search_average_similarity_per_file(
            user_query=user_input_query,
            qdrant_client=q_client,
            collection_name=QDRANT_COLLECTION_NAME,
            ollama_model_name=ollama_model, 
            top_n_files=3,
            normalize_chunks_before_averaging=True
        )
        if file_results_avg:
            print(f"\n--- Top {len(file_results_avg)} SIMILAR FILES (by averaged chunks) for model '{ollama_model}' ---")
            for i, (filename, score, num_chunks) in enumerate(file_results_avg):
                print(f"\nFile Result {i+1}:")
                print(f"  Source File: {filename}")
                print(f"  Avg. Similarity Score: {score:.4f}")
                print(f"  (Based on {num_chunks} chunks)")
        else:
            print("No similar files found by averaging or an error occurred.")
    else:
        print("No query entered. Exiting.")

def plot_writers(ollama_model):
    q_client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY, timeout=30)
    #plot_tsne_of_embeddings(q_client, QDRANT_COLLECTION_NAME, ollama_model, max_points=5000)
    #plot_tsne_by_author(q_client, QDRANT_COLLECTION_NAME, ollama_model, max_points=500)
    #plot_tsne_of_authors_avg(q_client, QDRANT_COLLECTION_NAME, ollama_model, max_points=500)  
    plot_tsne_by_author_allowed(q_client, QDRANT_COLLECTION_NAME, ollama_model, max_points=500) 
    # qdrant_client.create_payload_index(
    #     collection_name=QDRANT_COLLECTION_NAME,
    #     field_name="model_name",
    #     field_schema="keyword"
    # )

# --- Main execution logic ---
options = ["query", "process", "plot"]
models = {"mistral": "mistral",
          "bielik": "mwiewior/bielik",
          "pllum": "antoniprzybylik/llama-pllum:8b"}

if __name__ == "__main__":
    if QDRANT_CLOUD_URL is None:
        print("Error: QDRANT_CLOUD_URL environment variable must be set.")
        exit(1)
    if QDRANT_API_KEY is None:
        print("Error: QDRANT_API_KEY environment variable must be set.")
        exit(1)

    args = sys.argv[1:]
    if len(args) == 0 or args[1] not in options or args[0] not in models:
        print("Provide model as first argument and mode as second argument.")
        print("Available modes:")
        for opt in options:
            print(f"  - {opt}")
        print("Available models:")
        for model in models.keys():
            print(f"  - {model}")
        exit(1)
    mode = args[1]
    model = models[args[0]]

    if mode == "process":
        process_texts(model)

    if mode == "query":
        if len(args) < 3:
            print("Provide a query with quotation")
            exit(1)
        process_query(args[2], model)

    if mode == "plot":
        plot_writers(model)
 