import requests
import json
import os
import numpy as np
from qdrant_client import QdrantClient, models

OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_API_URL = f"{OLLAMA_API_BASE_URL}/api/embeddings"

def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_dim: int, current_ollama_model_name: str, recreate_if_needed: bool = False):
    print(f"Setting up Qdrant collection '{collection_name}' with vector dimension {vector_dim}...")
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
        existing_vector_size = None
        if isinstance(collection_info.config.params.vectors, models.VectorParams):
            existing_vector_size = collection_info.config.params.vectors.size
        elif isinstance(collection_info.config.params.vectors, dict): # Named vectors
            if "" in collection_info.config.params.vectors: # Default unnamed vector
                 existing_vector_size = collection_info.config.params.vectors[""].size
            elif collection_info.config.params.vectors: # First named vector
                 existing_vector_size = next(iter(collection_info.config.params.vectors.values())).size


        if existing_vector_size and existing_vector_size != vector_dim:
            print(f"CRITICAL MISMATCH: Collection '{collection_name}' exists with vector size {existing_vector_size}, "
                  f"but current model '{current_ollama_model_name}' produces embeddings of size {vector_dim}.")
            if recreate_if_needed:
                print(f"Recreating collection '{collection_name}' with new dimension {vector_dim}.")
                client.delete_collection(collection_name=collection_name)
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
                )
                print(f"Collection '{collection_name}' recreated.")
            else:
                print("This will lead to errors during upsert. Please use a different collection name, "
                      "ensure your model matches the collection, or enable recreation.")
                exit(1)
        else:
            print(f"Collection vector size {existing_vector_size or 'N/A (no default vector)'} is compatible or collection is new.")


    except Exception as e: # Catches Qdrant errors if collection doesn't exist or other API issues
        if "not found" in str(e).lower() or ("status_code=404" in str(e).lower() if hasattr(e, "status_code") else False):
            print(f"Collection '{collection_name}' not found. Creating...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Error during Qdrant collection check/setup for '{collection_name}': {e}")
            print("Please check your Qdrant connection, API key, and permissions.")
            exit(1)

def get_ollama_embeddings(text: str, model_name: str, api_url: str):# -> np.ndarray | None:
    try:
        payload = {"model": model_name, "prompt": text}
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        response_json = response.json()
        if "embedding" not in response_json:
            raise ValueError(f"Ollama API response for model '{model_name}' no 'embedding'. Resp: {response_json}")
        embedding_1d = np.array(response_json['embedding'], dtype=np.float32)
        return embedding_1d.reshape(1, -1)
    except requests.exceptions.RequestException as e:
        print(f"Error Ollama API ({api_url}) model '{model_name}': {e.response.text if hasattr(e,'response') and e.response else e}")
        return None
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"Error processing Ollama API response model '{model_name}': {e}")
        return None

def check_ollama_status_and_get_dim(model_to_check: str):
    global VECTOR_DIMENSION
    print(f"Checking Ollama status at {OLLAMA_API_BASE_URL} for model '{model_to_check}'...")
    try:
        response = requests.get(OLLAMA_API_BASE_URL, timeout=10)
        response.raise_for_status()
        print(f"Ollama server appears to be running at {OLLAMA_API_BASE_URL}.")
    except requests.exceptions.RequestException as e:
        print(f"Error: Ollama server not reachable at {OLLAMA_API_BASE_URL}. Ensure Ollama is running.")
        print(f"Details: {e}")
        exit(1)

    try:
        response = requests.post(f"{OLLAMA_API_BASE_URL}/api/show", json={"name": model_to_check}, timeout=10)
        if response.status_code == 404:
            print(f"Error: Ollama model '{model_to_check}' not found. Pull or create it (e.g., 'ollama pull {model_to_check}').")
            exit(1)
        response.raise_for_status()
        print(f"Ollama model '{model_to_check}' is available.")

        print(f"Determining vector dimension for '{model_to_check}'...")
        sample_embedding = get_ollama_embeddings("sample text", model_to_check, OLLAMA_EMBED_API_URL)
        if sample_embedding is not None and sample_embedding.ndim == 2 and sample_embedding.shape[0] == 1:
            new_dimension = sample_embedding.shape[1]
            if VECTOR_DIMENSION is not None and VECTOR_DIMENSION != new_dimension:
                print(f"Warning: The new model '{model_to_check}' (dim: {new_dimension}) has a different dimension "
                      f"than previously assumed for the collection (dim: {VECTOR_DIMENSION}).")
                print("If you are using the same Qdrant collection, this will cause issues unless the collection "
                      "was created with the new dimension or can accommodate multiple vector configurations (advanced).")
                # Decide on a strategy: exit, use new dim and warn, or require new collection.
                # For now, we'll proceed but this is a critical point for collection management.
            VECTOR_DIMENSION = new_dimension
            print(f"Determined vector dimension for '{model_to_check}': {VECTOR_DIMENSION}")
        else:
            print(f"Error: Could not determine vector dimension for '{model_to_check}'. Exiting.")
            print(f"Sample embedding received: {sample_embedding}")
            exit(1)

    except requests.exceptions.RequestException as e:
        print(f"Error during Ollama model check for '{model_to_check}': {e}")
        exit(1)
    print("-" * 30)
