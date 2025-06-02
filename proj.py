import requests
import numpy as np
import os
import sys
import pickle
import json 
import uuid
import dotenv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
dotenv.load_dotenv()

from typing import List, Dict, Tuple
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct, ScoredPoint

try:
    from qdrant_client.http.models import ScrollRequest, ScrollResponse
except ImportError:

    class ScrollRequest: pass # Placeholder
    class ScrollResponse: # Placeholder
        def __init__(self, points, next_page_offset):
            self.points = points
            self.next_page_offset = next_page_offset

OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_API_URL = f"{OLLAMA_API_BASE_URL}/api/embeddings"

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


def search_similar_embeddings(user_query: str,
                              qdrant_client: QdrantClient,
                              collection_name: str,
                              ollama_model_for_query: str,
                              top_k: int = 5):

    print(f"\nSearching for text similar to: \"{user_query[:100]}...\"")
    print(f"Using Ollama model '{ollama_model_for_query}' for query embedding.")
    print(f"Filtering results in Qdrant collection '{collection_name}' for payload.model_name = '{ollama_model_for_query}'.")

    # 1. Embed the user query
    query_embedding_2d = get_ollama_embeddings(user_query, ollama_model_for_query, OLLAMA_EMBED_API_URL)

    if query_embedding_2d is None:
        print("Failed to generate embedding for the user query. Cannot perform search.")
        return []

    query_vector_1d = query_embedding_2d[0].tolist()  # Qdrant expects a flat list

    # 2. Define the filter for Qdrant
    query_filter = Filter(
        must=[
            FieldCondition(
                key="model_name", 
                match=MatchValue(value=ollama_model_for_query)
            )
        ]
    )

    # 3. Użyj query_points zamiast search
    try:
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector_1d,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        return search_results
    except Exception as e:
        print(f"Error during Qdrant search: {e}")
        return []
    
def cosine_similarity_np(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.ndim > 1: vec1 = vec1.flatten()
    if vec2.ndim > 1: vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def normalize_vector_np(vec: np.ndarray) -> np.ndarray:
    if vec.ndim > 1: vec = vec.flatten()
    norm = np.linalg.norm(vec)
    if norm == 0: return vec
    return vec / norm


def search_average_similarity_per_file(
    user_query: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    ollama_model_name: str, # Model for query embedding AND for filtering chunks
    top_n_files: int = 3,
    normalize_chunks_before_averaging: bool = True
) -> List[Tuple[str, float, int]]: # Returns (source_file, similarity_score, num_chunks)
    print(f"\nSearching FILES (by averaged chunks) similar to: \"{user_query[:100].strip()}...\"")
    print(f"Using Ollama model '{ollama_model_name}' for query and filtering stored chunks.")
    if normalize_chunks_before_averaging:
        print("Note: Chunk embeddings will be L2 normalized before averaging.")

    # 1. Embed the user query
    query_embedding_2d = get_ollama_embeddings(user_query, ollama_model_name, OLLAMA_EMBED_API_URL)
    if query_embedding_2d is None or query_embedding_2d.size == 0:
        print("Failed to generate embedding for the user query. Cannot proceed.")
        return []
    query_vector_1d_np = query_embedding_2d[0] # Keep as 1D NumPy array for calculations

    # 2. Fetch all relevant chunk embeddings and their source_file
    points_by_source_file: Dict[str, List[np.ndarray]] = {}
    total_fetched_points = 0
    print(f"Fetching all chunks for model '{ollama_model_name}' from '{collection_name}'...")

    scroll_filter_conditions = [
        FieldCondition(key="model_name", match=MatchValue(value=ollama_model_name))
    ]

    
    current_offset = None
    scroll_limit = 250 

    try:
        while True:
            points, next_page_offset = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(must=scroll_filter_conditions),
                limit=scroll_limit,
                offset=current_offset,
                with_payload=["source_file"],
                with_vectors=True
            )
            if not points:
                if current_offset is None and total_fetched_points == 0:
                    print(f"No points found matching filter for model '{ollama_model_name}'.")
                break

            for point_record in points:
                total_fetched_points += 1
                if point_record.payload and 'source_file' in point_record.payload and point_record.vector:
                    source_file = point_record.payload['source_file']
                    
                    # Handle how vector is stored (list or dict for named vectors)
                    vector_data = point_record.vector
                    if isinstance(vector_data, dict): # Named vectors
                        chunk_vec_np = np.array(vector_data.get("", []), dtype=np.float32) # Assuming default unnamed vector ""
                    elif isinstance(vector_data, list): # Default unnamed vector
                        chunk_vec_np = np.array(vector_data, dtype=np.float32)
                    else:
                        print(f"Warning: Unexpected vector format for point {point_record.id}. Skipping.")
                        continue
                    
                    if chunk_vec_np.size == 0:
                        print(f"Warning: Empty vector for point {point_record.id}. Skipping.")
                        continue

                    if source_file not in points_by_source_file:
                        points_by_source_file[source_file] = []
                    points_by_source_file[source_file].append(chunk_vec_np)

            current_offset = next_page_offset
            if current_offset is None:
                break
            print(f"  Fetched {total_fetched_points} points so far for model '{ollama_model_name}'...")
            print(f"Total points fetched and processed for model '{ollama_model_name}': {total_fetched_points}")
        if not points_by_source_file:
            print("No data to average after fetching points.")
            return []

    except Exception as e:
        print(f"Error during Qdrant scroll operation: {e}")
        return []

    # 3. Calculate average vector for each source_file
    averaged_embeddings_per_file: Dict[str, Tuple[np.ndarray, int]] = {}
    print("Averaging embeddings per source file...")
    for source_file, vectors_list in points_by_source_file.items():
        if not vectors_list:
            continue
        
        stacked_vectors = np.array(vectors_list) # Shape: (num_chunks, vector_dimension)
        
        if normalize_chunks_before_averaging:
            norms = np.linalg.norm(stacked_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9 
            normalized_stacked_vectors = stacked_vectors / norms
            avg_vector = np.mean(normalized_stacked_vectors, axis=0)
        else:
            avg_vector = np.mean(stacked_vectors, axis=0)
        

        averaged_embeddings_per_file[source_file] = (avg_vector, len(vectors_list))
        print(f"  Averaged {len(vectors_list)} chunks for '{source_file}'. Avg vector shape: {avg_vector.shape}")


    if not averaged_embeddings_per_file:
        print("No averaged embeddings could be calculated.")
        return []

    # 4. Calculate similarity score between query and each averaged file embedding
    file_similarity_scores: List[Tuple[str, float, int]] = []
    print("Calculating similarity scores against averaged file embeddings...")
    for source_file, (avg_file_vector, num_chunks) in averaged_embeddings_per_file.items():
        similarity = cosine_similarity_np(query_vector_1d_np, avg_file_vector)
        file_similarity_scores.append((source_file, similarity, num_chunks))

    # 5. Sort by similarity score (descending)
    file_similarity_scores.sort(key=lambda item: item[1], reverse=True)

    return file_similarity_scores[:top_n_files]




def fetch_all_embeddings_for_tsne(qdrant_client: QdrantClient, collection_name: str, model_name: str, max_points: int = 1000):
    embeddings = []
    labels = []
    scroll_filter_conditions = [
        FieldCondition(key="model_name", match=MatchValue(value=model_name))
    ]
    current_offset = None
    scroll_limit = 250
    total = 0

    while total < max_points:
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=scroll_filter_conditions),
            limit=min(scroll_limit, max_points - total),
            offset=current_offset,
            with_payload=["source_file"],
            with_vectors=True
        )
        if not points:
            break
        for point in points:
            vector_data = point.vector
            if isinstance(vector_data, dict):
                vec = np.array(vector_data.get("", []), dtype=np.float32)
            else:
                vec = np.array(vector_data, dtype=np.float32)
            if vec.size == 0:
                continue
            embeddings.append(vec)
            labels.append(point.payload.get("source_file", "unknown"))
            total += 1
            if total >= max_points:
                break
        current_offset = next_page_offset
        if current_offset is None:
            break
    return np.array(embeddings), labels

def plot_tsne_of_embeddings(qdrant_client: QdrantClient, collection_name: str, model_name: str, max_points: int = 1000):
    print(f"Pobieranie embeddingów do t-SNE (max {max_points})...")
    X, labels = fetch_all_embeddings_for_tsne(qdrant_client, collection_name, model_name, max_points)
    if X.shape[0] == 0:
        print("Brak embeddingów do wizualizacji.")
        return
    print(f"Redukcja wymiarów t-SNE ({X.shape[0]} punktów)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    # Przypisz kolor do każdego unikalnego source_file
    unique_labels = list(sorted(set(labels)))
    color_map = {label: idx for idx, label in enumerate(unique_labels)}
    colors = [color_map[label] for label in labels]

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='tab20', alpha=0.7)

    # Dodaj legendę z nazwami plików
    handles = []
    for label, idx in color_map.items():
        handles.append(plt.Line2D([], [], marker="o", color='w', markerfacecolor=plt.cm.tab20(idx / max(1, len(unique_labels)-1)), label=label, markersize=8))
    plt.legend(handles=handles, title="source_file", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Opcjonalnie: wyświetl etykiety plików dla pierwszych N punktów
    for i in range(min(30, len(labels))):
        plt.annotate(labels[i], (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)
    plt.title(f"t-SNE embeddingów z Qdrant ({model_name})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()


def plot_tsne_by_author(qdrant_client: QdrantClient, collection_name: str, model_name: str, max_points: int = 1000):
    print(f"Pobieranie embeddingów do t-SNE (max {max_points})...")
    embeddings = []
    authors = []
    scroll_filter_conditions = [
        FieldCondition(key="model_name", match=MatchValue(value=model_name))
    ]
    current_offset = None
    scroll_limit = 250
    total = 0

    while total < max_points:
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=scroll_filter_conditions),
            limit=min(scroll_limit, max_points - total),
            offset=current_offset,
            with_payload=["author"],
            with_vectors=True
        )
        if not points:
            break
        for point in points:
            vector_data = point.vector
            if isinstance(vector_data, dict):
                vec = np.array(vector_data.get("", []), dtype=np.float32)
            else:
                vec = np.array(vector_data, dtype=np.float32)
            if vec.size == 0:
                continue
            embeddings.append(vec)
            authors.append(point.payload.get("author", "unknown"))
            total += 1
            if total >= max_points:
                break
        current_offset = next_page_offset
        if current_offset is None:
            break

    X = np.array(embeddings)
    if X.shape[0] == 0:
        print("Brak embeddingów do wizualizacji.")
        return

    print(f"Redukcja wymiarów t-SNE ({X.shape[0]} punktów)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    # Przypisz kolor do każdego unikalnego autora
    unique_authors = list(sorted(set(authors)))
    color_map = {author: idx for idx, author in enumerate(unique_authors)}
    colors = [color_map[author] for author in authors]

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='tab20', alpha=0.7)

    # Dodaj legendę z nazwami autorów
    handles = []
    for author, idx in color_map.items():
        handles.append(plt.Line2D([], [], marker="o", color='w', markerfacecolor=plt.cm.tab20(idx / max(1, len(unique_authors)-1)), label=author, markersize=8))
    plt.legend(handles=handles, title="author", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Opcjonalnie: wyświetl etykiety autorów dla pierwszych N punktów
    for i in range(min(30, len(authors))):
        plt.annotate(authors[i], (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)
    plt.title(f"t-SNE embeddingów z Qdrant ({model_name}) wg autora")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

def plot_tsne_of_authors_avg(qdrant_client: QdrantClient, collection_name: str, model_name: str, max_points: int = 1000):
    """
    Średnia embeddingów dla każdego autora i wizualizacja t-SNE (każdy punkt to autor).
    """
    print(f"Pobieranie embeddingów do t-SNE (max {max_points})...")
    # 1. Pobierz embeddingi i autorów
    author_vectors: Dict[str, list] = {}
    scroll_filter_conditions = [
        FieldCondition(key="model_name", match=MatchValue(value=model_name))
    ]
    current_offset = None
    scroll_limit = 250
    total = 0

    while total < max_points:
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=scroll_filter_conditions),
            limit=min(scroll_limit, max_points - total),
            offset=current_offset,
            with_payload=["author"],
            with_vectors=True
        )
        if not points:
            break
        for point in points:
            vector_data = point.vector
            if isinstance(vector_data, dict):
                vec = np.array(vector_data.get("", []), dtype=np.float32)
            else:
                vec = np.array(vector_data, dtype=np.float32)
            if vec.size == 0:
                continue
            author = point.payload.get("author", "unknown")
            if author not in author_vectors:
                author_vectors[author] = []
            author_vectors[author].append(vec)
            total += 1
            if total >= max_points:
                break
        current_offset = next_page_offset
        if current_offset is None:
            break

    # 2. Oblicz średnie embeddingi dla każdego autora
    avg_embeddings = []
    author_labels = []
    for author, vectors in author_vectors.items():
        stacked = np.stack(vectors)
        avg_vec = np.mean(stacked, axis=0)
        avg_embeddings.append(avg_vec)
        author_labels.append(author)

    X = np.array(avg_embeddings)
    if X.shape[0] == 0:
        print("Brak embeddingów do wizualizacji.")
        return

    print(f"Redukcja wymiarów t-SNE ({X.shape[0]} autorów)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
    X_2d = tsne.fit_transform(X)

    # Kolory dla autorów
    unique_authors = list(sorted(set(author_labels)))
    color_map = {author: idx for idx, author in enumerate(unique_authors)}
    colors = [color_map[author] for author in author_labels]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='tab20', alpha=0.8, s=120)

    # Legenda
    handles = []
    for author, idx in color_map.items():
        handles.append(plt.Line2D([], [], marker="o", color='w', markerfacecolor=plt.cm.tab20(idx / max(1, len(unique_authors)-1)), label=author, markersize=10))
    plt.legend(handles=handles, title="author", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Etykiety autorów
    for i, author in enumerate(author_labels):
        plt.annotate(author, (X_2d[i, 0], X_2d[i, 1]), fontsize=10, alpha=0.8)
    plt.title(f"t-SNE średnich embeddingów autorów ({model_name})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()


def plot_tsne_by_author_allowed(qdrant_client: QdrantClient, collection_name: str, model_name: str, max_points: int = 1000):
    print(f"Pobieranie embeddingów do t-SNE (max {max_points})...")
    embeddings = []
    authors = []
    allowed_authors = {"sienkiewicz_henryk", "słowacki_juliusz"} 
    scroll_filter_conditions = [
        FieldCondition(key="model_name", match=MatchValue(value=model_name))
    ]
    current_offset = None
    scroll_limit = 250
    total = 0

    while total < max_points:
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=scroll_filter_conditions),
            limit=min(scroll_limit, max_points - total),
            offset=current_offset,
            with_payload=["author"],
            with_vectors=True
        )
        if not points:
            break
        for point in points:
            author = point.payload.get("author", "unknown")
            if author not in allowed_authors:
                continue  
            vector_data = point.vector
            if isinstance(vector_data, dict):
                vec = np.array(vector_data.get("", []), dtype=np.float32)
            else:
                vec = np.array(vector_data, dtype=np.float32)
            if vec.size == 0:
                continue
            embeddings.append(vec)
            authors.append(author)
            total += 1
            if total >= max_points:
                break
        current_offset = next_page_offset
        if current_offset is None:
            break

    X = np.array(embeddings)
    if X.shape[0] == 0:
        print("Brak embeddingów do wizualizacji.")
        return

    print(f"Redukcja wymiarów t-SNE ({X.shape[0]} punktów)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    unique_authors = list(sorted(set(authors)))
    color_map = {author: idx for idx, author in enumerate(unique_authors)}
    colors = [color_map[author] for author in authors]

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='tab20', alpha=0.7)

    handles = []
    for author, idx in color_map.items():
        handles.append(plt.Line2D([], [], marker="o", color='w', markerfacecolor=plt.cm.tab20(idx / max(1, len(unique_authors)-1)), label=author, markersize=8))
    plt.legend(handles=handles, title="author", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    for i in range(min(30, len(authors))):
        plt.annotate(authors[i], (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)
    plt.title(f"t-SNE embeddingów z Qdrant ({model_name}) wg autora (tylko Sienkiewicz i Słowacki)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()



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
 