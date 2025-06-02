from typing import List, Dict, Tuple
from database import get_ollama_embeddings, OLLAMA_EMBED_API_URL
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct, ScoredPoint
import numpy as np

def cosine_similarity_np(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.ndim > 1: vec1 = vec1.flatten()
    if vec2.ndim > 1: vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def search_average_similarity_per_file(
    user_query,
    qdrant_client,
    collection_name,
    ollama_model_name, # Model for query embedding AND for filtering chunks
    top_n_files = 3,
    normalize_chunks_before_averaging = True
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
    