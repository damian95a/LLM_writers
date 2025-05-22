import requests
import numpy as np
import os
import pickle
import json 


OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_API_URL = f"{OLLAMA_API_BASE_URL}/api/embeddings"


#OLLAMA_MODEL_NAME = "mistral" 
OLLAMA_MODEL_NAME = "mwiewior/bielik"
#OLLAMA_MODEL_NAME = "antoniprzybylik/llama-pllum:8b"


DATA_DIR = "data/"
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 30


source_texts = ["balladyna.txt","kordian.txt","pan-tadeusz.txt","dziady-dziady-poema-dziady-czesc-iii.txt"]


def check_ollama_status():
    print(f"Checking Ollama status at {OLLAMA_API_BASE_URL} for model '{OLLAMA_MODEL_NAME}'...")
    try:
        response = requests.get(OLLAMA_API_BASE_URL) 
        response.raise_for_status()
        if "ollama is running" not in response.text.lower():
             print(f"Warning: Ollama is reachable at {OLLAMA_API_BASE_URL} but returned an unexpected message: {response.text[:100]}...")
        print(f"Ollama server appears to be running at {OLLAMA_API_BASE_URL}.")
    except requests.exceptions.RequestException as e:
        print(f"Error: Ollama server not reachable at {OLLAMA_API_BASE_URL}. Please ensure Ollama is running.")
        print(f"Details: {e}")
        exit(1)

    try:
        response = requests.post(f"{OLLAMA_API_BASE_URL}/api/show", json={"name": OLLAMA_MODEL_NAME})
        if response.status_code == 404:
            print(f"Error: Ollama model '{OLLAMA_MODEL_NAME}' not found on the Ollama server.")
            print(f"Please pull or create the model in Ollama (e.g., 'ollama pull {OLLAMA_MODEL_NAME}' or 'ollama create {OLLAMA_MODEL_NAME} ...').")
            print(f"Available models can be listed with 'ollama list'.")
            exit(1)
        response.raise_for_status()
        print(f"Ollama model '{OLLAMA_MODEL_NAME}' is available.")
    except requests.exceptions.RequestException as e:
        print(f"Error checking Ollama model '{OLLAMA_MODEL_NAME}'.")
        print(f"Response status: {response.status_code if 'response' in locals() and hasattr(response, 'status_code') else 'N/A'}")
        print(f"Response text: {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}")
        print(f"Details: {e}")
        exit(1)
    print("-" * 30)


def get_ollama_embeddings(text: str, model_name: str, api_url: str) -> np.ndarray:

    try:
        payload = {
            "model": model_name,
            "prompt": text
        }
        response = requests.post(api_url, json=payload)
        response.raise_for_status() 
        
        response_json = response.json()
        if "embedding" not in response_json:
            
            raise ValueError(f"Ollama API response for model '{model_name}' does not contain 'embedding' field. Response: {response_json}")
            
        embedding_1d = np.array(response_json['embedding'], dtype=np.float32)
        return embedding_1d.reshape(1, -1) 
        
    except requests.exceptions.RequestException as e:
       
        err_msg = f"Error calling Ollama API ({api_url}) for model '{model_name}'"
        if hasattr(e, 'response') and e.response is not None:
            err_msg += f"\nStatus: {e.response.status_code}, Response: {e.response.text[:500]}"
        else:
            err_msg += f"\nDetails: {e}"
        print(err_msg)
        raise 
    except (json.JSONDecodeError, ValueError, TypeError) as e:

        err_msg = f"Error processing Ollama API response for model '{model_name}'"
        if 'response' in locals() and hasattr(response, 'text'):
            err_msg += f"\nResponse text: {response.text[:500]}"
        err_msg += f"\nDetails: {e}"
        print(err_msg)
        raise 


def do_everything():

    check_ollama_status()

    for source_text_filename in source_texts:
        filepath = os.path.join(DATA_DIR, source_text_filename)
        if not os.path.exists(filepath):
            print(f"Warning: Source text file not found: {filepath}. Skipping.")
            continue
        
        print(f"\nProcessing file: {source_text_filename}")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + CHUNK_SIZE_WORDS]
            chunks.append(" ".join(chunk_words))
            if i + CHUNK_SIZE_WORDS >= len(words):
                break # All words processed
            i += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
        
        processed_chunks = chunks 

        print(f"Generating embeddings for {len(processed_chunks)} chunks from {source_text_filename} using Ollama model '{OLLAMA_MODEL_NAME}'...")

        vector_db_for_file = [] 
        
        for idx, chunk_text in enumerate(processed_chunks):

            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(processed_chunks) -1 : 
                 print(f"  Generating embedding for chunk {idx + 1}/{len(processed_chunks)}...")
            try:
                embedding_array = get_ollama_embeddings(chunk_text, OLLAMA_MODEL_NAME, OLLAMA_EMBED_API_URL)
                vector_db_for_file.append([embedding_array]) 
            except Exception as e:
                print(f"    Failed to get embedding for chunk {idx+1} of {source_text_filename}. Skipping this chunk.")
                print(f"    Chunk content (first 100 chars): {chunk_text[:100]}...")
                continue 
        
        if not vector_db_for_file:
            print(f"Warning: No embeddings were generated for {source_text_filename}. The .pkl file will be empty or not created if all chunks failed.")
        
        safe_model_dir_name = OLLAMA_MODEL_NAME.replace(":", "_").replace("/", "_")
        model_dir = os.path.join(DATA_DIR, safe_model_dir_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        output_pkl_path = os.path.join(model_dir, f"vector_db_{source_text_filename}.pkl")
        with open(output_pkl_path, "wb") as f:
            pickle.dump(vector_db_for_file, f)
        print(f"Embeddings for {source_text_filename} (found {len(vector_db_for_file)} valid embeddings) saved to {output_pkl_path}")


def load_embeddings():
    all_loaded_chunk_embeddings = []

    print("\nLoading embeddings from .pkl files...")
    safe_model_dir_name = OLLAMA_MODEL_NAME.replace(":", "_").replace("/", "_")
    model_dir = os.path.join(DATA_DIR, safe_model_dir_name)
    print(f"Model directory: {model_dir}")
    for source_text_filename in source_texts:
        pkl_path = os.path.join(model_dir, f"vector_db_{source_text_filename}.pkl")
        if not os.path.exists(pkl_path):
            print(f"Warning: Pickle file not found for {source_text_filename}: {pkl_path}. Skipping.")
            continue
        try:
            with open(pkl_path, "rb") as f:
                vector_db_from_file = pickle.load(f)
                for item_list_containing_one_embedding in vector_db_from_file:
                    all_loaded_chunk_embeddings.append(item_list_containing_one_embedding)
                print(f"  Loaded {len(vector_db_from_file)} embeddings from {pkl_path}")
        except Exception as e:
            print(f"Error loading embeddings from {pkl_path}: {e}")
            continue

    return all_loaded_chunk_embeddings


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")

    for st_file in source_texts:
        st_path = os.path.join(DATA_DIR, st_file)
        if not os.path.exists(st_path):
            print(f"Creating dummy file: {st_path} something goes wrong")

    print("Running do_everything() to generate embeddings using Ollama...")
    print("IMPORTANT: Ensure OLLAMA_MODEL_NAME is correctly set and the model is available in Ollama.")
    do_everything()
    print("Finished do_everything().")

    final_all_chunk_embeddings = load_embeddings()
    
    print(f"\nTotal loaded chunk embedding entries: {len(final_all_chunk_embeddings)}")
    if final_all_chunk_embeddings:
        print("Example of the first loaded chunk embedding entry (it's a list containing one embedding array):")
        print(final_all_chunk_embeddings[0])
        if final_all_chunk_embeddings[0] and \
           isinstance(final_all_chunk_embeddings[0], list) and \
           len(final_all_chunk_embeddings[0]) > 0 and \
           isinstance(final_all_chunk_embeddings[0][0], np.ndarray):
             print(f"Shape of the actual embedding numpy array inside: {final_all_chunk_embeddings[0][0].shape}")
        else:
            print("The first loaded item does not conform to the expected structure: list([numpy.ndarray]).")
            print(f"Content of first item: {final_all_chunk_embeddings[0]}")
    else:
        print("No embeddings were loaded. Ensure 'do_everything()' has run successfully and created .pkl files, or that .pkl files exist from a previous run.")