from transformers import AutoTokenizer, AutoModel
import torch
import os


HF_TOKEN = os.getenv("HF_TOKEN")
model_name = "speakleash/Bielik-4.5B-v3.0-Instruct" 

from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,  
                                    
        device_map="auto",          
        output_hidden_states=True   

    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have 'accelerate' installed (pip install accelerate).")
    print("Ensure you have enough VRAM/RAM and disk space.")
    print("Verify your HF_TOKEN and that you've accepted model terms on Hugging Face Hub.")
    exit()


model.eval() 

DATA_DIR = "data/"
CHUNK_SIZE_WORDS = 500 
CHUNK_OVERLAP_WORDS = 30 
BATCH_SIZE_EMBED = 8    
MAX_TOKENS_PER_CHUNK = 512

all_chunk_texts = []
all_chunk_embeddings = []


def get_decoder_only_embeddings(texts, tokenizer, model, pooling_strategy="mean"):
    inputs = tokenizer(
        texts,
        padding=True,
        return_tensors="pt"
    ).to(model.device if hasattr(model, 'device') else "cpu") # model.device if not using device_map fully

    with torch.no_grad():
        outputs = model(**inputs) 
        last_hidden_states = outputs.hidden_states[-1]

    if pooling_strategy == "mean":
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
    elif pooling_strategy == "last":
        sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1 
        batch_size = last_hidden_states.shape[0]
        sentence_embeddings = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths, :]
    else:
        raise ValueError(f"Unsupported pooling_strategy: {pooling_strategy}")

    return sentence_embeddings.cpu().numpy()

source_texts = ["balladyna.txt","kordian.txt","pan-tadeusz.txt","dziady-dziady-poema-dziady-czesc-iii.txt"]


def do_everything():
    for source_text in source_texts:
        with open(os.path.join(DATA_DIR, source_text), "r", encoding="utf-8") as f:
            text = f.read()


        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i + CHUNK_SIZE_WORDS]
            chunks.append(" ".join(chunk))
            if i + CHUNK_SIZE_WORDS >= len(words):
                break
            i += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

        sentences_pl = chunks




        print("Generating embeddings...")
        for chunk in sentences_pl:
            vector = []
            embeddings_llm = get_decoder_only_embeddings(chunk, tokenizer, model, pooling_strategy="mean")
            vector.append(embeddings_llm)

        vector_db = []
        vector_db.append(vector)


        with open(os.path.join(DATA_DIR, f"vector_db_{source_text}.pkl"), "wb") as f:
            pickle.dump(vector_db, f)


#do_everything()
def load_embeddings():
    all_chunk_texts = []
    all_chunk_embeddings = []

    for source_text in source_texts:
        with open(os.path.join(DATA_DIR, f"vector_db_{source_text}.pkl"), "rb") as f:
            vector_db = pickle.load(f)
            for vector in vector_db:
                all_chunk_embeddings.append(vector)

    return all_chunk_embeddings

all_chunk_embeddings = load_embeddings()
print(all_chunk_embeddings)
