from transformers import AutoTokenizer, AutoModel
import torch
import os


HF_TOKEN = os.getenv("HF_TOKEN")
model_name = "speakleash/Bielik-4.5B-v3.0-Instruct" 

from transformers import AutoTokenizer, AutoModelForCausalLM

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


sentences_pl = [
    "Jak stworzyć lokalne embeddingi używając polskiego LLM BIELIK?",
    "Sztuczna inteligencja rewolucjonizuje sposób, w jaki pracujemy.",
    "Polska ma bogatą historię i kulturę.",
    "Model BIELIK został opracowany przez Allegro."
]


def get_decoder_only_embeddings(texts, tokenizer, model, pooling_strategy="mean"):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
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

print("Generating embeddings...")
embeddings_llm = get_decoder_only_embeddings(sentences_pl, tokenizer, model, pooling_strategy="mean")


print(f"Shape of LLM embeddings (mean pooling): {embeddings_llm.shape}")
print("First LLM embedding vector (mean pooling, first 5 dimensions):", embeddings_llm[0][:5])