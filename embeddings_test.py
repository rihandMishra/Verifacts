# ================================
# Embeddings Validation Test File
# ================================

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# ----------------
# Device selection
# ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------
# Step 2.1: Load multilingual model
# ------------------------------------
print("Loading multilingual claim model...")
claim_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
print("Claim model loaded successfully")

# ------------------------------------
# Step 2.2: Load medical-domain model
# ------------------------------------
print("Loading medical domain model...")
medical_model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

medical_tokenizer = AutoTokenizer.from_pretrained(medical_model_name)
medical_model = AutoModel.from_pretrained(medical_model_name)
medical_model.to(device)
medical_model.eval()

print("Medical model loaded successfully")

# ------------------------------------
# Step 2.3: Define test claims
# ------------------------------------
claims = [
    "Vaccines cause autism",                 # 0
    "टीके से ऑटिज़्म होता है",               # 1 (same meaning, Hindi)
    "Vaccines cause infertility",            # 2 (same domain)
    "Drinking turmeric cures diabetes",      # 3 (different domain)
    "Polio drops are harmful"                # 4 (related category)
]

print("Test claims loaded")

# ------------------------------------
# Step 2.4: Embedding functions
# ------------------------------------
def embed_medical(text: str):
    inputs = medical_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = medical_model(**inputs)
    # Mean pooling over token embeddings
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Generate embeddings
claim_embeddings = claim_model.encode(claims)
medical_embeddings = np.vstack([embed_medical(c) for c in claims])

print("Claim text embeddings shape:", claim_embeddings.shape)
print("Medical embeddings shape:", medical_embeddings.shape)

# ------------------------------------
# Step 2.5: Similarity validation
# ------------------------------------
print("\n=== CLAIM TEXT COSINE SIMILARITY ===")
print(np.round(cosine_similarity(claim_embeddings), 3))

print("\n=== MEDICAL CONTEXT COSINE SIMILARITY ===")
print(np.round(cosine_similarity(medical_embeddings), 3))
