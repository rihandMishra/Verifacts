from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from datetime import datetime

# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load embedding models
# -----------------------------
claim_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

medical_model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
medical_tokenizer = AutoTokenizer.from_pretrained(medical_model_name)
medical_model = AutoModel.from_pretrained(medical_model_name)
medical_model.to(device)
medical_model.eval()

# -----------------------------
# Embedding function
# -----------------------------
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
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# -----------------------------
# Connect to Qdrant
# -----------------------------
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "health_claim_memory"

# -----------------------------
# Canonical health claim narratives
# -----------------------------
claims = [
    {
        "id": 1,
        "narrative_id": "vaccine_autism",
        "text": "Vaccines cause autism",
        "domain": "vaccination",
        "verdict": "false"
    },
    {
        "id": 2,
        "narrative_id": "vaccine_infertility",
        "text": "Vaccines cause infertility",
        "domain": "vaccination",
        "verdict": "false"
    },
    {
        "id": 3,
        "narrative_id": "polio_harm",
        "text": "Polio drops are harmful",
        "domain": "vaccination",
        "verdict": "false"
    },
    {
        "id": 4,
        "narrative_id": "turmeric_diabetes",
        "text": "Turmeric cures diabetes",
        "domain": "nutrition",
        "verdict": "false"
    },
    {
        "id": 5,
        "narrative_id": "covid_fertility",
        "text": "COVID vaccines affect fertility",
        "domain": "vaccination",
        "verdict": "false"
    }
]

# -----------------------------
# Build Qdrant points
# -----------------------------
points = []

for claim in claims:
    claim_embedding = claim_model.encode(claim["text"])
    medical_embedding = embed_medical(claim["text"])

    point = PointStruct(
        id=claim["id"],
        vector={
            "claim_text_embedding": claim_embedding,
            "medical_context_embedding": medical_embedding[0]
        },
        payload={
            "claim_text": claim["text"],
            "narrative_id": claim["narrative_id"],
            "domain": claim["domain"],
            "verdict": claim["verdict"],
            "language": "en",
            "credibility_tier": 1,
            "frequency_count": 1,
            "last_seen": datetime.utcnow().isoformat()
        }
    )

    points.append(point)

# -----------------------------
# Upsert into Qdrant
# -----------------------------
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Inserted {len(points)} health claim narratives into Qdrant")