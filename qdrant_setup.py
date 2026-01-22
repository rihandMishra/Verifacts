from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Connect to local Qdrant
client = QdrantClient(
    url="http://localhost:6333"
)

COLLECTION_NAME = "health_claim_memory"

# Recreate collection (safe during development)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "claim_text_embedding": VectorParams(
            size=768,
            distance=Distance.COSINE
        ),
        "medical_context_embedding": VectorParams(
            size=768,
            distance=Distance.COSINE
        ),
    }
)

print(f"Collection '{COLLECTION_NAME}' created successfully")
