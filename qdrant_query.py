from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, NamedVector
from sentence_transformers import SentenceTransformer

# -----------------------------
# Connect to Qdrant
# -----------------------------
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "health_claim_memory"

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# -----------------------------
# User query
# -----------------------------
user_claim = "I heard vaccines can make children autistic"
query_vector = model.encode(user_claim)

# -----------------------------
# Correct vector query using Prefetch + NamedVector
# -----------------------------
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    using="claim_text_embedding",
    limit=3
)

# -----------------------------
# Print results
# -----------------------------
print("\nUser claim:", user_claim)
print("\nTop retrieved narratives:\n")

for rank, res in enumerate(results.points, start=1):
    print(f"Rank {rank}")
    print("Score:", round(res.score, 4))
    print("Narrative ID:", res.payload["narrative_id"])
    print("Stored claim:", res.payload["claim_text"])
    print("Domain:", res.payload["domain"])
    print("Verdict:", res.payload["verdict"])
    print("-" * 40)
