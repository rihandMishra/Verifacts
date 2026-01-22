from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
COLLECTION_NAME = "health_claim_memory"
VECTOR_SIZE = 768

# -----------------------------
# Connect to Qdrant
# -----------------------------
client = QdrantClient(url="http://localhost:6333")

# -----------------------------
# Create collection (idempotent)
# -----------------------------
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "claim_embedding": VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    }
)

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# -----------------------------
# Seed misinformation narratives
# -----------------------------
MISINFO_CLAIMS = [
    {
        "id": 1,
        "claim": "Vaccines cause autism",
        "domain": "vaccination",
        "verdict": "false"
    },
    {
        "id": 2,
        "claim": "Vaccines cause infertility",
        "domain": "vaccination",
        "verdict": "false"
    },
    {
        "id": 3,
        "claim": "Polio drops are harmful",
        "domain": "vaccination",
        "verdict": "false"
    },
    {
        "id": 4,
        "claim": "Drinking turmeric cures diabetes",
        "domain": "nutrition",
        "verdict": "false"
    }
]

# -----------------------------
# Embed + upload
# -----------------------------
points = []

for item in MISINFO_CLAIMS:
    vector = model.encode(item["claim"]).tolist()

    point = PointStruct(
        id=item["id"],
        vector={"claim_embedding": vector},
        payload={
            "claim_text": item["claim"],
            "domain": item["domain"],
            "verdict": item["verdict"]
        }
    )

    points.append(point)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("✅ Claims successfully ingested into Qdrant")
