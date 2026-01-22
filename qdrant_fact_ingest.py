from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
COLLECTION_NAME = "health_fact_base"
VECTOR_SIZE = 768

# -----------------------------
# Connect to Qdrant
# -----------------------------
client = QdrantClient(url="http://localhost:6333")

# -----------------------------
# Create collection if needed
# -----------------------------
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "fact_embedding": VectorParams(
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
# Seed verified facts (MVP)
# -----------------------------
FACTS = [
    {
        "id": 1,
        "text": "Vaccines do not cause autism. Extensive studies show no causal link.",
        "domain": "vaccination",
        "source": "WHO",
        "url": "https://www.who.int/news-room/questions-and-answers/item/vaccines-and-autism",
        "confidence": 0.95
    },
    {
        "id": 2,
        "text": "COVID-19 vaccines do not affect fertility in men or women.",
        "domain": "vaccination",
        "source": "WHO",
        "url": "https://www.who.int",
        "confidence": 0.94
    }
]

# -----------------------------
# Embed + upload
# -----------------------------
points = []

for fact in FACTS:
    vector = model.encode(fact["text"]).tolist()

    point = PointStruct(
        id=fact["id"],
        vector={"fact_embedding": vector},
        payload={
            "fact_text": fact["text"],
            "domain": fact["domain"],
            "source": fact["source"],
            "source_url": fact["url"],
            "confidence": fact["confidence"]
        }
    )

    points.append(point)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("✅ Verified facts ingested into Qdrant")
