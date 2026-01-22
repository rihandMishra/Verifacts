import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from claim_decomposer import extract_atomic_claims

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "synthetic_hemt_fake.csv"
COLLECTION_NAME = "health_claim_memory"
VECTOR_NAME = "claim_text_embedding"

START_ID = 1000  # avoid clashing with manual seeds

# ============================================================
# INIT
# ============================================================

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

client = QdrantClient(url="http://localhost:6333")

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

points = []
point_id = START_ID

# ============================================================
# PROCESS DATASET
# ============================================================

for idx, row in df.iterrows():
    label = str(row["label"]).lower()

    # Only ingest misinformation
    if label != "fake":
        continue

    text = str(row["text"])
    language = row.get("language", "unknown")
    domain = row.get("domain", "unknown")

    # Extract atomic claims
    claims = extract_atomic_claims(text)

    for claim in claims:
        vector = embedding_model.encode(claim).tolist()

        point = PointStruct(
            id=point_id,
            vector={VECTOR_NAME: vector},
            payload={
                "claim_text": claim,
                "domain": domain,
                "language": language,
                "verdict": "false",
                "source_dataset": "synthetic_hemt",
            },
        )

        points.append(point)
        point_id += 1

# ============================================================
# INGEST INTO QDRANT
# ============================================================

if points:
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

print(f"✅ Ingested {len(points)} misinformation claims into Qdrant")
