from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

query = "vaccines cause autism"
query_vector = model.encode(query).tolist()

results = client.query_points(
    collection_name="health_fact_base",
    query=query_vector,
    using="fact_embedding",
    limit=2
)

print("\nQuery:", query)
print("\nRetrieved facts:\n")

for r in results.points:
    print("Score:", round(r.score, 4))
    print("Fact:", r.payload["fact_text"])
    print("Source:", r.payload["source"])
    print("URL:", r.payload["source_url"])
    print("-" * 40)
