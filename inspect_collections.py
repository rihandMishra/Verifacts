from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

print("\n--- health_claim_memory ---")
print(client.get_collection("health_claim_memory"))

print("\n--- health_fact_base ---")
print(client.get_collection("health_fact_base"))
