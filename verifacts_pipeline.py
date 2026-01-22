from claim_decomposer import extract_atomic_claims
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import ollama

# ============================================================
# Config
# ============================================================

MISINFO_COLLECTION = "health_claim_memory"
FACT_COLLECTION = "health_fact_base"

MISINFO_VECTOR_NAME = "claim_text_embedding"
FACT_VECTOR_NAME = "fact_embedding"

# ============================================================
# Initialize clients
# ============================================================

client = QdrantClient(url="http://localhost:6333")

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# ============================================================
# Core pipeline
# ============================================================

def check_health_claim(user_text: str):
    """
    End-to-end health misinformation check.
    Steps:
    1. Decompose input into atomic claims
    2. Retrieve similar misinformation narratives
    3. Retrieve verified medical facts
    4. Use LLM ONLY to explain retrieved evidence
    """

    atomic_claims = extract_atomic_claims(user_text)

    if not atomic_claims:
        return [{
            "verdict": "UNVERIFIED",
            "explanation": "No clear health-related factual claim detected.",
            "sources": []
        }]

    results = []

    for claim in atomic_claims:
        claim_vector = embedding_model.encode(claim).tolist()

        # -----------------------------
        # Retrieve misinformation narratives
        # -----------------------------
        misinfo_results = client.query_points(
            collection_name=MISINFO_COLLECTION,
            query=claim_vector,
            using=MISINFO_VECTOR_NAME,
            limit=2
        ).points

        # -----------------------------
        # Retrieve verified medical facts
        # -----------------------------
        fact_results = client.query_points(
            collection_name=FACT_COLLECTION,
            query=claim_vector,
            using=FACT_VECTOR_NAME,
            limit=2
        ).points

        # -----------------------------
        # Build fact context (STRICT)
        # -----------------------------
        if fact_results:
            fact_context = "\n".join(
                f"- {f.payload['fact_text']} (Source: {f.payload['source']})"
                for f in fact_results
            )
        else:
            fact_context = "No verified medical sources found."

        # -----------------------------
        # LLM prompt (explanation-only)
        # -----------------------------
        prompt = f"""
You are a health misinformation assistant.

User claim:
"{claim}"

Verified medical facts:
{fact_context}

Instructions:
1. Decide whether the claim is TRUE, FALSE, or UNVERIFIED
2. Explain briefly in simple language
3. Cite sources explicitly
4. Do NOT add any new medical information
5. Do NOT speculate beyond the provided facts

Answer format:

Verdict:
Explanation:
Sources:
"""

        llm_response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )

        results.append({
            "claim": claim,
            "response": llm_response["message"]["content"]
        })

    return results
# ============================================================