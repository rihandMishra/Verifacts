# VeriFacts Health  
## Accountability-First Health Misinformation Detector for Rural India

---

## 1. Overview

VeriFacts Health is an AI system designed to **detect, explain, and ground health misinformation** using verified medical evidence.  
The system is tailored for **rural and multilingual Indian contexts**, where misinformation spreads rapidly through informal and low-trust channels.

Unlike black-box fact-checking systems, VeriFacts prioritizes:
- **Transparency** – explains *why* a claim is false  
- **Accountability** – always cites verified sources  
- **Precision** – reasons at the level of atomic factual claims  

---

## 2. Problem Statement

Health misinformation in rural India spreads primarily through:
- WhatsApp forwards
- Informal community networks
- Non-expert medical advice

Challenges with existing systems:
- No explanation of *why* a claim is false
- Poor multilingual support
- Over-reliance on end-to-end LLM judgments
- No clear traceability to trusted medical sources

This creates **real-world harm** in areas such as vaccination, disease treatment, nutrition, and maternal health.

---

## 3. Our Solution

VeriFacts Health uses a **claim-centric, evidence-grounded pipeline**:

1. User input is decomposed into **atomic factual claims**
2. Claims are compared against a **misinformation memory**
3. Verified medical facts are retrieved from a **fact memory**
4. A local LLM is used **only to explain retrieved evidence**
5. The system returns a **verdict, explanation, and sources**

**Important:**  
The LLM never decides whether a claim is true or false.  
All decisions are driven by retrieval from trusted data.

---

## 4. System Architecture

User Input
↓
Claim Decomposer (syntax-aware)
↓
Atomic Health Claims
↓
Qdrant Vector Search
├─ Health Claim Memory (misinformation narratives)
└─ Health Fact Memory (verified medical facts)
↓
LLM (Explanation Only)
↓
Verdict + Explanation + Sources


---

## 5. Key Design Decisions

### 5.1 Claim-Level Reasoning (Not Document-Level)

- Articles and messages are decomposed into **atomic claims**
- Improves retrieval precision
- Enables explainability
- Avoids noisy document embeddings

### 5.2 Dual-Memory Architecture (Qdrant)

- **Health Claim Memory**  
  Stores known misinformation narratives
- **Health Fact Memory**  
  Stores verified medical counter-facts

Each memory uses **named vectors**, enabling explicit and auditable semantic search.

### 5.3 LLM as Explainer, Not Judge

- LLM does not classify claims
- LLM only summarizes retrieved evidence
- Prevents hallucinations and opaque reasoning

### 5.4 Precision-First Philosophy

- Not all sentences are treated as claims
- Opinion-based or ambiguous inputs return `UNVERIFIED`
- Prevents pollution of misinformation memory

---

## 6. Technology Stack

| Component | Technology |
|--------|-----------|
| Vector Database | Qdrant |
| Embeddings | paraphrase-multilingual-mpnet-base-v2 |
| NLP | spaCy |
| LLM | Mistral (via Ollama) |
| Languages | English, Hindi (extensible) |
| Interface | Python CLI |
| Deployment | Docker (Qdrant), Local Runtime |

---

## 7. Dataset Strategy

### 7.1 Synthetic HEMT-Style Dataset

Due to dataset access constraints during the hackathon, we created a **synthetic HEMT-style dataset** that mirrors:
- Structure
- Labels (`fake` / `real`)
- Multilingual health content
- Health-specific domains

The ingestion pipeline is **fully compatible with the real HEMT dataset** and can be swapped without code changes.

### 7.2 Claim Decomposition Rationale

Not all misinformation articles contain explicit factual claims.  
The system conservatively extracts **only syntactically valid atomic claims**, prioritizing **precision over recall**.

---

## 8. Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Docker (for Qdrant vector database)
- Ollama (for local LLM inference, optional for basic functionality)

### Step 1: Clone the Repository
```bash
git clone <[repository-url](https://github.com/rihandMishra/Verifacts.git)>
cd VeriFacts-Health
```

### Step 2: Set Up Python Virtual Environment
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Start Qdrant Database
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Step 5: Set Up the Collection
```bash
python qdrant_setup.py
```

### Step 6: Ingest Health Claim Data
```bash
python qdrant_ingestion.py
```

### Step 7: Test the System
```bash
python qdrant_query.py
```

### Step 8: Set Up Ollama (Optional, for LLM Explanations)
```bash
# Install Ollama from https://ollama.ai/
ollama pull mistral
```

### Step 9: Run the Full Pipeline (Optional)
```bash
python claim_decomposer.py
```
### Step 10: Example demo
'''bash
python verifacts_cli.py --example
'''





