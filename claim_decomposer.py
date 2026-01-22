import spacy

# ============================================================
# Load spaCy pretrained model
# ============================================================

nlp = spacy.load("en_core_web_sm")

# Verbs that indicate reporting / framing, not factual claims
REPORTING_VERBS = {
    "say", "tell", "claim", "believe", "think",
    "suggest", "warn", "report"
}


def extract_atomic_claims(text: str):
    """
    Extract atomic factual claims using dependency parsing.
    Handles:
    - compound verbs (conj)
    - framing clauses
    - subject inheritance
    - biomedical term preservation (no lemmatization of objects)
    """
    doc = nlp(text)
    claims = []

    for sent in doc.sents:
        sent_text = sent.text.lower()

        # Drop obvious non-claims
        if any(x in sent_text for x in ["share", "forward", "send", "viral"]):
            continue
        if sent_text.startswith("i feel") or sent_text.startswith("i think"):
            continue

        # Find root verb
        root = next(
            (t for t in sent if t.dep_ == "ROOT" and t.pos_ == "VERB"),
            None
        )
        if not root:
            continue

        # Subjects attached to root (for inheritance)
        root_subjects = [
            t for t in root.lefts
            if t.dep_ in ("nsubj", "nsubjpass")
        ]

        # Root + conjunct verbs
        verbs = [root] + list(root.conjuncts)

        for verb in verbs:
            # Skip reporting verbs
            if verb.lemma_ in REPORTING_VERBS:
                continue

            # Subjects (inherit from root if missing)
            subjects = [
                t for t in verb.lefts
                if t.dep_ in ("nsubj", "nsubjpass")
            ]
            if not subjects:
                subjects = root_subjects

            objects = [
                t for t in verb.rights
                if t.dep_ in ("dobj", "attr", "pobj", "obj")
            ]

            for subj in subjects:
                subj_text = " ".join(t.text for t in subj.subtree)

                if objects:
                    for obj in objects:
                        obj_text = " ".join(t.text for t in obj.subtree)
                        # IMPORTANT: use surface verb form, NOT lemma
                        claim = f"{subj_text} {verb.text} {obj_text}"
                        claims.append(claim.lower())
                else:
                    claim = f"{subj_text} {verb.text}"
                    claims.append(claim.lower())

    # Deduplicate while preserving order
    return list(dict.fromkeys(claims))


# ============================================================
# Validation Tests (FREEZE AFTER PASS)
# ============================================================

if __name__ == "__main__":
    TEST_CASES = [
        (
            "Vaccines cause autism.",
            ["vaccines cause autism"]
        ),
        (
            "Vaccines weaken immunity and cause autism.",
            ["vaccines weaken immunity", "vaccines cause autism"]
        ),
        (
            "Doctors won’t tell you this but vaccines cause infertility.",
            ["vaccines cause infertility"]
        ),
        (
            "Vaccines cause autism. Share this fast!",
            ["vaccines cause autism"]
        ),
        (
            "I feel vaccines are dangerous.",
            []
        ),
        (
            "Turmeric cures diabetes. Vaccines cause autism.",
            ["turmeric cures diabetes", "vaccines cause autism"]
        ),
        (
            "Forward this message immediately!!!",
            []
        ),

        (
        "Vaccines weaken immunity, cause autism, and lead to infertility.",
        [
            "vaccines weaken immunity",
            "vaccines cause autism",
            "vaccines lead to infertility",
        ]
    ),

    # 9️⃣ Passive voice (very common in misinformation)
    (
        "Autism is caused by vaccines.",
        [
            "vaccines caused autism"
        ]
    ),

    # 🔟 Negation + false assertion
    (
        "Vaccines do not improve immunity but cause long-term harm.",
        [
            "vaccines cause long-term harm"
        ]
    ),

    # 1️⃣1️⃣ Reporting verb + factual clause
    (
        "Some people say vaccines cause autism, but studies show they are safe.",
        [
            "vaccines cause autism"
        ]
    ),

    # 1️⃣2️⃣ Conditional misinformation
    (
        "If children take vaccines, they develop neurological disorders.",
        [
            "children develop neurological disorders"
        ]
    ),

    # 1️⃣3️⃣ Quantifiers + generalization
    (
        "Most vaccines damage the immune system.",
        [
            "vaccines damage the immune system"
        ]
    ),

    # 1️⃣4️⃣ Multi-sentence, mixed domains
    (
        "Garlic cures cancer. Vaccines cause infertility. Doctors hide this.",
        [
            "garlic cures cancer",
            "vaccines cause infertility"
        ]
    ),

    # 1️⃣5️⃣ Modal verbs (can/may/might)
    (
        "Vaccines can cause serious side effects in children.",
        [
            "vaccines cause serious side effects in children"
        ]
    ),

    # 1️⃣6️⃣ Relative clause
    (
        "Vaccines that are given to infants weaken their immunity.",
        [
            "vaccines weaken their immunity"
        ]
    ),

    # 1️⃣7️⃣ Comparison + exaggeration
    (
        "Vaccines are more dangerous than the diseases they prevent.",
        [
            "vaccines are more dangerous than the diseases they prevent"
        ]
    ),
    ]



    print("\n=== CLAIM DECOMPOSER VALIDATION (FINAL) ===\n")

    for i, (text, expected) in enumerate(TEST_CASES, start=1):
        output = extract_atomic_claims(text)

        print(f"Test {i}")
        print("Input:   ", text)
        print("Expected:", expected)
        print("Output:  ", output)

        if output == expected:
            print("✅ PASS\n")
        else:
            print("❌ FAIL\n")
