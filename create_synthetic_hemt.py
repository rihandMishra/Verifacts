import pandas as pd

data = [
    # ---------------- FAKE ----------------
    {
        "id": 1,
        "text": "Doctors won’t tell you this but vaccines cause autism in children.",
        "label": "fake",
        "language": "en",
        "domain": "vaccination",
        "source": "synthetic"
    },
    {
        "id": 2,
        "text": "COVID vaccines affect fertility in women and men.",
        "label": "fake",
        "language": "en",
        "domain": "vaccination",
        "source": "synthetic"
    },
    {
        "id": 3,
        "text": "Turmeric can completely cure diabetes without medicine.",
        "label": "fake",
        "language": "en",
        "domain": "nutrition",
        "source": "synthetic"
    },
    {
        "id": 4,
        "text": "पोलियो की दवा बच्चों के लिए खतरनाक है।",
        "label": "fake",
        "language": "hi",
        "domain": "vaccination",
        "source": "synthetic"
    },
    {
        "id": 5,
        "text": "घरेलू नुस्खों से कैंसर पूरी तरह ठीक हो सकता है।",
        "label": "fake",
        "language": "hi",
        "domain": "disease",
        "source": "synthetic"
    },

    # ---------------- REAL ----------------
    {
        "id": 6,
        "text": "Vaccines are safe and effective in preventing infectious diseases.",
        "label": "real",
        "language": "en",
        "domain": "vaccination",
        "source": "govt"
    },
    {
        "id": 7,
        "text": "COVID-19 vaccines do not affect fertility according to medical studies.",
        "label": "real",
        "language": "en",
        "domain": "vaccination",
        "source": "govt"
    },
    {
        "id": 8,
        "text": "Regular exercise and diet help manage diabetes but do not cure it.",
        "label": "real",
        "language": "en",
        "domain": "nutrition",
        "source": "govt"
    },
    {
        "id": 9,
        "text": "टीके बच्चों को गंभीर बीमारियों से बचाने में मदद करते हैं।",
        "label": "real",
        "language": "hi",
        "domain": "vaccination",
        "source": "govt"
    },
    {
        "id": 10,
        "text": "कैंसर के इलाज के लिए डॉक्टर की सलाह आवश्यक है।",
        "label": "real",
        "language": "hi",
        "domain": "disease",
        "source": "govt"
    },
]

df = pd.DataFrame(data)
df.to_csv("synthetic_hemt_fake.csv", index=False)

print("✅ Synthetic HEMT-style dataset created: synthetic_hemt_fake.csv")
print("Shape:", df.shape)
print(df.head())
