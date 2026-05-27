from transformers import pipeline


classifier = pipeline("text-classification", model="winegarj/distilbert-base-uncased-finetuned-sst2", device="cpu")
tests = [
    ("This movie was absolutely fantastic!", "POSITIVE"),
    ("What a terrible waste of time.", "NEGATIVE"),
    ("The plot was dull and the acting was wooden.", "NEGATIVE"),
    ("I loved every minute of it.", "POSITIVE"),
]
for text, expected in tests:
    result = classifier(text)[0]
    status = "✓" if result["label"] == expected else "✗"
    print(f'{status} "{text}" → {result["label"]} ({result["score"]:.4f})')
