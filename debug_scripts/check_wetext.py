
try:
    from wetext import Normalizer
    print("WeText found")
    norm = Normalizer(lang="en")
    texts = [
        "Hello! How are you?",
        "It costs $5.",
        "Date is 2024-01-01",
        "I have 2 dogs.",
        "Mr. Smith is here."
    ]
    for t in texts:
        print(f"Original: '{t}' -> Normalized: '{norm.normalize(t)}'")
except ImportError:
    print("WeText not installed")
