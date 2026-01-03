import time

from wetext import Normalizer as EnNormalizer
from wetext import Normalizer as ZhNormalizer


def benchmark_wetext():
    print("Initializing wetext models...")
    start_time = time.time()
    zh_tn_model = ZhNormalizer(remove_erhua=False)
    en_tn_model = EnNormalizer()
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time * 1000:.2f} ms")

    test_sentences = [
        ("en", "Hello world, this is a test number 123."),
        ("en", "The date is 2025-12-28 and the time is 10:30 AM."),
        ("zh", "你好世界，这是一个测试 123。"),
        ("zh", "今天不仅是2025年1月1日，由于天气原因，原定于12:00的会议推迟。"),
    ]

    print("\nBenchmarking inference...")
    for lang, text in test_sentences:
        start_time = time.time()
        if lang == "en":
            result = en_tn_model.normalize(text)
        else:
            result = zh_tn_model.normalize(text)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        print(f"[{lang}] Input: {text}")
        print(f"      Output: {result}")
        print(f"      Time: {duration_ms:.4f} ms")

    # Stress test
    print("\nRunning stress test (1000 iterations English)...")
    start_time = time.time()
    for _ in range(1000):
        en_tn_model.normalize("The quick brown fox jumps over the 13 lazy dogs.")
    total_time = time.time() - start_time
    avg_time = total_time / 1000 * 1000
    print(f"Average time per sentence: {avg_time:.4f} ms")


if __name__ == "__main__":
    benchmark_wetext()
