import time

import inflect
from wetext import Normalizer as EnNormalizer

from cosyvoice.utils.frontend_utils import spell_out_number


def benchmark_alternatives():
    inflect_parser = inflect.engine()
    en_tn_model = EnNormalizer(lang="en")

    texts = ["123", "The date is 2025-12-28", "$100", "10:30 AM"]

    print("Comparing wetext vs spell_out_number (inflect):")
    for text in texts:
        print(f"\nInput: '{text}'")

        # Wetext
        start = time.time()
        res_wetext = en_tn_model.normalize(text)
        dur_wetext = (time.time() - start) * 1000
        print(f"  wetext: '{res_wetext}' ({dur_wetext:.2f} ms)")

        # spell_out_number
        start = time.time()
        res_inflect = spell_out_number(text, inflect_parser)
        dur_inflect = (time.time() - start) * 1000
        print(f"  inflect: '{res_inflect}' ({dur_inflect:.2f} ms)")


if __name__ == "__main__":
    benchmark_alternatives()
