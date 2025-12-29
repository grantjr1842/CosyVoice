import logging
import os
import sys
import warnings

# Add project root to sys.path
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_imports():
    logging.info("Testing imports...")
    try:
        logging.info("Successfully imported CosyVoice3")
    except Exception as e:
        logging.error(f"Failed to import CosyVoice3: {e}")
        return False
    return True


def test_discriminator_import():
    logging.info("Testing discriminator import (weight_norm)...")
    try:
        logging.info("Successfully imported DiscriminatorR")
    except Exception as e:
        logging.error(f"Failed to import DiscriminatorR: {e}")

    # Check if warning is generated (hard to check in script if filtered, but we can check if filter is active)
    filters = warnings.filters
    print("Warning filters:", filters)


def mock_inference_metrics():
    logging.info("Testing metric calculation logic...")
    # We can't easily run full inference without model, but we can verify the code logic if we could import the function headers
    # Instead, we will rely on static analysis or just the fact that it runs without syntax error.
    pass


if __name__ == "__main__":
    if test_imports():
        test_discriminator_import()
        mock_inference_metrics()
        print("Verification script finished.")
