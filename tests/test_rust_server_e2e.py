
import os
import time

import requests

def test_synthesize():
    server_url = os.getenv("COSYVOICE_SERVER_URL", "http://127.0.0.1:3000").rstrip("/")
    url = f"{server_url}/synthesize"
    prompt_audio = os.getenv("COSYVOICE_PROMPT_AUDIO", "asset/zero_shot_prompt.wav")
    payload = {
        "text": os.getenv(
            "COSYVOICE_TTS_TEXT",
            "Hello, this is a test of the Rust-backed TTS engine.",
        ),
        "prompt_text": os.getenv(
            "COSYVOICE_PROMPT_TEXT",
            "You are a helpful assistant.<|endofprompt|>Greetings, how are you today?",
        ),
        "prompt_audio": prompt_audio,
        "speed": 1.0
    }

    # Ensure prompt audio exists
    if not os.path.exists(payload["prompt_audio"]):
        print(f"Error: {payload['prompt_audio']} not found. Please provide a valid prompt audio file.")
        return

    output_path = os.getenv("COSYVOICE_OUTPUT", "test_output_rust.wav")

    print(f"Sending request to {url}...")
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=60)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            print(f"Success! Received audio in {elapsed:.2f}s")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Saved output to {output_path}")
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_synthesize()
