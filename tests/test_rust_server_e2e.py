
import requests
import time
import os

def test_synthesize():
    url = "http://127.0.0.1:12321/synthesize"
    payload = {
        "text": "Hello, this is a test of the Rust-backed TTS engine.",
        "prompt_text": "Greetings, how are you today?",
        "prompt_audio": "asset/zero_shot_prompt.wav",
        "speed": 1.0
    }

    # Ensure prompt audio exists
    if not os.path.exists(payload["prompt_audio"]):
        print(f"Error: {payload['prompt_audio']} not found. Please provide a valid prompt audio file.")
        return

    print(f"Sending request to {url}...")
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=60)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            print(f"Success! Received audio in {elapsed:.2f}s")
            with open("test_output_rust.wav", "wb") as f:
                f.write(response.content)
            print("Saved output to test_output_rust.wav")
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_synthesize()
