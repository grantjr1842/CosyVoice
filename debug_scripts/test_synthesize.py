import requests
import json
import os

def test_synthesize():
    url = "http://localhost:8000/synthesize"
    
    # Using one of the found wav files
    prompt_audio = os.path.abspath("asset/zero_shot_prompt.wav")
    
    payload = {
        "text": "Hello, this is a test of the Rust-backed TTS engine.",
        "prompt_text": "希望你以后能够做得更好。", # Assuming this matches the prompt audio
        "prompt_audio": prompt_audio,
        "speed": 1.0
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            with open("output_test.wav", "wb") as f:
                f.write(response.content)
            print("Successfully saved output to output_test.wav")
            print(f"Audio duration: {response.headers.get('x-audio-duration')}s")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_synthesize()
