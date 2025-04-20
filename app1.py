import requests

GROQ_API_KEY = "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"
BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama3-8b-8192",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "temperature": 0.7,
    "max_tokens": 100
}

response = requests.post(BASE_URL, headers=HEADERS, json=payload)
print(response.status_code)
print(response.text)
