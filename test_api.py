"""Minimal smoke test for the Azure OpenAI credentials.

Usage: python test_api.py
Reads config from .env (never hardcode the key here).
"""
import os
import sys
import json
import urllib.request
import urllib.error

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

payload = {
    "model": deployment,
    "input": "Reply with exactly the word: pong",
}

req = urllib.request.Request(
    endpoint,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "api-key": api_key},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.load(resp)
        print(f"HTTP {resp.status} OK")
        print(json.dumps(body, indent=2)[:2000])
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code} ERROR")
    print(e.read().decode()[:2000])
    sys.exit(1)
except urllib.error.URLError as e:
    print(f"Connection error: {e.reason}")
    sys.exit(1)
