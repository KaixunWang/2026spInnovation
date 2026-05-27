"""CrEval warmup and speed test."""
import time, sys
print("Test 1: simple request (first warmup may be slow)...", flush=True)
from openai import OpenAI
client = OpenAI(api_key="0", base_url="http://127.0.0.1:8000/v1", timeout=3600)

# First request
t0 = time.perf_counter()
try:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=5,
        temperature=0,
    )
    dt = time.perf_counter() - t0
    print(f"Test1 DONE in {dt:.1f}s: {resp.choices[0].message.content}", flush=True)

    # Second request immediately after
    print("Test 2: second request...", flush=True)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5,
        temperature=0,
    )
    dt = time.perf_counter() - t0
    print(f"Test2 DONE in {dt:.1f}s: {resp.choices[0].message.content}", flush=True)
except Exception as e:
    dt = time.perf_counter() - t0
    print(f"ERR after {dt:.1f}s: {e}", flush=True)
