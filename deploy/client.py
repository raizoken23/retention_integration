import argparse
import json
import sys
from typing import Iterator

import requests


def sse_events(resp: requests.Response) -> Iterator[str]:
    buf = ""
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if not line:
            if buf:
                yield buf
                buf = ""
            continue
        # Accumulate lines until blank line
        buf += line + "\n"


def stream_completions(base_url: str, model: str, prompt: str) -> None:
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_new_tokens": 128,
        "temperature": 1.0,
        "top_p": 0.2,
        "stream": True,
    }
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        print(f"status={r.status_code}")
        for event in sse_events(r):
            for line in event.splitlines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    print("<DONE>")
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    print(f"[raw] {data}")
                    continue
                choices = obj.get("choices", [])
                if choices:
                    text = choices[0].get("text")
                    if text:
                        sys.stdout.write(text)
                        sys.stdout.flush()


def stream_chat(base_url: str, model: str, messages) -> None:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": True,
    }
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        print(f"status={r.status_code}")
        for event in sse_events(r):
            for line in event.splitlines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    print("<DONE>")
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    print(f"[raw] {data}")
                    continue
                choices = obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        sys.stdout.write(content)
                        sys.stdout.flush()


def completions_full(base_url: str, model: str, prompt: str) -> None:
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_new_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
    }
    r = requests.post(url, json=payload)
    r.raise_for_status()
    obj = r.json()
    choices = obj.get("choices", [])
    if choices:
        text = choices[0].get("text", "")
        print(text)


def chat_full(base_url: str, model: str, messages) -> None:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False,
    }
    r = requests.post(url, json=payload)
    r.raise_for_status()
    obj = r.json()
    choices = obj.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        print(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="./models/powercoder")
    parser.add_argument(
        "--mode",
        choices=["text", "chat", "text_full", "chat_full"],
        default="text",
    )
    args = parser.parse_args()

    if args.mode in ("text", "text_full"):
        prompt = '''<fim_prefix>
# test.py
def merge_sort(<fim_suffix>)<fim_middle>'''

        if args.mode == "text_full":
            completions_full(args.base_url, args.model, prompt)
        else:
            stream_completions(args.base_url, args.model, prompt)
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a Python function that reverses a list."},
        ]
        if args.mode == "chat_full":
            chat_full(args.base_url, args.model, messages)
        else:
            stream_chat(args.base_url, args.model, messages)
