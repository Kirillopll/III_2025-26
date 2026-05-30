from __future__ import annotations

import json

import httpx


def main() -> None:
    with open("data/sample_predict.json", "r", encoding="utf-8") as file:
        payload = json.load(file)
    response = httpx.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
    print(response.status_code)
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
