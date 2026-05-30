from __future__ import annotations

import uvicorn

from src.config import load_config


def main() -> None:
    service_cfg = load_config()["service"]
    uvicorn.run(
        "src.service.app:app",
        host=str(service_cfg["host"]),
        port=int(service_cfg["port"]),
        log_level=str(service_cfg["log_level"]),
    )


if __name__ == "__main__":
    main()
