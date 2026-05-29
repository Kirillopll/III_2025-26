from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Скачать датасет Home Credit Default Risk с Kaggle.")
    parser.add_argument("--output-dir", default="data/raw", help="Папка для скачанных и распакованных файлов.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI не установлен. Установите его через `pip install kaggle` и сначала настройте kaggle.json."
        )

    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "home-credit-default-risk", "-p", str(output_dir), "--unzip"],
        check=True,
    )
    print(f"Датасет скачан в {output_dir.resolve()}")


if __name__ == "__main__":
    main()
