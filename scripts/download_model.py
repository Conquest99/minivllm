"""把实验用的权重（约 1 GB）下到 ``models/``。

有些网络访问不到 ``huggingface.co``；如果你有本地代理，先设上，例如
``$env:HTTPS_PROXY = "http://127.0.0.1:10081"``，或者把 ``HF_ENDPOINT`` 指向镜像。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--local-dir", default="models/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    target = Path(args.local_dir)
    target.mkdir(parents=True, exist_ok=True)
    print(f"endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"proxy   : {os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY') or '(none)'}")
    path = snapshot_download(
        repo_id=args.model,
        local_dir=str(target),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        ignore_patterns=["*.gguf", "original/*"],
    )
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"downloaded {args.model} -> {path} ({total / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
