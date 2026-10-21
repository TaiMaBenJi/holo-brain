#!/usr/bin/env bash
# 下载开源语料（国内镜像优先）→ data_raw/
# 用法: bash scripts/fetch_data.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$HERE/data/raw"
mkdir -p "$RAW"

echo "[1/3] firefly-train-1.1M（中文多任务指令，采样 160MB）"
curl -fL --retry 5 -r 0-160000000 -o "$RAW/firefly.part" \
  "https://hf-mirror.com/datasets/YeungNLP/firefly-train-1.1M/resolve/main/firefly-train-1.1M.jsonl" \
  && python3 - "$RAW" <<'PY'
import os, sys
raw = sys.argv[1]
lines = open(os.path.join(raw, "firefly.part"), "rb").readlines()
if lines and not lines[-1].endswith(b"\n"):
    lines = lines[:-1]                       # 丢掉半行
open(os.path.join(raw, "firefly.jsonl"), "wb").write(b"".join(lines))
os.remove(os.path.join(raw, "firefly.part"))
print("   firefly 行数:", len(lines))
PY

echo "[2/3] alpaca-data-gpt4-chinese（中文指令，全量）"
curl -fL --retry 5 -o "$RAW/alpaca_zh.jsonl" \
  "https://hf-mirror.com/datasets/silk-road/alpaca-data-gpt4-chinese/resolve/main/Alpaca_data_gpt4_zh.jsonl"

echo "[3/3] clinc/oos-eval（151 类意图）"
curl -fL --retry 5 -o "$RAW/clinc.tar.gz" \
  "https://codeload.github.com/clinc/oos-eval/tar.gz/refs/heads/master" \
  && tar xzf "$RAW/clinc.tar.gz" -C "$RAW"

echo "完成。接着：python3 ingest_data.py && python3 distill.py"
