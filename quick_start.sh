#!/usr/bin/env bash
# 一键：从零构建知识库 + 全息脑 + 跑验收测试
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
[ -d data/raw ] || bash scripts/fetch_data.sh
[ -f knowledge.db ] || python3 ingest_data.py
[ -f holo_brain.npz ] || python3 distill.py
python3 test_holo.py
echo
echo "现在可以："
echo "  python3 agent.py '记住我的手机是一加13'"
echo "  python3 agent.py '我手机是什么型号'"
echo "  python3 agent.py --task '统计沙箱里所有文本文件的行数'"
echo "  python3 agent.py --evolve 5      # 自我进化"
echo "  python3 agent.py --learn         # 睡眠重放巩固"
