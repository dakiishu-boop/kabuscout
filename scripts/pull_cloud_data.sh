#!/bin/bash
# GitHub Actions で動かしている場合に、最新の DB とモデルを手元に取ってくる（gh CLI が必要）
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
gh run download --name kabuscout-data --dir "$TMP"
mkdir -p "$DIR/data" "$DIR/reports"
cp -R "$TMP/data/." "$DIR/data/"
cp -R "$TMP/reports/." "$DIR/reports/" 2>/dev/null || true
rm -rf "$TMP"
echo "取得しました。streamlit run dashboard.py で確認できます"
