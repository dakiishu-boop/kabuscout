#!/bin/bash
# TDnet の決算短信・業績予想の修正の PDF を、読まずにそのまま手元に保存する launchd ジョブを毎日 21:45 に登録する
# （kabuscout/tdnet_archive.py、ジャーナル §101）。Claude API は使わない（費用 0）。
# TDnet は書類を約1ヶ月残すので、Mac がスリープ中で動かなかった日があっても、次に起きたときの実行で直近30日を取り戻せる。
# 使い方: bash scripts/install_pdf_archive_launchd.sh   /  解除: bash scripts/install_pdf_archive_launchd.sh uninstall
set -euo pipefail
LABEL="com.kabuscout.pdfarchive"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$DIR/.venv/bin/python"

if [[ "${1:-}" == "uninstall" ]]; then
  launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "解除しました"
  exit 0
fi

mkdir -p "$DIR/logs" "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$PY</string><string>-m</string><string>kabuscout.tdnet_archive</string><string>30</string><string>90</string></array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>21</integer><key>Minute</key><integer>45</integer></dict>
  <key>StandardOutPath</key><string>$DIR/logs/pdf_archive.log</string>
  <key>StandardErrorPath</key><string>$DIR/logs/pdf_archive.log</string>
  <key>EnvironmentVariables</key>
  <dict><key>PYTHONWARNINGS</key><string>ignore</string></dict>
</dict>
</plist>
PL

launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "登録しました: $PLIST"
echo "今すぐ試す: launchctl kickstart gui/$(id -u)/$LABEL  （ログ: $DIR/logs/pdf_archive.log）"
