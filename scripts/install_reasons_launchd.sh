#!/bin/bash
# Claude で TDnet の書類を読んで貯める launchd ジョブを、平日 21:30 に登録する。
#   1) 業績予想の修正の理由（kabuscout/tdnet_reasons.py、毎日）
#   2) 決算短信の定性情報（kabuscout/tdnet_quals.py、2026年秋の決算シーズンの分だけ。12/15 まで。費用が大きいので期限付き）
# TDnet は書類を約1ヶ月残すので、Mac がスリープ中で動かなかった日があっても、次に起きたときの実行で取り戻せる。
# API キーは .env の ANTHROPIC_API_KEY を使う。モデルを変えるときは .env に KABUSCOUT_REASON_MODEL=claude-haiku-5-5 など。
# 使い方: bash scripts/install_reasons_launchd.sh   /  解除: bash scripts/install_reasons_launchd.sh uninstall
set -euo pipefail
LABEL="com.kabuscout.reasons"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$DIR/.venv/bin/python"

if [[ "${1:-}" == "uninstall" ]]; then
  launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "解除しました"
  exit 0
fi

"$PY" -m pip install -q -r "$DIR/requirements-llm.txt"
mkdir -p "$DIR/logs" "$HOME/Library/LaunchAgents"
entries=""
for wd in 1 2 3 4 5; do
  entries+="    <dict><key>Weekday</key><integer>$wd</integer><key>Hour</key><integer>21</integer><key>Minute</key><integer>30</integer></dict>
"
done

cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string><string>-c</string>
    <string>"$PY" -m kabuscout.tdnet_reasons 60; if [ "\$(date +%Y%m%d)" -le "\${QUALS_UNTIL:-20261215}" ]; then "$PY" -m kabuscout.tdnet_quals 150; fi</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartCalendarInterval</key>
  <array>
$entries  </array>
  <key>StandardOutPath</key><string>$DIR/logs/reasons.log</string>
  <key>StandardErrorPath</key><string>$DIR/logs/reasons.log</string>
  <key>EnvironmentVariables</key>
  <dict><key>PYTHONWARNINGS</key><string>ignore</string></dict>
</dict>
</plist>
PL

launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "登録しました: $PLIST"
echo "今すぐ試す: launchctl kickstart gui/$(id -u)/$LABEL  （ログ: $DIR/logs/reasons.log）"
