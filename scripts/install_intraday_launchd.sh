#!/bin/bash
# 全銘柄の5分足を貯める launchd ジョブ（kabuscout/intraday.py）を、平日 16:45 に登録する。
# Yahoo は5分足を直近約60日分しか返さないので、Mac がスリープ中で動かなかった日があっても、次に起きたときの実行で取り戻せる。
# 使い方: bash scripts/install_intraday_launchd.sh   /  解除: bash scripts/install_intraday_launchd.sh uninstall
set -euo pipefail
LABEL="com.kabuscout.intraday"
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
entries=""
for wd in 1 2 3 4 5; do
  entries+="    <dict><key>Weekday</key><integer>$wd</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>45</integer></dict>
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
    <string>$PY</string><string>-m</string><string>kabuscout.intraday</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartCalendarInterval</key>
  <array>
$entries  </array>
  <key>StandardOutPath</key><string>$DIR/logs/intraday.log</string>
  <key>StandardErrorPath</key><string>$DIR/logs/intraday.log</string>
  <key>EnvironmentVariables</key>
  <dict><key>PYTHONWARNINGS</key><string>ignore</string></dict>
</dict>
</plist>
PL

launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "登録しました: $PLIST"
echo "今すぐ試す: launchctl kickstart gui/$(id -u)/$LABEL  （ログ: $DIR/logs/intraday.log）"
