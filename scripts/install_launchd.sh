#!/bin/bash
# macOS で平日 16:30 に自動実行する launchd ジョブを登録する。
# Mac がスリープ中だった場合は、次に起動・復帰したタイミングで1回実行される（処理済みならスキップ）。
# 使い方: bash scripts/install_launchd.sh   /  解除: bash scripts/install_launchd.sh uninstall
set -euo pipefail
LABEL="com.kabuscout.daily"
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
  entries+="    <dict><key>Weekday</key><integer>$wd</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict>
"
done
# 翌朝 7:30 の保険枠（前日分が未処理なら実行。処理済みなら何もしない）
for wd in 2 3 4 5 6; do
  entries+="    <dict><key>Weekday</key><integer>$wd</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
"
done

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PY</string><string>-m</string><string>kabuscout</string><string>daily</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartCalendarInterval</key>
  <array>
$entries  </array>
  <key>StandardOutPath</key><string>$DIR/logs/launchd.log</string>
  <key>StandardErrorPath</key><string>$DIR/logs/launchd.log</string>
  <key>EnvironmentVariables</key>
  <dict><key>PYTHONWARNINGS</key><string>ignore</string></dict>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "登録しました: $PLIST"
echo "今すぐ試す: launchctl kickstart gui/$(id -u)/$LABEL  （ログ: $DIR/logs/launchd.log）"
