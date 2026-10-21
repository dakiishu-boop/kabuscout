"""通知: Discord Webhook / ntfy (スマホにプッシュ・OSS・アカウント不要) / macOS 通知センター。"""
from __future__ import annotations

import logging
import os
import platform
import subprocess

import requests

log = logging.getLogger(__name__)


def _chunks(text: str, size: int) -> list[str]:
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > size:
            out.append(cur)
            cur = ""
        cur += line
    if cur:
        out.append(cur)
    return out


def discord(webhook: str, text: str) -> None:
    for part in _chunks(text, 1900):
        r = requests.post(webhook, json={"content": part}, timeout=20)
        r.raise_for_status()


def ntfy(server: str, topic: str, title: str, text: str) -> None:
    body = text.encode("utf-8")[:3900].decode("utf-8", "ignore")
    r = requests.post(f"{server.rstrip('/')}/{topic}", data=body.encode("utf-8"), timeout=20,
                      headers={"Title": title.encode("utf-8"), "Tags": "chart_with_upwards_trend",
                               "Markdown": "yes"})
    r.raise_for_status()


def macos(title: str, text: str) -> bool:
    if platform.system() != "Darwin" or os.environ.get("CI"):
        return False
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display notification "{esc(text)}" with title "{esc(title)}" sound name "Glass"'],
                   check=False, timeout=10)
    return True


def send(cfg: dict, title: str, full: str, short: str) -> list[str]:
    """設定済みのチャネル全てに送る。送れたチャネル名を返す。"""
    n = cfg["notify"]
    sent = []
    if n.get("discord_webhook"):
        try:
            discord(n["discord_webhook"], f"**{title}**\n{full}")
            sent.append("discord")
        except Exception as e:
            log.warning("Discord通知失敗: %s", e)
    if n.get("ntfy_topic"):
        try:
            ntfy(n.get("ntfy_server", "https://ntfy.sh"), n["ntfy_topic"], title, full)
            sent.append("ntfy")
        except Exception as e:
            log.warning("ntfy通知失敗: %s", e)
    if n.get("macos"):
        try:
            if macos(title, short):
                sent.append("macos")
        except Exception as e:
            log.warning("macOS通知失敗: %s", e)
    return sent
