"""通知: Discord Webhook（カード表示）/ ntfy（スマホにプッシュ・OSS・アカウント不要）/ macOS 通知センター。"""
from __future__ import annotations

import logging
import os
import platform
import subprocess

import requests

from . import report

log = logging.getLogger(__name__)


def discord(webhook: str, payload: dict) -> None:
    r = requests.post(webhook, json=payload, timeout=20)
    r.raise_for_status()


def ntfy(server: str, topic: str, title: str, text: str, click: str = "") -> None:
    headers = {"Title": title.encode("utf-8"), "Tags": "chart_with_upwards_trend"}
    if click:
        headers["Click"] = click
    r = requests.post(f"{server.rstrip('/')}/{topic}", data=text.encode("utf-8")[:3900], timeout=20, headers=headers)
    r.raise_for_status()


def macos(title: str, text: str) -> bool:
    if platform.system() != "Darwin" or os.environ.get("CI"):
        return False
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display notification "{esc(text)}" with title "{esc(title)}" sound name "Glass"'],
                   check=False, timeout=10)
    return True


def send(cfg: dict, rep: dict) -> list[str]:
    """設定済みのチャネル全てに日次レポートを送る。送れたチャネル名を返す。"""
    title = f"KabuScout {rep['date_label']} 引け後スクリーニング"
    short = f"{len(rep['picks'])}件: " + ", ".join(f"{p['code']} {p['name']}" for p in rep["picks"][:5])
    return _send(cfg, title, short,
                 discord_payload=report.discord_payload(rep),
                 ntfy_body=report.ntfy_text(rep), click=rep["site_url"])


def send_text(cfg: dict, title: str, text: str) -> list[str]:
    return _send(cfg, title, text, discord_payload={"username": "KabuScout", "content": f"**{title}**\n{text}"},
                 ntfy_body=text)


def _send(cfg, title, short, discord_payload, ntfy_body, click="") -> list[str]:
    n = cfg["notify"]
    sent = []
    if n.get("discord_webhook"):
        try:
            discord(n["discord_webhook"], discord_payload)
            sent.append("discord")
        except Exception as e:
            log.warning("Discord通知失敗: %s", e)
    if n.get("ntfy_topic"):
        try:
            ntfy(n.get("ntfy_server", "https://ntfy.sh"), n["ntfy_topic"], title, ntfy_body, click)
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
