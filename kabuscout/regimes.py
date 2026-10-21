"""相場の局面と、その時代の世の中の出来事。

25年分のデータには、リーマン・ショックやコロナ・ショックのように「普段とは別のルールで
動いた期間」が含まれる。そうした期間を平常時と同じように扱うと、
- 暴落時の極端な値動きを「よくあるパターン」と学習してしまう
- 戦略の成績が、たまたま危機をうまく避けた/当たったかどうかに左右される
ので、危機局面は「特例」として印を付け、学習では重みを下げ、評価は平常時と分けて行う。

特例の判定は2段構え:
1. 既知の出来事カレンダー（EVENTS）… 何が起きていたかを人が読める形で残す
2. 市場の値動きからの自動判定（stress）… カレンダーに無い未知の急変にも反応する
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Event:
    name: str
    start: str
    end: str
    kind: str  # crisis = 特例として扱う急変 / structural = 市場の仕組み・前提が変わった出来事（参考情報）
    note: str


EVENTS: list[Event] = [
    Event("ライブドア・ショック", "2006-01-16", "2006-02-28", "crisis", "強制捜査で新興市場が急落、東証のシステムも売買停止に"),
    Event("パリバ・ショック", "2007-08-09", "2007-08-31", "crisis", "サブプライムローン問題が表面化し世界的に信用収縮"),
    Event("リーマン・ショック", "2008-09-15", "2009-03-31", "crisis", "米大手証券の破綻から世界金融危機。日経平均は7,000円割れまで下落"),
    Event("東日本大震災", "2011-03-11", "2011-04-15", "crisis", "震災と原発事故で急落、その後も電力・保険など業種ごとに特殊な動き"),
    Event("欧州債務危機・米国債格下げ", "2011-08-01", "2011-10-31", "crisis", "超円高（1ドル75円台）と世界的なリスク回避"),
    Event("バーナンキ・ショック", "2013-05-23", "2013-06-30", "crisis", "FRBの緩和縮小示唆でアベノミクス相場が急反落"),
    Event("チャイナ・ショック", "2015-08-18", "2015-09-30", "crisis", "人民元切り下げと中国景気不安で世界同時株安"),
    Event("原油安・マイナス金利導入期の急落", "2016-01-04", "2016-02-29", "crisis", "原油急落と円高、日銀マイナス金利導入で銀行株が急落"),
    Event("英国EU離脱ショック", "2016-06-24", "2016-07-08", "crisis", "国民投票で離脱決定、円急騰"),
    Event("VIXショック", "2018-02-02", "2018-02-28", "crisis", "米国の金利上昇と変動率商品の崩壊で急落"),
    Event("米中貿易摩擦・年末急落", "2018-10-02", "2018-12-31", "crisis", "関税合戦と米利上げ懸念で年末にかけて急落"),
    Event("コロナ・ショック", "2020-02-21", "2020-04-30", "crisis", "感染拡大で世界同時暴落、その後の大規模緩和で急反発"),
    Event("令和のブラックマンデー", "2024-07-31", "2024-08-16", "crisis", "日銀利上げと円キャリートレード巻き戻しで8/5に日経平均が史上最大の下げ幅"),
    Event("米国の相互関税ショック", "2025-04-01", "2025-04-25", "crisis", "米国の大規模な関税発表で世界同時株安"),
    Event("異次元緩和", "2013-04-04", "2013-04-04", "structural", "日銀の量的・質的金融緩和。円安・株高の流れが加速"),
    Event("マイナス金利導入", "2016-01-29", "2016-01-29", "structural", "金融株・内需株の評価が大きく変化"),
    Event("売買単位100株に統一", "2018-10-01", "2018-10-01", "structural", "それ以前は1000株単位の銘柄も多く、少額での売買条件が異なる"),
    Event("東証の市場区分再編", "2022-04-04", "2022-04-04", "structural", "一部・二部・マザーズ・JASDAQ → プライム・スタンダード・グロース"),
    Event("東証のPBR改善要請", "2023-03-31", "2023-03-31", "structural", "低PBR・バリュー株の見直しが進む"),
    Event("新NISA開始", "2024-01-01", "2024-01-01", "structural", "個人の資金流入が増え、大型株・高配当株に買いが集まりやすく"),
    Event("マイナス金利解除", "2024-03-19", "2024-03-19", "structural", "「金利のある世界」へ。銀行株などの評価が変化"),
    Event("取引時間の延長", "2024-11-05", "2024-11-05", "structural", "大引けが15:00から15:30に"),
]

# 時代区分（成績を時代ごとに比べるため）
ERAS = [
    ("2002-01-01", "ITバブル後〜新興市場ブーム"),
    ("2008-01-01", "金融危機〜デフレ・超円高"),
    ("2013-01-01", "アベノミクス"),
    ("2020-01-01", "コロナ〜世界的インフレ"),
    ("2023-04-01", "東証改革・新NISA・金利のある世界"),
]


def era_of(dates) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    starts = np.array([pd.Timestamp(s) for s, _ in ERAS], dtype="datetime64[ns]")
    names = np.array([n for _, n in ERAS])
    return names[np.clip(np.searchsorted(starts, d, side="right") - 1, 0, len(names) - 1)]


def event_of(dates) -> np.ndarray:
    """危機イベントの期間内ならその名前、なければ空文字。"""
    d = pd.to_datetime(pd.Series(dates))
    out = np.array([""] * len(d), dtype=object)
    for e in EVENTS:
        if e.kind != "crisis":
            continue
        m = ((d >= pd.Timestamp(e.start)) & (d <= pd.Timestamp(e.end))).to_numpy()
        out[m] = e.name
    return out


def market_stress(m: pd.Series) -> pd.Series:
    """市場の値動きから「平常時と違う」局面を自動判定する（カレンダーに無い未知の急変にも反応）。
    - 20日の変動率（年率）が30%超、または過去1年の中央値の2倍超
    - 120日高値から15%以上の下落"""
    r = m.pct_change()
    vol = r.rolling(20, min_periods=15).std() * np.sqrt(250)
    base = vol.rolling(250, min_periods=120).median()
    dd = m / m.rolling(120, min_periods=60).max() - 1
    return ((vol > 0.30) | (vol > 2 * base) | (dd <= -0.15)).fillna(False)


def crisis_flag(dates, stress) -> np.ndarray:
    """特例扱いにするか: 既知の危機イベント中、または自動判定でストレス状態。"""
    return (event_of(dates) != "") | np.asarray(stress, dtype=bool)


def events_table() -> list[dict]:
    return [dict(name=e.name, start=e.start, end=e.end, kind=e.kind, note=e.note) for e in EVENTS]
