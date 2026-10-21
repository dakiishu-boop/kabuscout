"""KabuScout ダッシュボード:  streamlit run dashboard.py"""
from __future__ import annotations

import json
import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from kabuscout.config import load_config
from kabuscout.db import connect
from kabuscout.strategies import STRATEGIES, STRATEGY_NAMES

warnings.filterwarnings("ignore")
st.set_page_config(page_title="KabuScout", page_icon="📈", layout="wide")
cfg = load_config()
con = connect(cfg["data"]["db_path"])
q = lambda sql, p=(): pd.read_sql(sql, con, params=p)


def strat_label(s: str) -> str:
    return "・".join(STRATEGY_NAMES.get(k, k) for k in (s or "").split(",") if k)


def chart(code: str, name: str, marks: pd.DataFrame | None = None):
    df = q("SELECT * FROM prices WHERE code=? ORDER BY date DESC LIMIT 260", (code,)).iloc[::-1]
    if df.empty:
        st.info("株価データがありません")
        return
    df["date"] = pd.to_datetime(df["date"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 name=code, increasing_line_color="#d62728", decreasing_line_color="#1f77b4"), 1, 1)
    for n, col in ((5, "#999"), (25, "#2ca02c"), (75, "#ff7f0e"), (200, "#9467bd")):
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"].rolling(n).mean(), name=f"{n}日線",
                                 line=dict(width=1.2, color=col)), 1, 1)
    if marks is not None and len(marks):
        m = marks.assign(date=pd.to_datetime(marks["date"]))
        fig.add_trace(go.Scatter(x=m["date"], y=m["close"] * 0.97, mode="markers", name="シグナル",
                                 marker=dict(symbol="triangle-up", size=11, color="#e377c2")), 1, 1)
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="出来高", marker_color="#aaa"), 2, 1)
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, title=f"{code} {name}",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, use_container_width=True)


st.title("📈 KabuScout")
dates = q("SELECT DISTINCT date FROM signals ORDER BY date DESC")["date"].tolist()
tab1, tab2, tab3, tab4 = st.tabs(["今日の候補", "AIの学習状況", "答え合わせ", "ペーパートレード"])

with tab1:
    if not dates:
        st.info("まだスクリーニング結果がありません。`python -m kabuscout daily` を実行してください。")
    else:
        d = st.selectbox("日付", dates, index=0)
        sig = q("SELECT * FROM signals WHERE date=? ORDER BY rank", (d,))
        view = pd.DataFrame({
            "順位": sig["rank"], "コード": sig["code"], "銘柄": sig["name"],
            "戦略": sig["strategies"].map(strat_label),
            "AI勝率予測": sig["prob"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "-"),
            "終値": sig["close"].round(0), "利確目安": sig["tp"].round(0), "損切目安": sig["sl"].round(0),
            "決算まで": sig["earnings"].map(lambda v: f"{int(v)}日" if pd.notna(v) else ""),
            "根拠": sig["reasons"],
            "結果": sig.apply(lambda r: ("✅" if r["label"] == 1 else "❌") + f" {r['ret']:+.1%}" if r["labeled"] else "", axis=1),
        })
        st.dataframe(view, hide_index=True, use_container_width=True)
        if len(sig):
            pick = st.selectbox("チャート", [f"{r.code} {r.name}" for r in sig.itertuples()])
            code = pick.split()[0]
            chart(code, sig.loc[sig["code"] == code, "name"].iat[0], q("SELECT date, close FROM signals WHERE code=?", (code,)))

with tab2:
    models = q("SELECT * FROM models ORDER BY version DESC")
    if models.empty:
        st.info("モデルはまだ学習されていません")
    else:
        act = models[models["active"] == 1].iloc[0] if (models["active"] == 1).any() else models.iloc[0]
        m = json.loads(act["metrics"])
        c = st.columns(5)
        c[0].metric("現役モデル", act["version"])
        c[1].metric("検証AUC", f"{m['auc']:.3f}", help="0.5=当てずっぽう。0.55超で実用的な識別力")
        c[2].metric("上位推奨の勝率", f"{m['daily_top_win']:.0%}", f"{(m['daily_top_win'] - m['base_win']) * 100:+.1f}pt vs 全シグナル")
        c[3].metric("上位推奨の期待値", f"{m['daily_top_ret']:+.2%}", f"{(m['daily_top_ret'] - m['base_ret']) * 100:+.2f}pt")
        c[4].metric("学習データ", f"{act['n_train']:,}件")
        st.caption(f"検証期間: {m.get('holdout_from', '?')} 以降（学習に使っていない期間） / 判定: {m.get('verdict', '')} / "
                   f"ルールとの配合比 AI {m['alpha']:.0%}")
        st.subheader("AIが重視している要素")
        imp = pd.DataFrame(m["importance"], columns=["特徴量", "重要度"])
        st.bar_chart(imp.set_index("特徴量"))
        st.subheader("モデルの世代交代")
        hist = models.assign(**{k: models["metrics"].map(lambda s, k=k: json.loads(s).get(k)) for k in ("auc", "daily_top_win", "daily_top_ret")})
        st.line_chart(hist.set_index("version")[["auc", "daily_top_win"]].iloc[::-1])
    st.subheader("戦略ごとの直近成績（120営業日）と自動調整された重み")
    ss = q("SELECT * FROM strategy_stats WHERE date=(SELECT MAX(date) FROM strategy_stats)")
    if len(ss):
        ss["戦略"] = ss["strategy"].map(STRATEGY_NAMES)
        ss["説明"] = ss["strategy"].map({s.key: s.desc for s in STRATEGIES})
        st.dataframe(ss[["戦略", "n", "win_rate", "avg_ret", "weight", "説明"]].rename(
            columns={"n": "件数", "win_rate": "勝率", "avg_ret": "平均リターン", "weight": "重み"}).style.format(
            {"勝率": "{:.1%}", "平均リターン": "{:+.2%}", "重み": "{:.2f}"}), hide_index=True, use_container_width=True)

with tab3:
    lab = q("SELECT * FROM signals WHERE labeled=1")
    if lab.empty:
        st.info(f"通知した候補の結果は、{cfg['labeling']['horizon_days']}営業日後から順次ここに並びます")
    else:
        top5 = lab[lab["rank"] <= 5]
        c = st.columns(3)
        c[0].metric("答え合わせ済み", f"{len(lab)}件")
        c[1].metric("上位5件の勝率", f"{top5['label'].mean():.0%}")
        c[2].metric("上位5件の平均リターン", f"{top5['ret'].mean():+.2%}")
        by = lab.groupby("date").agg(勝率=("label", "mean"), 平均リターン=("ret", "mean"))
        st.line_chart(by)
        st.dataframe(lab.sort_values("date", ascending=False), hide_index=True, use_container_width=True)

with tab4:
    pos = q("SELECT * FROM paper_positions ORDER BY id DESC")
    if pos.empty:
        st.info("まだ仮想売買はありません")
    else:
        closed = pos[pos["status"] == "closed"].sort_values("exit_date")
        c = st.columns(4)
        c[0].metric("保有中", int((pos["status"] == "open").sum()))
        c[1].metric("決済済み", len(closed))
        c[2].metric("勝率", f"{(closed['pnl'] > 0).mean():.0%}" if len(closed) else "-")
        c[3].metric("実現損益", f"{closed['pnl'].sum():+,.0f}円")
        if len(closed):
            eq = cfg["paper_trading"]["capital_jpy"] + closed.set_index("exit_date")["pnl"].cumsum()
            st.line_chart(eq)
        st.dataframe(pos, hide_index=True, use_container_width=True)
