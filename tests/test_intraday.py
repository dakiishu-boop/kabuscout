"""5分足の蓄積（kabuscout/intraday.py）: ネットにつながらず、作り物の yfinance の表で確かめる。"""
import numpy as np
import pandas as pd

from kabuscout import intraday


def fake_download(times_utc, tickers=("7203.T", "^N225")):
    idx = pd.DatetimeIndex(times_utc, tz="UTC", name="Datetime")
    cols = pd.MultiIndex.from_product([list(tickers), ["Open", "High", "Low", "Close", "Adj Close", "Volume"]],
                                      names=["Ticker", "Price"])
    v = np.arange(len(idx) * len(cols), dtype=float).reshape(len(idx), len(cols)) + 100
    df = pd.DataFrame(v, index=idx, columns=cols)
    df.iloc[-1, :6] = np.nan  # 最後の足は値が無い（取引の無い銘柄・時刻）
    return df


def test_to_long_converts_to_japan_time_and_codes():
    df = fake_download(["2026-11-04 00:00", "2026-11-04 00:05", "2026-11-04 06:25"])
    lg = intraday.to_long(df)
    assert set(lg["code"]) == {"7203", "^N225"}
    assert lg["datetime"].min() == pd.Timestamp("2026-11-04 09:00")  # UTC 0:00 = 日本時間 9:00
    assert lg["datetime"].dt.tz is None
    assert len(lg[lg["code"] == "7203"]) == 2  # 値の無い足は捨てる
    assert list(lg.columns) == ["datetime", "code", "open", "high", "low", "close", "volume"]


def test_to_long_drops_today_before_the_close():
    df = fake_download(["2026-11-04 05:00", "2026-11-05 01:00", "2026-11-05 01:05"])
    lg = intraday.to_long(df, now=pd.Timestamp("2026-11-05 11:00"))
    assert lg["datetime"].dt.normalize().max() == pd.Timestamp("2026-11-04")
    lg2 = intraday.to_long(df, now=pd.Timestamp("2026-11-05 16:45"))
    assert lg2["datetime"].dt.normalize().max() == pd.Timestamp("2026-11-05")


def test_merge_save_overwrites_same_bar_and_splits_by_month(tmp_path):
    a = intraday.to_long(fake_download(["2026-10-30 05:00", "2026-11-04 00:00", "2026-11-04 00:05"]))
    intraday.merge_save(a, tmp_path)
    b = a[a["datetime"] == pd.Timestamp("2026-11-04 09:00")].assign(close=1.0)
    counts = intraday.merge_save(b, tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == ["2026-10.parquet", "2026-11.parquet"]
    nov = pd.read_parquet(tmp_path / "2026-11.parquet")
    assert counts["2026-11"] == len(nov) and not nov.duplicated(["datetime", "code"]).any()
    assert (nov.loc[nov["datetime"] == pd.Timestamp("2026-11-04 09:00"), "close"] == 1.0).all()
