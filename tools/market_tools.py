"""Market data tools. Real API calls only - no fabricated fallback prices."""
import time
from datetime import datetime, timezone

import requests

from config import MARKETSTACK_API_KEY, ALPHAVANTAGE_API_KEY
from state.schemas import ok, fail
from tools import risk_tools as rt

_CACHE = {}            # (ticker, days) -> (fetched_at, series, source)
_CACHE_TTL = 300
STALE_AFTER_DAYS = 5


def _marketstack(ticker, days):
    r = requests.get("http://api.marketstack.com/v1/eod", timeout=10, params={
        "access_key": MARKETSTACK_API_KEY, "symbols": ticker + ".XNSE", "limit": days})
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"MarketStack: {data['error'].get('message', data['error'])}")
    rows = data.get("data") or []
    if not rows:
        raise RuntimeError("MarketStack returned no rows")
    return sorted([(d["date"][:10], float(d["close"])) for d in rows])


def _alphavantage(ticker, days):
    r = requests.get("https://www.alphavantage.co/query", timeout=10, params={
        "function": "TIME_SERIES_DAILY", "symbol": ticker + ".BSE", "apikey": ALPHAVANTAGE_API_KEY})
    ts = r.json().get("Time Series (Daily)")
    if not ts:
        raise RuntimeError("AlphaVantage returned no data (symbol unsupported or rate-limited)")
    return sorted((d, float(v["4. close"])) for d, v in ts.items())[-days:]


def fetch_series(ticker, days=100):
    """-> (series[(date, close)], source). Raises RuntimeError if every source fails."""
    key = (ticker, days)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1], hit[2]
    errors = []
    for name, fn in (("MarketStack", _marketstack), ("AlphaVantage", _alphavantage)):
        for attempt in range(2):                      # retry limit: 2 per source
            try:
                series = fn(ticker, days)
                _CACHE[key] = (time.time(), series, name)
                return series, name
            except Exception as e:
                errors.append(f"{name}: {e}")
                time.sleep(0.5)
    raise RuntimeError("; ".join(errors[-2:]))


def _stale(last_date):
    age = (datetime.now(timezone.utc).date() - datetime.strptime(last_date, "%Y-%m-%d").date()).days
    return age > STALE_AFTER_DAYS, age


def get_stock_quote(ticker, **_):
    try:
        series, src = fetch_series(ticker, 30)
    except Exception as e:
        return fail(e, "NO_MARKET_DATA")
    d, close = series[-1]
    stale, age = _stale(d)
    warn = [f"Latest available close is from {d} ({age} days old) - not a live quote."]
    return ok({"ticker": ticker, "last_close": close, "currency": "INR", "price_date": d},
              source=src, as_of=d, stale=stale, warnings=warn)


def get_historical_prices(ticker, days=100, **_):
    try:
        series, src = fetch_series(ticker, days)
    except Exception as e:
        return fail(e, "NO_MARKET_DATA")
    closes = [c for _, c in series]
    stale, _ = _stale(series[-1][0])
    return ok({"ticker": ticker, "points": len(series), "from": series[0][0], "to": series[-1][0],
               "first_close": closes[0], "last_close": closes[-1],
               "min_close": min(closes), "max_close": max(closes),
               "period_return_pct": round(rt.period_return(closes), 2),
               "last_10_closes": series[-10:]},
              source=src, as_of=series[-1][0], stale=stale)


def stock_risk_metrics(ticker, days=100, **_):
    try:
        series, src = fetch_series(ticker, days)
        closes = [c for _, c in series]
        stale, _ = _stale(series[-1][0])
        return ok({"ticker": ticker, "window_days": len(closes),
                   "annualised_volatility_pct": round(rt.volatility(closes), 2),
                   "max_drawdown_pct": round(rt.max_drawdown(closes), 2),
                   "period_return_pct": round(rt.period_return(closes), 2),
                   "trend_ma20_vs_ma50": rt.trend(closes),
                   "assumptions": "close-to-close simple returns, 252 trading days/year, drawdown on closes"},
                  source=src, as_of=series[-1][0], stale=stale)
    except ValueError as e:
        return fail(e, "INSUFFICIENT_DATA")
    except Exception as e:
        return fail(e, "NO_MARKET_DATA")


def compare_stocks(tickers, days=100, **_):
    rows, missing, as_of, stale_any = [], [], None, False
    for t in tickers:
        r = stock_risk_metrics(t, days)
        if not r["ok"]:
            missing.append({"ticker": t, "reason": r["error"]})
            continue
        q = get_stock_quote(t)
        rows.append({**r["data"], "last_close": q["data"].get("last_close")})
        as_of = max(filter(None, [as_of, r["as_of"]]))
        stale_any = stale_any or r["stale"]
    if not rows:
        return fail("no data for any requested ticker", "NO_MARKET_DATA")
    warn = [f"No data for {m['ticker']}: {m['reason']}" for m in missing]
    return ok({"comparison": rows, "missing": missing}, source="MarketStack/AlphaVantage",
              as_of=as_of, stale=stale_any, warnings=warn)
