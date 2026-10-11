"""Deterministic risk maths. Pure functions - no LLM, no network."""
import math
import numpy as np

TRADING_DAYS = 252
# Assumptions (returned with every result so the answer can state them)
THRESHOLDS = {"single_stock_max_pct": 25.0, "top3_max_pct": 60.0, "sector_max_pct": 40.0}


def volatility(prices):
    """Annualised volatility (%) from close-to-close simple returns."""
    p = np.asarray(prices, dtype=float)
    if len(p) < 15:
        raise ValueError("need at least 15 prices")
    r = np.diff(p) / p[:-1]
    return float(np.std(r, ddof=1) * math.sqrt(TRADING_DAYS) * 100)


def max_drawdown(prices):
    """Max peak-to-trough decline (%), as a positive number."""
    p = np.asarray(prices, dtype=float)
    if len(p) < 2:
        raise ValueError("need at least 2 prices")
    peaks = np.maximum.accumulate(p)
    return float(np.max((peaks - p) / peaks) * 100)


def period_return(prices):
    return float((prices[-1] / prices[0] - 1) * 100)


def trend(prices):
    p = np.asarray(prices, dtype=float)
    if len(p) < 50:
        return "Insufficient data"
    ma20, ma50 = p[-20:].mean(), p[-50:].mean()
    return "Uptrend" if ma20 > ma50 else "Downtrend" if ma20 < ma50 else "Sideways"


def weights(holdings):
    vals = [(h["ticker"], float(h.get("currentValue") or 0)) for h in holdings]
    total = sum(v for _, v in vals)
    if total <= 0:
        raise ValueError("portfolio total value is zero")
    return [(t, v / total * 100) for t, v in vals], total


def allocation(holdings):
    w, total = weights(holdings)
    sectors = {}
    sec_of = {h["ticker"]: h.get("sector") or "Other" for h in holdings}
    for t, pct in w:
        sectors[sec_of[t]] = sectors.get(sec_of[t], 0) + pct
    return {
        "total_value": round(total, 2),
        "by_stock_pct": {t: round(p, 2) for t, p in sorted(w, key=lambda x: -x[1])},
        "by_sector_pct": {s: round(p, 2) for s, p in sorted(sectors.items(), key=lambda x: -x[1])},
    }


def concentration(holdings):
    w, _ = weights(holdings)
    pcts = sorted((p for _, p in w), reverse=True)
    hhi = sum((p / 100) ** 2 for p in pcts)
    alloc = allocation(holdings)
    flags = []
    if pcts[0] > THRESHOLDS["single_stock_max_pct"]:
        top = max(w, key=lambda x: x[1])
        flags.append(f"{top[0]} is {top[1]:.1f}% of portfolio (> {THRESHOLDS['single_stock_max_pct']:.0f}%)")
    if sum(pcts[:3]) > THRESHOLDS["top3_max_pct"]:
        flags.append(f"Top 3 holdings are {sum(pcts[:3]):.1f}% of portfolio (> {THRESHOLDS['top3_max_pct']:.0f}%)")
    for s, p in alloc["by_sector_pct"].items():
        if p > THRESHOLDS["sector_max_pct"] and s not in ("ETF", "Other"):
            flags.append(f"Sector {s} is {p:.1f}% of portfolio (> {THRESHOLDS['sector_max_pct']:.0f}%)")
    return {
        "holdings_count": len(pcts),
        "top1_pct": round(pcts[0], 2),
        "top3_pct": round(sum(pcts[:3]), 2),
        "hhi": round(hhi, 4),
        "effective_number_of_stocks": round(1 / hhi, 2),
        "flags": flags,
        "thresholds_used": THRESHOLDS,
    }


def diversification(holdings):
    conc = concentration(holdings)
    alloc = allocation(holdings)
    n_sectors = len([s for s in alloc["by_sector_pct"] if s not in ("Other",)])
    eff = conc["effective_number_of_stocks"]
    label = "Poor" if eff < 4 or n_sectors < 3 else "Moderate" if eff < 8 or n_sectors < 5 else "Good"
    return {"label": label, "effective_number_of_stocks": eff, "sector_count": n_sectors,
            "note": "Heuristic: effective N = 1/HHI; label thresholds are 4/8 stocks and 3/5 sectors."}
