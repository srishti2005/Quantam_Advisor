"""Portfolio tools. The user_id is injected by the runtime from the server session."""
from datetime import datetime, timezone

from state.schemas import ok, fail
from state.portfolio_store import load_portfolio
from tools import risk_tools as rt

PRICE_NOTE = ("Holding values use the uploaded file's price, else a static price table, else the buy "
              "price. They are not live quotes - verify with get_stock_quote before drawing P&L conclusions.")


def _holdings(user_id):
    p = load_portfolio(user_id)
    if not p or not p.get("stocks"):
        return None, fail("No portfolio uploaded for this user. Ask the user to upload a CSV/XLSX.", "NO_PORTFOLIO")
    return p, None


def get_portfolio(user_id, **_):
    p, err = _holdings(user_id)
    if err:
        return err
    stocks = [{k: s.get(k) for k in ("ticker", "quantity", "buyPrice", "currentPrice",
                                     "currentValue", "pnl", "pnlPct", "sector")} for s in p["stocks"]]
    up = datetime.fromtimestamp(p["uploaded_at"], timezone.utc).isoformat(timespec="seconds")
    return ok({"holdings": stocks, "total_value": p["totalValue"], "total_cost": p["totalCost"],
               "total_pnl": p["totalPnL"], "total_pnl_pct": p["totalPnLPct"]},
              source="user upload", as_of=up, warnings=[PRICE_NOTE])


def calculate_allocation(user_id, **_):
    p, err = _holdings(user_id)
    if err:
        return err
    try:
        return ok(rt.allocation(p["stocks"]), source="computed from uploaded holdings", warnings=[PRICE_NOTE])
    except ValueError as e:
        return fail(e, "INVALID_PORTFOLIO")


def calculate_concentration(user_id, **_):
    p, err = _holdings(user_id)
    if err:
        return err
    try:
        return ok(rt.concentration(p["stocks"]), source="computed from uploaded holdings", warnings=[PRICE_NOTE])
    except ValueError as e:
        return fail(e, "INVALID_PORTFOLIO")


def calculate_diversification(user_id, **_):
    p, err = _holdings(user_id)
    if err:
        return err
    try:
        return ok(rt.diversification(p["stocks"]), source="computed from uploaded holdings")
    except ValueError as e:
        return fail(e, "INVALID_PORTFOLIO")
