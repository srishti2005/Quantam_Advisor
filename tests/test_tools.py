import pytest
from tools import risk_tools as rt
from tools.registry import execute_tool_safely, openai_tool_schemas, TOOL_REGISTRY

H = [{"ticker": "A", "currentValue": 600, "sector": "IT"},
     {"ticker": "B", "currentValue": 300, "sector": "Banking"},
     {"ticker": "C", "currentValue": 100, "sector": "Banking"}]


def test_drawdown_and_return():
    p = [100, 120, 90, 110]
    assert rt.max_drawdown(p) == pytest.approx(25.0)
    assert rt.period_return(p) == pytest.approx(10.0)


def test_volatility_flat_series_is_zero():
    assert rt.volatility([100] * 30) == 0.0
    with pytest.raises(ValueError):
        rt.volatility([100] * 5)


def test_concentration_flags_and_hhi():
    c = rt.concentration(H)
    assert c["top1_pct"] == 60.0 and c["top3_pct"] == 100.0
    assert c["hhi"] == pytest.approx(0.36 + 0.09 + 0.01)
    assert any("A is 60.0%" in f for f in c["flags"])
    assert any("Banking" in f for f in c["flags"]) is False   # banking is 40.0, not > 40


def test_allocation_sums_to_100():
    a = rt.allocation(H)
    assert sum(a["by_sector_pct"].values()) == pytest.approx(100, abs=0.01)


def test_zero_value_portfolio_rejected():
    with pytest.raises(ValueError):
        rt.allocation([{"ticker": "A", "currentValue": 0}])


def test_registry_rejects_unknown_tool_bad_args_and_extra_args():
    assert execute_tool_safely("place_trade", {}, "u")["error_code"] == "UNKNOWN_TOOL"
    assert execute_tool_safely("get_stock_quote", {"ticker": "RELIANCE; DROP"}, "u")["error_code"] == "BAD_ARGS"
    assert execute_tool_safely("get_stock_quote", {}, "u")["error_code"] == "BAD_ARGS"
    assert execute_tool_safely("get_portfolio", {"user_id": "someone-else"}, "u")["error_code"] == "BAD_ARGS"
    assert execute_tool_safely("compare_stocks", {"tickers": ["TCS"]}, "u")["error_code"] == "BAD_ARGS"


def test_user_id_never_exposed_to_model():
    for s in openai_tool_schemas():
        assert "user_id" not in s["function"]["parameters"]["properties"]
    assert not any("trade" in n or "order" in n for n in TOOL_REGISTRY)


def test_portfolio_tool_reads_only_session_user(tmp_path, monkeypatch):
    import state.portfolio_store as ps
    monkeypatch.setattr(ps, "DB_PATH", str(tmp_path / "p.db"))
    ps.save_portfolio("u:alice", {"stocks": [{"ticker": "TCS", "currentValue": 100, "sector": "IT"}],
                                  "totalValue": 100, "totalCost": 90, "totalPnL": 10, "totalPnLPct": 11.1})
    assert execute_tool_safely("get_portfolio", {}, "u:alice")["ok"]
    r = execute_tool_safely("get_portfolio", {}, "u:bob")
    assert r["error_code"] == "NO_PORTFOLIO"


def test_market_failure_becomes_envelope_not_exception(monkeypatch):
    from tools import market_tools as mt
    def boom(*a, **k): raise RuntimeError("API down")
    monkeypatch.setattr(mt, "fetch_series", boom)
    r = execute_tool_safely("get_stock_quote", {"ticker": "TCS"}, "u")
    assert r["ok"] is False and r["error_code"] == "NO_MARKET_DATA"


def test_tool_timeout(monkeypatch):
    import time
    from tools import registry
    monkeypatch.setitem(registry.TOOL_REGISTRY, "get_stock_quote",
                        registry.Tool(lambda **k: time.sleep(2), "x", {"ticker": registry.TICKER_P}, ("ticker",)))
    assert execute_tool_safely("get_stock_quote", {"ticker": "TCS"}, "u", timeout=0.2)["error_code"] == "TIMEOUT"
