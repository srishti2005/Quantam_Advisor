"""Orchestrator tests use a scripted fake planner - no OpenAI, no network."""
import itertools
import pytest
from agent_orchestrator import AgentOrchestrator, unsupported_figures
from state.schemas import ok, fail
from tools import registry

_ids = itertools.count()


def call(name, **args):
    return {"type": "tool_calls", "calls": [{"id": f"c{next(_ids)}", "name": name, "arguments": args}]}


def final(text):
    return {"type": "final", "text": text}


def scripted(*steps):
    it = iter(steps)
    return lambda messages, tools: next(it)


def stub_tool(monkeypatch, name, result_or_fn, counter=None):
    def fn(**k):
        if counter is not None: counter.append(k)
        return result_or_fn(**k) if callable(result_or_fn) else result_or_fn
    t = registry.TOOL_REGISTRY[name]
    monkeypatch.setitem(registry.TOOL_REGISTRY, name, registry.Tool(fn, t.description, t.properties, t.required, t.needs_user))


QUOTE = ok({"ticker": "TCS", "last_close": 2370.5, "price_date": "2026-10-09"}, source="MarketStack",
           as_of="2026-10-09", warnings=["Latest close is 2 days old"])


def test_simple_question_uses_one_tool_and_reports_date(monkeypatch):
    calls = []
    stub_tool(monkeypatch, "get_stock_quote", QUOTE, calls)
    out = AgentOrchestrator(decide=scripted(call("get_stock_quote", ticker="TCS"),
                                            final("TCS last closed at 2370.5 on 2026-10-09."))).run("TCS price?", "u")
    assert len(calls) == 1 and out["status"] == "complete" and out["verified"] is True
    assert [t["tool"] for t in out["trace"]] == ["get_stock_quote"]


def test_wrong_tool_name_is_rejected_and_loop_recovers(monkeypatch):
    stub_tool(monkeypatch, "get_stock_quote", QUOTE)
    out = AgentOrchestrator(decide=scripted(call("execute_trade", ticker="TCS"),
                                            call("get_stock_quote", ticker="TCS"),
                                            final("TCS closed at 2370.5."))).run("buy TCS", "u")
    assert out["trace"][0]["ok"] is False and "UNKNOWN_TOOL" in out["trace"][0]["summary"]
    assert out["trace"][1]["ok"] is True and out["status"] == "complete"


def test_failed_api_is_retried_then_answer_states_limitation(monkeypatch):
    seq = iter([fail("API down", "NO_MARKET_DATA"), fail("API down", "NO_MARKET_DATA")])
    stub_tool(monkeypatch, "get_stock_quote", lambda **k: next(seq))
    out = AgentOrchestrator(decide=scripted(call("get_stock_quote", ticker="TCS"),
                                            call("get_stock_quote", ticker="TCS"),
                                            call("get_stock_quote", ticker="TCS"),
                                            final("I could not fetch a TCS quote."))).run("TCS?", "u")
    assert [t["ok"] for t in out["trace"]] == [False, False, False]
    assert "DUPLICATE_CALL" in out["trace"][2]["summary"]       # 3rd identical attempt blocked, API not hit
    assert out["tool_calls"] == 2


def test_unnecessary_duplicate_success_call_is_blocked(monkeypatch):
    calls = []
    stub_tool(monkeypatch, "get_stock_quote", QUOTE, calls)
    AgentOrchestrator(decide=scripted(call("get_stock_quote", ticker="TCS"),
                                      call("get_stock_quote", ticker="TCS"),
                                      final("TCS closed at 2370.5."))).run("TCS?", "u")
    assert len(calls) == 1


def test_fabricated_number_is_flagged(monkeypatch):
    stub_tool(monkeypatch, "get_stock_quote", QUOTE)
    out = AgentOrchestrator(decide=scripted(call("get_stock_quote", ticker="TCS"),
                                            final("TCS closed at 2370.5 and will hit 3100.75 soon."))).run("TCS?", "u")
    assert out["verified"] is False
    assert any("3100.75" in w for w in out["warnings"])


def test_answer_with_no_tool_data_is_flagged():
    out = AgentOrchestrator(decide=scripted(final("TCS trades at 4123.40 today."))).run("TCS?", "u")
    assert out["verified"] is False


def test_max_steps_and_tool_budget_stop_runaway_loop(monkeypatch):
    stub_tool(monkeypatch, "get_stock_quote", QUOTE)
    n = itertools.count()
    forever = lambda m, t: call("get_stock_quote", ticker=f"T{next(n)}")
    out = AgentOrchestrator(decide=forever, max_steps=5, max_tool_calls=3).run("loop", "u")
    assert out["status"] == "incomplete" and out["steps"] == 5 and out["tool_calls"] == 3


def test_missing_portfolio_yields_needs_input(monkeypatch, tmp_path):
    import state.portfolio_store as ps
    monkeypatch.setattr(ps, "DB_PATH", str(tmp_path / "p.db"))
    out = AgentOrchestrator(decide=scripted(call("get_portfolio"),
                                            final("Please upload your portfolio CSV or XLSX first."))).run("my risk?", "u:new")
    assert out["status"] == "needs_input"


def test_stale_data_surfaces_as_warning(monkeypatch):
    stale = ok({"last_close": 100.0}, as_of="2026-09-01", stale=True)
    stub_tool(monkeypatch, "get_stock_quote", stale)
    out = AgentOrchestrator(decide=scripted(call("get_stock_quote", ticker="TCS"), final("Last close 100.0."))).run("q", "u")
    assert any("Stale" in w for w in out["warnings"])


def test_planner_crash_returns_incomplete_not_exception():
    def bad(m, t): raise RuntimeError("openai down")
    out = AgentOrchestrator(decide=bad).run("q", "u")
    assert out["status"] == "incomplete" and "planner error" in out["answer"]


def test_unsupported_figures_ignores_counts_and_years():
    res = [{"args": {}, "result": ok({"x": 12.5})}]
    assert unsupported_figures("3 stocks in 2026, up 12.5%", res) == []
    assert unsupported_figures("up 99.9%", res) == ["99.9"]
