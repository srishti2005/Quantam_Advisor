"""Stock pipeline as a LangGraph: same output as the old sequential orchestrator, plus the new behaviour."""
import threading
import time
import pytest
from orchestrator import Orchestrator


class Stub:
    def __init__(self, fn, log, name, delay=0):
        self.fn, self.log, self.name, self.delay = fn, log, name, delay

    def analyze(self, *a):
        self.log.append((self.name, threading.current_thread().name, time.time()))
        time.sleep(self.delay)
        return self.fn(*a)


def make(market_ok=True, log=None, delay=0):
    log = log if log is not None else []
    o = Orchestrator()
    def market(t):
        if not market_ok: raise Exception("No data from any API")
        return {"ticker": t, "currentPrice": 100.0}
    o.user      = Stub(lambda u: {"userType": "Beginner Investor"}, log, "user")
    o.market    = Stub(market, log, "market")
    o.fund      = Stub(lambda t: {"pe": None}, log, "fund", delay)
    o.sentiment = Stub(lambda t: {"newsSentiment": "Neutral", "sentimentScore": 0}, log, "sent", delay)
    o.risk      = Stub(lambda u, m, f: {"riskScore": 20}, log, "risk")
    o.strategy  = Stub(lambda u, m, f, r, s: {"recommendation": "HOLD"}, log, "strategy")
    o.portfolio = Stub(lambda u, m, s: {"portfolioAdvice": ["x"]}, log, "portfolio")
    o.report    = Stub(lambda t, m, r, s: {"aiReport": "r"}, log, "report")
    return o, log


def legacy(o, ticker, user):
    """The old hardcoded sequence, verbatim logic."""
    u = o.user.analyze(user); m = o.market.analyze(ticker); f = o.fund.analyze(ticker)
    sent = o.sentiment.analyze(ticker); r = o.risk.analyze(user, m, f)
    s = o.strategy.analyze(user, m, f, r, sent); p = o.portfolio.analyze(user, m, s)
    rep = o.report.analyze(ticker, m, r, sent)
    return {**u, **m, **f, **sent, **r, **s, **p, **rep}


def test_graph_output_identical_to_legacy_sequence():
    o, _ = make()
    assert o.analyze("TCS", {"age": 30}) == legacy(o, "TCS", {"age": 30})


def test_market_failure_stops_early_and_raises():
    o, log = make(market_ok=False)
    with pytest.raises(Exception, match="No data from any API"):
        o.analyze("BAD", {})
    ran = {n for n, _, _ in log}
    assert not ran & {"fund", "sent", "risk", "strategy", "portfolio", "report"}


def test_risk_runs_once_after_both_branches():
    o, log = make()
    o.analyze("TCS", {})
    names = [n for n, _, _ in log]
    assert names.count("risk") == 1
    assert names.index("risk") > max(names.index("fund"), names.index("sent"), names.index("user"))
    assert names[-1] == "report"


def test_fundamentals_and_sentiment_run_in_parallel():
    o, _ = make(delay=0.3)
    t0 = time.time()
    o.analyze("TCS", {})
    assert time.time() - t0 < 0.55        # sequential would be >= 0.6s


def test_graph_shape():
    nodes = set(Orchestrator().graph.get_graph().nodes)
    assert {"profile_user", "fetch_market", "fundamentals", "sentiment", "risk", "strategy", "portfolio", "report"} <= nodes
    assert "flowchart" in Orchestrator().graph.get_graph().draw_mermaid()
