"""Stock-analysis pipeline as a LangGraph StateGraph.

    START -> profile_user ----------------------------+
          -> fetch_market -(ok)-> fundamentals --+    |
                          |  (ok)-> sentiment ---+--> risk -> strategy -> portfolio -> report -> END
                          +-(market failed)-> END

* fundamentals and sentiment run IN PARALLEL (sentiment is a network call).
* risk waits for profile_user, fundamentals and sentiment (a join).
* if market data is unavailable the graph stops early instead of running agents on nothing.
analyze() keeps the old signature and output, so app.py and the frontend are unchanged.
"""
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from agents.market_agent import MarketAgent
from agents.fundamentals_agent import FundamentalsAgent
from agents.risk_agent import RiskAgent
from agents.strategy_agent import StrategyAgent
from agents.portfolio_agent import PortfolioAgent
from agents.user_agent import UserAgent
from agents.report_agent import ReportAgent
from agents.news_sentiment_agent import NewsSentimentAgent


class PipelineState(TypedDict, total=False):
    ticker: str
    user: dict
    u: dict       # user profile
    m: dict       # market
    f: dict       # fundamentals
    sent: dict    # news sentiment
    r: dict       # risk
    s: dict       # strategy
    p: dict       # portfolio sizing
    rep: dict     # report
    error: Optional[str]


class Orchestrator:

    def __init__(self):
        self.market    = MarketAgent()
        self.fund      = FundamentalsAgent()
        self.risk      = RiskAgent()
        self.strategy  = StrategyAgent()
        self.portfolio = PortfolioAgent()
        self.user      = UserAgent()
        self.report    = ReportAgent()
        self.sentiment = NewsSentimentAgent()
        self.graph     = self._build_graph()

    # ---------------------------------------------------------------- nodes
    def _profile_user(self, s):
        return {"u": self.user.analyze(s["user"])}

    def _fetch_market(self, s):
        try:
            return {"m": self.market.analyze(s["ticker"])}
        except Exception as e:                      # e.g. "No data from any API"
            return {"error": str(e)}

    def _fundamentals(self, s):
        return {"f": self.fund.analyze(s["ticker"])}

    def _sentiment(self, s):
        return {"sent": self.sentiment.analyze(s["ticker"])}

    def _risk(self, s):
        return {"r": self.risk.analyze(s["user"], s["m"], s["f"])}

    def _strategy(self, s):
        return {"s": self.strategy.analyze(s["user"], s["m"], s["f"], s["r"], s["sent"])}

    def _portfolio(self, s):
        return {"p": self.portfolio.analyze(s["user"], s["m"], s["s"])}

    def _report(self, s):
        return {"rep": self.report.analyze(s["ticker"], s["m"], s["r"], s["sent"])}

    # ---------------------------------------------------------------- edges
    @staticmethod
    def _after_market(s):
        return ["fundamentals", "sentiment"] if not s.get("error") else END

    def _build_graph(self):
        g = StateGraph(PipelineState)
        g.add_node("profile_user", self._profile_user)
        g.add_node("fetch_market", self._fetch_market)
        g.add_node("fundamentals", self._fundamentals)
        g.add_node("sentiment",    self._sentiment)
        g.add_node("risk",         self._risk)
        g.add_node("strategy",     self._strategy)
        g.add_node("portfolio",    self._portfolio)
        g.add_node("report",       self._report)

        g.add_edge(START, "profile_user")
        g.add_edge(START, "fetch_market")
        g.add_conditional_edges("fetch_market", self._after_market, ["fundamentals", "sentiment", END])
        g.add_edge(["profile_user", "fundamentals", "sentiment"], "risk")   # join
        g.add_edge("risk", "strategy")
        g.add_edge("strategy", "portfolio")
        g.add_edge("portfolio", "report")
        g.add_edge("report", END)
        return g.compile()

    # ---------------------------------------------------------------- public
    def analyze(self, ticker, user):
        out = self.graph.invoke({"ticker": ticker, "user": user})
        if out.get("error"):
            raise Exception(out["error"])
        return {**out["u"], **out["m"], **out["f"], **out["sent"],
                **out["r"], **out["s"], **out["p"], **out["rep"]}
