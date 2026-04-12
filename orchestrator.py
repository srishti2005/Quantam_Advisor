from agents.market_agent import MarketAgent
from agents.fundamentals_agent import FundamentalsAgent
from agents.risk_agent import RiskAgent
from agents.strategy_agent import StrategyAgent
from agents.portfolio_agent import PortfolioAgent
from agents.user_agent import UserAgent
from agents.report_agent import ReportAgent
from agents.news_sentiment_agent import NewsSentimentAgent   # ← NEW


class Orchestrator:

    def __init__(self):

        self.market    = MarketAgent()
        self.fund      = FundamentalsAgent()
        self.risk      = RiskAgent()
        self.strategy  = StrategyAgent()
        self.portfolio = PortfolioAgent()
        self.user      = UserAgent()
        self.report    = ReportAgent()
        self.sentiment = NewsSentimentAgent()                # ← NEW

    def analyze(self, ticker, user):

        # 1. User profiling
        u = self.user.analyze(user)

        # 2. Market data (price, RSI, trend, volatility)
        m = self.market.analyze(ticker)

        # 3. Fundamentals (PE, ROE, D/E, health score)
        f = self.fund.analyze(ticker)

        # 4. News sentiment                                  ← NEW (before risk)
        sent = self.sentiment.analyze(ticker)

        # 5. Risk assessment
        r = self.risk.analyze(user, m, f)

        # 6. Strategy / recommendation (sentiment passed in)
        s = self.strategy.analyze(user, m, f, r, sent)      # ← sentimentScore added

        # 7. Portfolio sizing advice
        p = self.portfolio.analyze(user, m, s)

        # 8. AI narrative report (sentiment passed in)
        rep = self.report.analyze(ticker, m, r, sent)       # ← was {}

        return {
            **u,
            **m,
            **f,
            **sent,   # ← exposes newsSentiment, topHeadlines, etc.
            **r,
            **s,
            **p,
            **rep,
        }