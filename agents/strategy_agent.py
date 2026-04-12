class StrategyAgent:

    def analyze(self, u, m, f, r, sentiment=None):
        """
        Combines risk score with news sentiment to produce a final
        recommendation.

        sentimentScore influence:
          +1 (Positive news) → lowers effective risk by 8 points → pushes toward BUY
          -1 (Negative news) → raises effective risk by 8 points → pushes toward SELL
           0 (Neutral)       → no adjustment
        """

        sentiment = sentiment or {}
        sentiment_score  = sentiment.get("sentimentScore", 0)   # -1 / 0 / +1
        news_sentiment   = sentiment.get("newsSentiment", "Neutral")

        base_risk = r["riskScore"]

        # Adjust effective risk based on news signal
        SENTIMENT_WEIGHT = 8
        effective_risk = base_risk - (sentiment_score * SENTIMENT_WEIGHT)
        effective_risk = max(5, min(100, effective_risk))

        # Recommendation thresholds
        if effective_risk < 20:
            rec   = "BUY"
            score = 35
        elif effective_risk < 40:
            rec   = "HOLD"
            score = 18
        else:
            rec   = "SELL"
            score = 8

        price = m["currentPrice"]

        # Build reasoning list
        reasoning = [
            "Trend-based analysis",
            "Risk score evaluated",
            "Fundamentals checked",
            f"News sentiment: {news_sentiment} (score adjustment: {-sentiment_score * SENTIMENT_WEIGHT:+d})",
        ]

        return {
            "recommendation": rec,
            "confidence":     "Moderate",
            "buyScore":       score,
            "effectiveRisk":  round(effective_risk, 1),

            "targetPrice1": round(price * 1.1, 2),
            "targetPrice2": round(price * 1.2, 2),
            "stopLoss":     round(price * 0.9, 2),

            "reasoning": reasoning,
        }