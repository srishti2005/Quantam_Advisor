class RiskAgent:

    def analyze(self, user, market, fund):

        risk = 0
        factors = []

        rsi = market["rsi"]
        vol = market["volatility"]
        trend = market["trend"]

        # volatility
        if vol > 3:
            risk += 20
            factors.append("High volatility")
        elif vol > 1:
            risk += 10
        else:
            risk += 5

        # RSI
        if rsi > 70:
            risk += 10
            factors.append("Overbought")
        elif rsi < 30:
            risk += 8
            factors.append("Oversold")
        else:
            risk += 4

        # trend
        if trend == "Downtrend":
            risk += 10
        elif trend == "Sideways":
            risk += 6
        else:
            risk += 3

        risk = max(5, min(100, risk))

        if risk < 25:
            level = "Low-Medium"
        elif risk < 50:
            level = "Medium"
        else:
            level = "High"

        return {

            "riskScore": risk,
            "riskLevel": level,
            "riskFactors": factors,
        }