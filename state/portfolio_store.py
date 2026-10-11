"""Per-user portfolio storage. user_id ALWAYS comes from the server session, never the client."""
import json, os, sqlite3, time

DB_PATH = os.environ.get(
    "PORTFOLIO_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "portfolios.db"))


def _db():
    c = sqlite3.connect(DB_PATH)
    c.execute("CREATE TABLE IF NOT EXISTS portfolios "
              "(user_id TEXT PRIMARY KEY, payload TEXT NOT NULL, uploaded_at REAL NOT NULL)")
    return c


def save_portfolio(user_id: str, analysis: dict):
    keep = {k: analysis.get(k) for k in ("stocks", "totalValue", "totalCost", "totalPnL", "totalPnLPct")}
    with _db() as c:
        c.execute("INSERT OR REPLACE INTO portfolios VALUES (?,?,?)",
                  (user_id, json.dumps(keep), time.time()))


def load_portfolio(user_id: str):
    with _db() as c:
        row = c.execute("SELECT payload, uploaded_at FROM portfolios WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        return None
    data = json.loads(row[0])
    data["uploaded_at"] = row[1]
    return data
