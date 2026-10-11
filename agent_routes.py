"""Flask blueprint for the agentic endpoint. user_id comes from the SERVER session only."""
import secrets, time
from collections import defaultdict, deque

from flask import Blueprint, jsonify, request, session

from agent_orchestrator import AgentOrchestrator
from state.portfolio_store import save_portfolio

agent_bp = Blueprint("agent", __name__)
_agent = AgentOrchestrator()
_hits = defaultdict(deque)
RATE_LIMIT, WINDOW_S = 10, 60          # requests per user per minute (in-memory; use Redis if you scale out)


def current_user_id():
    if session.get("user_email"):
        return "u:" + session["user_email"]
    if "anon_id" not in session:
        session["anon_id"] = secrets.token_hex(8)
    return "anon:" + session["anon_id"]


def save_user_portfolio(analysis: dict):
    """Call from /upload_portfolio after a successful analysis."""
    save_portfolio(current_user_id(), analysis)


@agent_bp.route("/api/agent", methods=["POST"])
def agent_query():
    uid = current_user_id()
    q = deque(_hits[uid]); now = time.time()
    while q and now - q[0] > WINDOW_S:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return jsonify({"error": "Too many requests, try again in a minute"}), 429
    q.append(now); _hits[uid] = q

    data = request.get_json(silent=True) or {}
    query = data.get("query")
    if not isinstance(query, str) or not 3 <= len(query.strip()) <= 500:
        return jsonify({"error": "query must be a string of 3-500 characters"}), 400
    # NOTE: any user_id in the request body is deliberately ignored.
    return jsonify(_agent.run(query.strip(), uid))
