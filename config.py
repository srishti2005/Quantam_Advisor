"""Settings come from environment variables - never hardcode keys in git.
Locally: set them in your shell or a .env file you load yourself (and .gitignore it)."""
import os

MARKETSTACK_API_KEY  = os.environ.get("MARKETSTACK_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
NEWS_API_KEY         = os.environ.get("NEWS_API_KEY", "")
OPENAI_API_KEY       = os.environ.get("OPENAI_API_KEY", "")

# Agent runtime limits
AGENT_MODEL          = os.environ.get("AGENT_MODEL", "gpt-4.1-mini")
AGENT_MAX_STEPS      = int(os.environ.get("AGENT_MAX_STEPS", "8"))
AGENT_MAX_TOOL_CALLS = int(os.environ.get("AGENT_MAX_TOOL_CALLS", "10"))
AGENT_TIME_BUDGET_S  = int(os.environ.get("AGENT_TIME_BUDGET_S", "60"))
TOOL_TIMEOUT_S       = int(os.environ.get("TOOL_TIMEOUT_S", "15"))
