"""Shared state for ONE analysis request (not persisted)."""
import time
from dataclasses import dataclass, field


def ok(data=None, source=None, as_of=None, stale=False, warnings=None):
    return {"ok": True, "data": data or {}, "error": None, "error_code": None,
            "source": source, "as_of": as_of, "stale": stale, "warnings": warnings or []}


def fail(error, code="ERROR", warnings=None):
    return {"ok": False, "data": {}, "error": str(error), "error_code": code,
            "source": None, "as_of": None, "stale": False, "warnings": warnings or []}


@dataclass
class AgentState:
    user_query: str
    user_id: str
    messages: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)   # [{step, tool, args, result, ms}]
    missing_data: list = field(default_factory=list)
    iteration: int = 0
    tool_calls_used: int = 0
    status: str = "running"        # running | complete | incomplete | needs_input
    started: float = field(default_factory=time.time)

    def elapsed(self):
        return time.time() - self.started
