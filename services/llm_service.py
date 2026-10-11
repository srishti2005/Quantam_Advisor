from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def ask_llm(prompt):
    try:
        if client is None:
            return "AI reasoning not available"
        r = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content
    except Exception as e:
        print("LLM ERROR:", e)
        return "AI reasoning not available"


def chat_with_tools(messages, tools, model, timeout=30):
    """One planner step. Returns the raw assistant message (may contain tool_calls)."""
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set")
    r = client.chat.completions.create(
        model=model, messages=messages, tools=tools,
        tool_choice="auto", temperature=0, timeout=timeout,
    )
    return r.choices[0].message
