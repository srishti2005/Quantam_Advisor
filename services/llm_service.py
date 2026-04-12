from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def ask_llm(prompt):

    try:

        r = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        return r.choices[0].message.content

    except Exception as e:

        print("LLM ERROR:", e)

        return "AI reasoning not available"