"""
agent.py
--------
A small, transparent agent loop:

    1. Send the conversation + tool schemas to Groq.
    2. If the model responds with tool_calls, execute each one locally
       and feed the results back in as "tool" messages.
    3. Repeat until the model returns a plain text answer (or we hit a
       safety cap on iterations).

This is a generator so the Streamlit UI can show live progress
("Agent is calling get_weather(city='Lisbon')...") instead of one long
silent wait.
"""

import json
from groq import Groq
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

MAX_ITERATIONS = 8

SYSTEM_PROMPT = """You are an expert, meticulous travel-planning agent.

You have tools available to fetch real weather forecasts, real city facts,
live currency conversion rates, and to compute exact budget math. Use them
whenever a fact, number, or forecast would improve the plan — never guess
at a number you could instead calculate or look up.

When you have gathered what you need, produce a final answer that is a
complete, well-formatted Markdown travel itinerary with:
- A short intro (2-3 sentences) grounded in real facts about the destination
- A day-by-day plan (morning / afternoon / evening) tailored to the
  traveler's interests and pace
- Practical notes: weather/packing advice, and a budget breakdown table
  (use calculate_budget for the total; do not hand-compute it)
- A short "tips" section (local customs, transport, safety)

Be concrete: name real neighborhoods, real landmarks, real dish names.
Keep the tone warm and practical, like a knowledgeable friend, not generic
marketing copy.
"""


class TravelAgent:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def plan(self, user_brief: str):
        """Generator yielding progress events while planning a trip.

        Yields dicts of shape:
          {"type": "tool_call", "name": ..., "args": {...}}
          {"type": "tool_result", "name": ..., "result": {...}}
          {"type": "final", "content": "...markdown itinerary..."}
          {"type": "error", "message": "..."}
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_brief},
        ]

        for _ in range(MAX_ITERATIONS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0.6,
                    max_tokens=4096,
                )
            except Exception as e:
                yield {"type": "error", "message": str(e)}
                return

            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                yield {"type": "final", "content": msg.content or ""}
                return

            # Record the assistant's tool-call turn
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                yield {"type": "tool_call", "name": name, "args": args}

                func = TOOL_FUNCTIONS.get(name)
                if func is None:
                    result = {"error": f"Unknown tool '{name}'"}
                else:
                    try:
                        result = func(**args)
                    except Exception as e:
                        result = {"error": str(e)}

                yield {"type": "tool_result", "name": name, "result": result}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

        yield {"type": "error", "message": "Agent hit the maximum number of steps without finishing. Try simplifying the request."}
