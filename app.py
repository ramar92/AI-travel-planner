import streamlit as st
from datetime import date, timedelta

from agent import TravelAgent

st.set_page_config(page_title="AI Travel Planner", page_icon="🧭", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — Groq credentials & model settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧭 Settings")

    api_key = st.text_input(
        "Groq API key",
        type="password",
        value=st.session_state.get("groq_api_key", ""),
        help="Get a free key at https://console.groq.com/keys",
    )
    st.session_state["groq_api_key"] = api_key

    model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
        index=0,
        help="Larger models plan more thoroughly but are slower.",
    )

    show_agent_steps = st.checkbox("Show agent's tool calls", value=True)

    st.markdown("---")
    st.caption(
        "This app uses an **agentic** loop: the LLM decides when to call "
        "live tools (weather, city facts, currency conversion, budget math) "
        "before writing your itinerary. No API keys besides Groq are needed — "
        "the other tools use free public endpoints."
    )

# ---------------------------------------------------------------------------
# Main form — trip brief
# ---------------------------------------------------------------------------
st.title("✈️ AI Travel Planner")
st.write("Tell the agent about your trip and it will research and build a real itinerary.")

with st.form("trip_form"):
    col1, col2 = st.columns(2)
    with col1:
        destination = st.text_input("Destination(s)", placeholder="e.g. Kyoto, Japan")
        start_date = st.date_input("Start date", value=date.today() + timedelta(days=30))
        end_date = st.date_input("End date", value=date.today() + timedelta(days=37))
        num_travelers = st.number_input("Number of travelers", min_value=1, max_value=20, value=2)
    with col2:
        origin = st.text_input("Departing from", placeholder="e.g. Chicago, USA")
        budget_amount = st.number_input("Total budget", min_value=0, value=3000, step=100)
        budget_currency = st.text_input("Budget currency (3-letter code)", value="USD", max_chars=3)
        pace = st.select_slider("Pace", options=["Relaxed", "Balanced", "Packed"], value="Balanced")

    interests = st.multiselect(
        "Interests",
        [
            "History & culture", "Food & drink", "Nature & hiking", "Art & museums",
            "Nightlife", "Shopping", "Beaches", "Architecture", "Family-friendly",
            "Off the beaten path", "Photography", "Relaxation / spa",
        ],
        default=["Food & drink", "History & culture"],
    )

    extra_notes = st.text_area(
        "Anything else the agent should know?",
        placeholder="e.g. traveling with a toddler, vegetarian, mobility considerations, must-see landmark...",
    )

    submitted = st.form_submit_button("Plan my trip 🪄", use_container_width=True)

# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------
if submitted:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar first.")
    elif not destination:
        st.error("Please enter a destination.")
    elif end_date <= start_date:
        st.error("End date must be after start date.")
    else:
        num_days = (end_date - start_date).days

        brief = f"""
Plan a trip with the following details:
- Destination(s): {destination}
- Departing from: {origin or "not specified"}
- Dates: {start_date.isoformat()} to {end_date.isoformat()} ({num_days} days)
- Number of travelers: {num_travelers}
- Total budget: {budget_amount} {budget_currency or "USD"}
- Preferred pace: {pace}
- Interests: {", ".join(interests) if interests else "no strong preferences"}
- Additional notes: {extra_notes or "none"}

Research real weather and city facts for the destination(s) via your tools,
and use calculate_budget to check the plan fits the stated budget (convert
currencies if needed). Then produce the final itinerary as described in your
instructions.
""".strip()

        agent = TravelAgent(api_key=api_key, model=model)

        result_placeholder = st.empty()
        step_container = st.container()
        final_markdown = None

        with st.status("Agent is planning your trip...", expanded=show_agent_steps) as status:
            for event in agent.plan(brief):
                if event["type"] == "tool_call":
                    args_str = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
                    status.write(f"🔧 Calling `{event['name']}({args_str})`")
                elif event["type"] == "tool_result":
                    if show_agent_steps:
                        with step_container.expander(f"Result of `{event['name']}`", expanded=False):
                            st.json(event["result"])
                elif event["type"] == "final":
                    final_markdown = event["content"]
                    status.update(label="Itinerary ready!", state="complete")
                elif event["type"] == "error":
                    status.update(label="Something went wrong", state="error")
                    st.error(event["message"])

        if final_markdown:
            st.markdown("---")
            st.markdown(final_markdown)
            st.download_button(
                "⬇️ Download itinerary as Markdown",
                data=final_markdown,
                file_name=f"itinerary_{destination.replace(' ', '_').replace(',', '')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
else:
    st.info("Fill in the trip details above and click **Plan my trip** to get started.")
