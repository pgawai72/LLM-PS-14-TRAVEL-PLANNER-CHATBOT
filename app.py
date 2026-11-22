"""Streamlit UI for the Travel Planner Chatbot (PS14). Run: streamlit run app.py"""
import pandas as pd
import streamlit as st

import planner as p

st.set_page_config(page_title="AI Travel Planner", page_icon="🧳", layout="wide")
st.title("🧳 AI Travel Planner Chatbot")
st.caption("Day-wise itinerary with cost estimates. Then chat to refine it.")

ss = st.session_state
ss.setdefault("itin", None)
ss.setdefault("chat", [])
ss.setdefault("budget", None)

with st.sidebar:
    st.header("⚙️ Model")
    label = st.selectbox("LLM provider", list(p.PROVIDERS))
    provider, model = p.PROVIDERS[label]
    key = st.text_input("API key (optional if set in .env)", type="password")
    st.divider()
    st.header("📝 Trip details")
    with st.form("trip"):
        dest = st.text_input("Destination", "Goa")
        days = st.slider("Number of days", 1, 10, 3)
        budget = st.number_input("Total budget (₹, whole group)", 2000, 1000000, 20000, step=1000)
        travelers = st.number_input("Travelers", 1, 20, 2)
        style = st.selectbox("Travel style", ["Budget", "Balanced", "Comfort", "Luxury"], index=1)
        interests = st.multiselect(
            "Interests",
            ["Beaches", "History & Heritage", "Food", "Adventure", "Nature", "Temples", "Shopping", "Nightlife", "Relaxation"],
            default=["Beaches", "Food"],
        )
        go = st.form_submit_button("Generate itinerary 🚀", width="stretch")

if go:
    if not dest.strip():
        st.error("Please enter a destination.")
    else:
        with st.spinner(f"Planning {days} days in {dest}..."):
            try:
                ss.itin = p.generate_itinerary(provider, model, dest.strip(), days, budget,
                                               interests, travelers, style, key or None)
                ss.budget, ss.chat = budget, []
            except p.PlannerError as e:
                st.error(str(e))

itin = ss.itin
if not itin:
    st.info("👈 Fill in the trip details in the sidebar and click **Generate itinerary**.")
    st.stop()

st.subheader(f"📍 {itin['destination']}")
st.write(itin["summary"])

total, bud = itin["total_cost_inr"], ss.budget
c1, c2, c3 = st.columns(3)
c1.metric("Estimated total", f"₹{total:,}")
c2.metric("Budget", f"₹{bud:,}")
c3.metric("Remaining", f"₹{bud - total:,}", delta_color="normal" if bud >= total else "inverse")
if total > bud:
    st.warning(f"Estimated cost is ₹{total - bud:,} over budget. Ask the chatbot below to make it cheaper.")

tabs = st.tabs([f"Day {d['day']}" for d in itin["days"]])
for tab, d in zip(tabs, itin["days"]):
    with tab:
        st.markdown(f"**{d['title']}**")
        rows = [{"Time": a["time"], "Place / Activity": a["place"],
                 "Details": a["description"], "Cost (₹)": a["cost_inr"]} for a in d["activities"]]
        rows.append({"Time": "All day", "Place / Activity": "Stay + food", "Details": "Hotel share and meals",
                     "Cost (₹)": d["stay_food_cost_inr"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(f"**Day total: ₹{d['day_total_inr']:,}**")

if itin["tips"]:
    with st.expander("💡 Travel tips"):
        for t in itin["tips"]:
            st.write("- " + t)

d1, d2, _ = st.columns([1, 1, 3])
d1.download_button("⬇️ Download .txt", p.to_text(itin, bud), "itinerary.txt", width="stretch")
d2.download_button("⬇️ Download PDF", p.to_pdf_bytes(itin, bud), "itinerary.pdf", "application/pdf", width="stretch")

st.divider()
st.subheader("💬 Chat with your planner")
st.caption('Try: "Make it cheaper", "Add a day trip", "Replace day 2 evening with something relaxing", "Is it safe for solo women?"')
for m in ss.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if msg := st.chat_input("Ask or request a change..."):
    ss.chat.append({"role": "user", "content": msg})
    with st.chat_message("user"):
        st.write(msg)
    with st.spinner("Thinking..."):
        try:
            reply, new = p.refine_itinerary(provider, model, itin, ss.chat[:-1], msg, key or None)
        except p.PlannerError as e:
            reply, new = f"⚠️ {e}", None
    ss.chat.append({"role": "assistant", "content": reply})
    if new:
        ss.itin = new
    st.rerun()
