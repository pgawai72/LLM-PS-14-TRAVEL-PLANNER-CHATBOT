"""Core logic for the Travel Planner Chatbot (PS14).

Flow: user inputs -> build prompt -> call LLM (Groq or Gemini) -> parse JSON
-> Python recomputes all totals (LLMs are unreliable at arithmetic).
"""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert Indian travel planner. Reply with ONLY valid JSON, no markdown fences, no extra text.
Plan realistic, geographically sensible days (group nearby places together).
All costs are in Indian Rupees (INR), integers, per whole group (not per person).
Use this exact schema:
{
  "destination": "string",
  "summary": "2 sentence overview of the trip",
  "days": [
    {
      "day": 1,
      "title": "short theme of the day",
      "activities": [
        {"time": "Morning|Afternoon|Evening|Night",
         "place": "name of place or activity",
         "description": "one short line",
         "cost_inr": 0}
      ],
      "stay_food_cost_inr": 0
    }
  ],
  "tips": ["3 to 5 short practical tips"]
}
Rules:
- Exactly the requested number of days, 3 to 5 activities per day.
- cost_inr = entry fees + local transport + activity cost. stay_food_cost_inr = hotel share + meals for that day.
- Match the interests and travel style. Keep the total close to (not above) the budget.
- If the budget is too low for the destination, still plan the cheapest realistic trip and say so in "summary"."""

REFINE_PROMPT = """You are a travel planner chatbot. The user has an existing itinerary (JSON) and sends a message.
Reply with ONLY valid JSON of the form:
{"reply": "short friendly answer (max 3 sentences)", "itinerary": <full updated itinerary in the SAME schema, or null if the message needs no change>}
If the user asks to change the plan (cheaper, add a day, swap an activity, more adventure...), return the whole updated itinerary.
If they only ask a question, answer in "reply" and set "itinerary" to null."""

PROVIDERS = {
    "Groq (GPT-OSS 120B)": ("groq", "openai/gpt-oss-120b"),
    "Groq (GPT-OSS 20B, faster)": ("groq", "openai/gpt-oss-20b"),
    "Gemini 2.5 Flash": ("gemini", "gemini-2.5-flash"),
}


class PlannerError(Exception):
    """Raised for any user-facing failure (missing key, bad JSON, API error)."""


def build_user_prompt(destination, days, budget, interests, travelers, style):
    return (
        f"Destination: {destination}\n"
        f"Number of days: {days}\n"
        f"Total budget for the whole group: INR {budget}\n"
        f"Travelers: {travelers}\n"
        f"Travel style: {style}\n"
        f"Interests: {', '.join(interests) if interests else 'general sightseeing'}\n"
        "Create the itinerary now."
    )


def call_llm(provider, model, system, messages, api_key=None):
    """messages: list of {"role": "user"|"assistant", "content": str}. Returns raw text."""
    try:
        if provider == "groq":
            from groq import Groq

            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise PlannerError("GROQ_API_KEY not set. Add it to .env or the sidebar.")
            resp = Groq(api_key=key).chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}] + messages,
                response_format={"type": "json_object"},
                temperature=0.5,
            )
            return resp.choices[0].message.content
        if provider == "gemini":
            from google import genai
            from google.genai import types

            key = api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                raise PlannerError("GEMINI_API_KEY not set. Add it to .env or the sidebar.")
            contents = [
                types.Content(
                    role="user" if m["role"] == "user" else "model",
                    parts=[types.Part(text=m["content"])],
                )
                for m in messages
            ]
            resp = genai.Client(api_key=key).models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    temperature=0.5,
                ),
            )
            return resp.text
        raise PlannerError(f"Unknown provider: {provider}")
    except PlannerError:
        raise
    except Exception as e:  # network, quota, auth ...
        raise PlannerError(f"LLM call failed: {e}") from e


def parse_json(text):
    """Parse model output into a dict, tolerating ```json fences or stray text."""
    if not text:
        raise PlannerError("Empty response from the model.")
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise PlannerError("Model did not return valid JSON. Please try again.")


def _to_int(v):
    try:
        return max(0, int(float(str(v).replace(",", "").replace("₹", "").strip())))
    except (ValueError, TypeError):
        return 0


def normalize(itin):
    """Validate the schema and recompute every total in Python."""
    if not isinstance(itin, dict) or not isinstance(itin.get("days"), list) or not itin["days"]:
        raise PlannerError("Itinerary is missing the 'days' list.")
    total = 0
    for i, d in enumerate(itin["days"], start=1):
        d["day"] = i
        d.setdefault("title", f"Day {i}")
        acts = d.get("activities") or []
        for a in acts:
            a["time"] = a.get("time", "")
            a["place"] = a.get("place", "")
            a["description"] = a.get("description", "")
            a["cost_inr"] = _to_int(a.get("cost_inr"))
        d["activities"] = acts
        d["stay_food_cost_inr"] = _to_int(d.get("stay_food_cost_inr"))
        d["day_total_inr"] = sum(a["cost_inr"] for a in acts) + d["stay_food_cost_inr"]
        total += d["day_total_inr"]
    itin["total_cost_inr"] = total
    itin.setdefault("destination", "")
    itin.setdefault("summary", "")
    itin["tips"] = itin.get("tips") or []
    return itin


def generate_itinerary(provider, model, destination, days, budget, interests,
                       travelers, style, api_key=None):
    prompt = build_user_prompt(destination, days, budget, interests, travelers, style)
    raw = call_llm(provider, model, SYSTEM_PROMPT, [{"role": "user", "content": prompt}], api_key)
    return normalize(parse_json(raw))


def refine_itinerary(provider, model, itinerary, history, user_message, api_key=None):
    """Chat follow-up. Returns (reply_text, new_itinerary_or_None)."""
    context = "Current itinerary JSON:\n" + json.dumps(
        {k: v for k, v in itinerary.items() if k not in ("total_cost_inr",)}, ensure_ascii=False
    )
    msgs = [{"role": "user", "content": context}, {"role": "assistant", "content": '{"reply":"Got it.","itinerary":null}'}]
    msgs += history[-6:]
    msgs.append({"role": "user", "content": user_message})
    data = parse_json(call_llm(provider, model, REFINE_PROMPT, msgs, api_key))
    new = data.get("itinerary")
    return data.get("reply", "Done."), (normalize(new) if new else None)


def to_text(itin, budget=None):
    lines = [f"TRAVEL ITINERARY: {itin['destination']}", "=" * 50, itin["summary"], ""]
    for d in itin["days"]:
        lines.append(f"Day {d['day']}: {d['title']}")
        for a in d["activities"]:
            lines.append(f"  [{a['time']}] {a['place']} - {a['description']} (Rs. {a['cost_inr']})")
        lines.append(f"  Stay + food: Rs. {d['stay_food_cost_inr']}")
        lines.append(f"  Day total: Rs. {d['day_total_inr']}")
        lines.append("")
    lines.append(f"ESTIMATED TOTAL: Rs. {itin['total_cost_inr']}" + (f" (budget Rs. {budget})" if budget else ""))
    if itin["tips"]:
        lines += ["", "Tips:"] + [f"  - {t}" for t in itin["tips"]]
    return "\n".join(lines)


def to_pdf_bytes(itin, budget=None):
    from fpdf import FPDF

    def clean(s):  # core PDF fonts are latin-1 only
        return str(s).replace("₹", "Rs.").encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean(f"Travel Itinerary: {itin['destination']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, clean(itin["summary"]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    for d in itin["days"]:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, clean(f"Day {d['day']}: {d['title']}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for a in d["activities"]:
            pdf.multi_cell(0, 5, clean(f"- [{a['time']}] {a['place']}: {a['description']} (Rs. {a['cost_inr']})"),
                           new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 5, clean(f"Stay + food: Rs. {d['stay_food_cost_inr']}   |   Day total: Rs. {d['day_total_inr']}"),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    tot = f"Estimated total: Rs. {itin['total_cost_inr']}" + (f" (budget Rs. {budget})" if budget else "")
    pdf.cell(0, 8, clean(tot), new_x="LMARGIN", new_y="NEXT")
    if itin["tips"]:
        pdf.set_font("Helvetica", "", 10)
        for t in itin["tips"]:
            pdf.multi_cell(0, 5, clean(f"* {t}"), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
