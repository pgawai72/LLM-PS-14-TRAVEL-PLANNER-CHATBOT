# PS14 – Travel Planner Chatbot (LLM Mini Project)

**Problem statement:** Planning trips takes hours of searching. This app generates a day-wise itinerary
with estimated costs from a destination, budget, number of days and interests, then lets the user refine
it by chatting ("make it cheaper", "add a day trip"). Itinerary can be downloaded as text or PDF.

## Features
- Form: destination, days, budget (INR), travelers, style, interests
- LLM returns strict JSON -> shown as one table per day (time, place, details, cost)
- Python recomputes all totals (LLMs make arithmetic mistakes) and warns if over budget
- Chat follow-up that updates the itinerary or answers questions
- Download as `.txt` and `.pdf`
- Works with Groq (Llama 3.3 70B) or Gemini 2.5 Flash

## Run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add GROQ_API_KEY or GEMINI_API_KEY
streamlit run app.py
```
Test: `pytest -q`

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `planner.py` | Prompts, LLM calls, JSON parsing, cost totals, TXT/PDF export |
| `tests/test_planner.py` | Unit tests (no API key needed) |

## Screenshots
Add 1–2 screenshots here.
