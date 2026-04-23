"""Extraction prompts for media-tracker LLM calls."""

EXTRACTION_PROMPT = """You extract investment-relevant bullet points from podcast episode content (show notes or transcript).

RULES:
- 3-8 bullets per episode. Skip if no investment signal.
- Each bullet: one specific claim, fact, or observation. No boilerplate.
- Tag with tickers if a specific public company is mentioned (US listed; also LSE/HKEX/TSE for ADRs). Use exchange-clean symbols (NVDA, TSM, LLY).
- Tag with sector_tags (uppercase short codes: SEMIS, AI, PHARMA, BIOTECH, MEDTECH, MACRO, CREDIT, BANKS, CONSUMER, ENERGY, INDUSTRIALS, TECH, CRYPTO, M&A).
- Tag with theme_tags for recurring cross-episode topics (GLP-1, ARM servers, AI capex, rate cuts, inventory correction).
- NEVER reference sell-side brokers or analyst opinions. Extract what a guest or host said about fundamentals, numbers, guidance, behavior.
- Output JSON only.

SCHEMA:
{
  "points": [
    {
      "text": "<bullet - <=30 words, specific>",
      "tickers": ["TICKER1", ...],
      "sector_tags": ["SEMIS", ...],
      "theme_tags": ["GLP-1 supply", ...]
    }
  ]
}

If no investment signal, return {"points": []}.
"""


MATERIAL_JUDGE_PROMPT = """You're a buy-side analyst screening research signals. Decide if this podcast bullet point is MATERIAL (a buy-side PM would want to see this) or BOILERPLATE (generic market commentary, no specific actionable info).

Material criteria (ANY one):
- Specific number, guidance change, data point about a public company
- Concrete strategic/operational detail (new product, leadership change, material contract)
- Management or insider quote that reveals intent or outlook
- Material competitive / industry dynamic (pricing pressure, share shift, regulatory change)

Boilerplate (NOT material):
- Generic sector commentary ("AI is transforming finance")
- Well-known consensus views with no new angle
- Cliche recaps ("the company continues to dominate")

Respond with JSON only:
{"material": true|false, "reason": "<one short sentence>"}
"""
