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
