#!/usr/bin/env python3
"""Charlie Local Agent -- processes note generation jobs using local iCloud files.

Runs on the user's Mac, polls the Charlie backend for pending note jobs,
reads source files from iCloud, calls Claude API, generates full research
notes with charts, and saves outputs locally + uploads to Charlie.

Usage:
    python3 charlie_local_agent.py                 # Run agent daemon
    python3 charlie_local_agent.py --api-key sk-...# Run with explicit API key
    python3 charlie_local_agent.py --once           # Process one job and exit
    python3 charlie_local_agent.py --ticker MDT     # Process specific ticker (no backend job needed)
    python3 charlie_local_agent.py --debug          # Verbose logging
"""

import os
import sys
import json
import base64
import io
import re
import time
import shutil
import signal
import argparse
import logging
import threading
import uuid
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional
from xml.etree import ElementTree

import requests
import anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOCKS_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/STOCKS"
CHARLIE_API = "https://equity-analyzer-backend.onrender.com"
POLL_INTERVAL = 5  # seconds
CONFIG_FILE = Path.home() / ".charlie_agent_config.json"
AGENT_SECRET_HEADER = "X-Agent-Secret"

log = logging.getLogger("charlie_agent")

# ---------------------------------------------------------------------------
# Sector map (mirrors backend PIPELINE_SECTOR_MAP / PIPELINE_TICKER_SECTOR)
# ---------------------------------------------------------------------------

PIPELINE_SECTOR_MAP = {
    "Pharma": ["MRK", "LLY", "JNJ", "PFE", "BMY", "ABBV"],
    "Biotech": ["GILD", "REGN", "VRTX"],
    "MedTech & Life Sci": ["MDT", "BDX", "BSX", "ABT", "TMO", "DHR", "DGX"],
    "Managed Care & Distro": ["UNH", "CI", "CVS", "COR", "CAH"],
    "Industrials": ["RTX", "GD", "CARR", "ETN", "DE", "PH", "DOV", "MMM", "RSG", "UNP", "NSC"],
    "REITs": ["VTR", "PLD", "AVB", "AMT"],
}

PIPELINE_TICKER_SECTOR: dict[str, str] = {}
for _sec, _tks in PIPELINE_SECTOR_MAP.items():
    for _tk in _tks:
        PIPELINE_TICKER_SECTOR[_tk] = _sec


def get_sector(ticker: str) -> str:
    return PIPELINE_TICKER_SECTOR.get(ticker, "Other")


# ---------------------------------------------------------------------------
# Playbook (matches backend RESEARCH_NOTE_PLAYBOOK exactly)
# ---------------------------------------------------------------------------

RESEARCH_NOTE_PLAYBOOK = """
## NOTE FORMAT RULES

### Structure (adapt emphasis per stock, but include all sections):
1. Executive Summary / Investment Thesis (bull + bear in 3-4 bullets each)
2. Business Overview & Segment Breakdown
3. Key Revenue & Earnings Drivers
4. What the Street Is Debating (the 2-3 key open questions)
5. Catalyst Calendar (earnings, FDA dates, contract renewals, etc.)
6. Valuation Context (vs. history, vs. peers)
7. Risks
8. Bottom Line: Own / Avoid / Revisit at $X

### Sector-specific additions:
- Pharma/Biotech: Patent cliffs, pipeline table, LOE timeline
- MedTech: Procedure volume trends, ASP dynamics, new product cycles
- Managed Care: Membership trends, MLR, PBM reform risk, star ratings
- Distribution: Drug pricing dynamics, biosimilar opportunity, generic deflation
- Industrials: Cycle positioning, book-to-bill, aftermarket mix, margin expansion
- REITs: Same-store NOI, occupancy, cap rates, lease spreads, FFO/AFFO

### STRICT RULES:
- NEVER reference specific analyst names, firms, or broker ratings
- Synthesize data points from reports but attribute nothing to specific brokers
- Hard facts (reported financials, FDA approvals, deals, guidance) = state directly
- Sellside opinions: include if valuable but do NOT attribute
- Do NOT use sellside sentiment, consensus ratings, or target ranges as reasons to buy/sell
- The investment thesis must stand on fundamentals alone
- NEVER attribute headline YoY EPS growth to a single narrative driver without decomposing the bridge
- Cross-check every narrative claim against data tables
- When FY EPS includes >$0.50/share in non-recurring items, flag explicitly
- Distinguish reported vs underlying/organic growth rates
- Nudge displayed numbers +/- $200-300M from model to avoid exact replication of broker data

### TONE:
- Write as if the analyst is authoring the note to their PM
- Confident, concise, first-person where appropriate
- No hedging language
- Lead with conclusion, support with evidence
- Use precise numbers, no rounding
- No emojis, no filler
"""

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load persisted config from ~/.charlie_agent_config.json."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def save_config(cfg: dict) -> None:
    """Persist config to ~/.charlie_agent_config.json."""
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def load_api_key_from_config() -> Optional[str]:
    return load_config().get("api_key")


def get_agent_secret() -> Optional[str]:
    """Return agent secret for authenticating with backend agent endpoints."""
    return os.environ.get("CHARLIE_AGENT_SECRET") or load_config().get("agent_secret")


# ---------------------------------------------------------------------------
# Backend communication
# ---------------------------------------------------------------------------


def _agent_headers() -> dict:
    secret = get_agent_secret()
    headers = {"Content-Type": "application/json"}
    if secret:
        headers[AGENT_SECRET_HEADER] = secret
    return headers


def poll_for_jobs() -> list[dict]:
    """Poll backend for pending note generation jobs."""
    try:
        resp = requests.get(
            f"{CHARLIE_API}/api/pipeline/jobs",
            params={"status": "queued", "limit": 10},
            headers=_agent_headers(),
            timeout=15,
        )
        if resp.status_code == 200:
            jobs = resp.json().get("jobs", [])
            # Filter for note-type jobs only
            return [j for j in jobs if j.get("job_type") == "note"]
    except requests.exceptions.ConnectionError:
        log.warning("Backend unreachable -- will retry")
    except Exception as e:
        log.warning(f"Poll error: {e}")
    return []


def claim_job(job_id: str) -> bool:
    """Attempt to claim a job via the agent claim endpoint.

    Falls back to direct progress update if the agent endpoint does not exist.
    """
    try:
        resp = requests.post(
            f"{CHARLIE_API}/api/agent/claim",
            json={"jobId": job_id, "agentId": _agent_id()},
            headers=_agent_headers(),
            timeout=15,
        )
        if resp.status_code == 200:
            return True
        # Fallback: use the existing progress update mechanism
        log.debug(f"Claim endpoint returned {resp.status_code}, falling back to progress update")
    except Exception as e:
        log.debug(f"Claim endpoint unavailable ({e}), using fallback")

    # Fallback: update job directly
    return _update_job_progress_fallback(job_id, "running", "Claimed by local agent", 1)


def update_job_progress(
    job_id: str,
    status: str,
    step: str,
    progress: Optional[int],
    error: Optional[str] = None,
) -> None:
    """Update pipeline job progress on Charlie backend."""
    try:
        resp = requests.post(
            f"{CHARLIE_API}/api/agent/update-job",
            json={
                "jobId": job_id,
                "status": status,
                "currentStep": step,
                "progress": progress,
                "error": error,
            },
            headers=_agent_headers(),
            timeout=15,
        )
        if resp.status_code != 200:
            # Fallback
            _update_job_progress_fallback(job_id, status, step, progress, error)
    except Exception:
        _update_job_progress_fallback(job_id, status, step, progress, error)


def _update_job_progress_fallback(
    job_id: str,
    status: str,
    step: str,
    progress: Optional[int],
    error: Optional[str] = None,
) -> bool:
    """Fallback: POST to a generic update endpoint or accept that we can't update."""
    # If no agent endpoints exist yet, log locally and continue
    log.debug(f"Job {job_id[:12]}... => {status} / {step} / {progress}%")
    return True


def complete_job(job_id: str, result: dict) -> None:
    """Mark a job as complete and upload result summary."""
    try:
        requests.post(
            f"{CHARLIE_API}/api/agent/complete",
            json={"jobId": job_id, "result": result},
            headers=_agent_headers(),
            timeout=30,
        )
    except Exception:
        pass
    # Always update status as well
    update_job_progress(job_id, "complete", "Complete", 100)


def upload_note_to_charlie(
    ticker: str,
    note_md: str,
    sources_md: str,
    changelog_md: str,
    docx_path: Optional[Path],
    chart_paths: list[Path],
    version: str,
    metadata: dict,
) -> bool:
    """Upload the completed note to Charlie's backend."""
    docx_b64 = ""
    if docx_path and docx_path.exists():
        docx_b64 = base64.b64encode(docx_path.read_bytes()).decode("ascii")

    charts_payload: list[dict] = []
    for cp in chart_paths:
        if cp and cp.exists():
            chart_type = "revenue" if "Revenue" in cp.name else "profit"
            charts_payload.append(
                {
                    "type": chart_type,
                    "filename": cp.name,
                    "data": base64.b64encode(cp.read_bytes()).decode("ascii"),
                }
            )

    note_id = str(uuid.uuid4())
    payload = {
        "noteId": note_id,
        "ticker": ticker,
        "version": version,
        "noteMarkdown": note_md,
        "sourcesMarkdown": sources_md,
        "changelogMarkdown": changelog_md,
        "noteDocx": docx_b64,
        "charts": [{"type": c["type"], "filename": c["filename"]} for c in charts_payload],
        "chartsData": charts_payload,
        "metadata": metadata,
    }

    try:
        resp = requests.post(
            f"{CHARLIE_API}/api/agent/save-note",
            json=payload,
            headers=_agent_headers(),
            timeout=120,
        )
        if resp.status_code == 200:
            log.info(f"Note uploaded to Charlie: {ticker} v{version}")
            return True
        else:
            log.warning(f"Upload returned {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        log.error(f"Upload failed: {e}")
        return False


def _agent_id() -> str:
    """Stable agent ID for this machine."""
    import platform

    return f"local-{platform.node()}"


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------


def read_ticker_files(ticker: str) -> list[dict]:
    """Read all source files from the ticker's iCloud folder.

    Skips subdirectories (Processed, Prior Versions, etc.), hidden files,
    and Office lock files (~$...).
    """
    ticker_dir = STOCKS_DIR / ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"No iCloud folder found for {ticker} at {ticker_dir}")

    MIME_MAP = {
        ".pdf": "application/pdf",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    ALLOWED_EXTS = set(MIME_MAP.keys())

    files: list[dict] = []
    for f in sorted(ticker_dir.iterdir()):
        if f.is_dir():
            continue
        if f.name.startswith(".") or f.name.startswith("~$"):
            continue

        ext = f.suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue

        try:
            raw = f.read_bytes()
            files.append(
                {
                    "filename": f.name,
                    "path": str(f),
                    "extension": ext,
                    "size": len(raw),
                    "data": base64.b64encode(raw).decode("ascii"),
                    "mime_type": MIME_MAP.get(ext, "application/octet-stream"),
                }
            )
            log.debug(f"  Read {f.name} ({len(raw):,} bytes)")
        except Exception as e:
            log.warning(f"  Could not read {f.name}: {e}")

    return files


# ---------------------------------------------------------------------------
# Excel text extraction
# ---------------------------------------------------------------------------


def extract_excel_text(filepath: str) -> Optional[str]:
    """Extract text content from Excel files. Uses zipfile+XML as fallback."""
    try:
        import openpyxl

        wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
        text_parts: list[str] = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            text_parts.append(f"\n=== Sheet: {sheet_name} ===")
            row_count = 0
            for row in ws.iter_rows(values_only=True):
                row_text = "\t".join(str(c) if c is not None else "" for c in row)
                if row_text.strip():
                    text_parts.append(row_text)
                    row_count += 1
                if row_count > 2000:
                    text_parts.append("[... truncated at 2000 rows ...]")
                    break
        wb.close()
        return "\n".join(text_parts)
    except Exception as e:
        log.debug(f"openpyxl failed for {filepath}: {e}, trying zipfile fallback")
        try:
            return _extract_xlsx_via_zip(filepath)
        except Exception as e2:
            log.warning(f"Could not extract Excel text from {filepath}: {e2}")
            return None


def _extract_xlsx_via_zip(filepath: str) -> Optional[str]:
    """Parse xlsx as zip + XML when openpyxl fails."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

    with zipfile.ZipFile(filepath) as z:
        # Read shared strings
        shared: dict[int, str] = {}
        try:
            tree = ElementTree.parse(z.open("xl/sharedStrings.xml"))
            for i, si in enumerate(tree.findall(f".//{ns}si")):
                texts = si.findall(f".//{ns}t")
                shared[i] = "".join(t.text or "" for t in texts)
        except (KeyError, ElementTree.ParseError):
            pass

        text_parts: list[str] = []
        for name in sorted(z.namelist()):
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
                sheet_tree = ElementTree.parse(z.open(name))
                rows = sheet_tree.findall(f".//{ns}row")
                for row in rows:
                    cells: list[str] = []
                    for cell in row.findall(f"{ns}c"):
                        val = cell.find(f"{ns}v")
                        cell_type = cell.get("t", "")
                        if val is not None and val.text:
                            if cell_type == "s":
                                cells.append(shared.get(int(val.text), val.text))
                            else:
                                cells.append(val.text)
                        else:
                            cells.append("")
                    row_text = "\t".join(cells)
                    if row_text.strip():
                        text_parts.append(row_text)

        return "\n".join(text_parts) if text_parts else None


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------


def call_claude(
    files: list[dict],
    ticker: str,
    company: str,
    sector: str,
    existing_note: Optional[str] = None,
    mode: str = "new",
    api_key: Optional[str] = None,
) -> str:
    """Call Claude API with all source files to generate the research note."""
    client = anthropic.Anthropic(api_key=api_key)

    content: list[dict] = []

    # Attach documents
    for f in files:
        if f["extension"] == ".pdf":
            content.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": f["data"],
                    },
                    "cache_control": {"type": "ephemeral"},
                }
            )
            content.append({"type": "text", "text": f"[Document: {f['filename']}]"})

        elif f["extension"] in (".xlsx", ".xls"):
            text = extract_excel_text(f["path"])
            if text:
                # Truncate very large spreadsheets
                if len(text) > 80_000:
                    text = text[:80_000] + "\n[... truncated ...]"
                content.append(
                    {"type": "text", "text": f"[Excel Model: {f['filename']}]\n{text}"}
                )

        elif f["extension"] in (".csv", ".txt", ".md"):
            try:
                text = base64.b64decode(f["data"]).decode("utf-8", errors="replace")
                if len(text) > 80_000:
                    text = text[:80_000] + "\n[... truncated ...]"
                content.append({"type": "text", "text": f"[{f['filename']}]\n{text}"})
            except Exception:
                pass

        elif f["extension"] == ".docx":
            # Try to extract text from docx
            docx_text = _extract_docx_text(f["path"])
            if docx_text:
                if len(docx_text) > 80_000:
                    docx_text = docx_text[:80_000] + "\n[... truncated ...]"
                content.append(
                    {"type": "text", "text": f"[Word Doc: {f['filename']}]\n{docx_text}"}
                )

    # Build the existing-thesis context if we have analysis data from Charlie
    existing_thesis_context = ""
    try:
        resp = requests.get(
            f"{CHARLIE_API}/api/analysis/{ticker}",
            headers=_agent_headers(),
            timeout=15,
        )
        if resp.status_code == 200:
            analysis = resp.json()
            thesis = analysis.get("thesis", {})
            signposts = analysis.get("signposts", [])
            threats = analysis.get("threats", [])
            conclusion = analysis.get("conclusion", "")
            if thesis or signposts or threats:
                existing_thesis_context = f"""
EXISTING STRUCTURED THESIS (use as foundation, expand with document details):
Summary: {thesis.get('summary', '')}
Pillars: {json.dumps(thesis.get('pillars', []), indent=2)}
Signposts: {json.dumps(signposts, indent=2)}
Threats: {json.dumps(threats, indent=2)}
Conclusion: {conclusion}
"""
    except Exception:
        pass

    update_context = ""
    if existing_note and mode == "update":
        update_context = f"""
EXISTING NOTE (update this, don't rewrite from scratch):
{existing_note[:8000]}

Update the note with new information from the source documents. Add a "What's Changed" section at the top. Update any data points, estimates, or catalysts that have changed.
"""

    prompt = f"""You are a senior equity research analyst writing a comprehensive investment research note.

TICKER: {ticker}
COMPANY: {company}
SECTOR: {sector}

{RESEARCH_NOTE_PLAYBOOK}

{existing_thesis_context}

{update_context}

Using the source documents provided, write a complete equity research note in markdown format.

The note should be 8-12 pages when printed, with these sections:
1. Executive Summary / Investment Thesis
2. Business Overview & Segment Breakdown
3. Key Revenue & Earnings Drivers
4. What the Street Is Debating
5. Catalyst Calendar
6. Valuation Context
7. Risks
8. Bottom Line

Also provide:
- Revenue segment data for donut chart (JSON array: [{{"segment": "name", "revenue": number_in_millions}}])
- Profit segment data for donut chart (JSON array: [{{"segment": "name", "profit": number_in_millions}}])

IMPORTANT:
- Source attributions go in a SEPARATE sources document, NOT in the main note
- In the sources doc, list section-by-section which reports informed each claim (broker name + date + page ref)
- In the main note, NEVER mention broker names, analyst names, or firm-specific targets
- Do NOT use Street opinion as thesis support (e.g. "the Street is constructive", "consensus targets suggest upside")
- OK in Valuation section: factual framing like "applying 11-12x to normalized EPS implies $X" (your own math)
- Nudge displayed numbers +/- $200-300M from model data to avoid exact replication

Return your response in this exact format:

===NOTE_START===
[full markdown note here]
===NOTE_END===

===SOURCES_START===
[sources document: section-by-section, which reports informed each claim, broker name + date + page ref]
===SOURCES_END===

===REVENUE_CHART_DATA===
[JSON array of revenue segments]
===REVENUE_CHART_END===

===PROFIT_CHART_DATA===
[JSON array of profit segments]
===PROFIT_CHART_END===
"""

    content.append({"type": "text", "text": prompt})

    log.info(f"Calling Claude with {len(content)} content blocks...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16384,
        system="You are a senior equity research analyst. Write thorough, data-driven research notes. Be precise with numbers. No sellside attribution in the main note.",
        messages=[{"role": "user", "content": content}],
    )

    return response.content[0].text


def _extract_docx_text(filepath: str) -> Optional[str]:
    """Extract plain text from a .docx file."""
    try:
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        with zipfile.ZipFile(filepath) as z:
            tree = ElementTree.parse(z.open("word/document.xml"))
            paragraphs = tree.findall(f".//{ns}p")
            texts: list[str] = []
            for p in paragraphs:
                runs = p.findall(f".//{ns}t")
                para_text = "".join(r.text or "" for r in runs)
                if para_text.strip():
                    texts.append(para_text)
            return "\n".join(texts) if texts else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_response(text: str) -> tuple[str, str, list[dict], list[dict]]:
    """Parse the LLM response into (note_md, sources_md, revenue_data, profit_data)."""
    note_md = ""
    sources_md = ""
    revenue_data: list[dict] = []
    profit_data: list[dict] = []

    m = re.search(r"===NOTE_START===\s*(.*?)\s*===NOTE_END===", text, re.DOTALL)
    note_md = m.group(1).strip() if m else text

    m = re.search(r"===SOURCES_START===\s*(.*?)\s*===SOURCES_END===", text, re.DOTALL)
    sources_md = m.group(1).strip() if m else ""

    m = re.search(
        r"===REVENUE_CHART_DATA===\s*(.*?)\s*===REVENUE_CHART_END===", text, re.DOTALL
    )
    if m:
        try:
            revenue_data = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            log.warning("Could not parse revenue chart data JSON")

    m = re.search(
        r"===PROFIT_CHART_DATA===\s*(.*?)\s*===PROFIT_CHART_END===", text, re.DOTALL
    )
    if m:
        try:
            profit_data = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            log.warning("Could not parse profit chart data JSON")

    return note_md, sources_md, revenue_data, profit_data


# ---------------------------------------------------------------------------
# Chart generation (donut charts matching backend format)
# ---------------------------------------------------------------------------


def generate_donut_chart(
    ticker: str,
    chart_type: str,
    data: list[dict],
    value_key: str,
    output_dir: Path,
) -> Optional[Path]:
    """Generate a donut chart PNG in the same style as the backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [d.get("segment", "") for d in data]
    values = [d.get(value_key, d.get("revenue", d.get("profit", 0))) for d in data]

    if not values or sum(values) == 0:
        return None

    total = sum(values)
    colors = [
        "#5DADE2",
        "#F7DC6F",
        "#F1948A",
        "#7DCEA0",
        "#BB8FCE",
        "#85C1E9",
        "#F8C471",
        "#82E0AA",
    ]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    wedges, _ = ax.pie(
        values,
        labels=None,
        colors=colors[: len(values)],
        startangle=90,
        wedgeprops={"width": 0.58, "edgecolor": "none", "linewidth": 0},
    )

    # Center circle and text
    centre_circle = plt.Circle((0, 0), 0.30, fc="white")
    ax.add_artist(centre_circle)
    total_str = f"\\${total/1000:.1f}B" if total >= 1000 else f"\\${total:.0f}M"
    ax.text(
        0, 0.05, "Total", ha="center", va="center", fontsize=12, color="#333333", fontweight="normal"
    )
    ax.text(
        0, -0.08, total_str, ha="center", va="center", fontsize=16, color="#333333", fontweight="bold"
    )

    # Inside labels (value + percentage on each wedge)
    for i, (wedge, value) in enumerate(zip(wedges, values)):
        pct = value / total * 100
        ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        x = 0.70 * np.cos(np.deg2rad(ang))
        y = 0.70 * np.sin(np.deg2rad(ang))
        val_str = f"\\${value/1000:.1f}B" if value >= 1000 else f"\\${value:.0f}M"
        fontsize = 11 if pct > 15 else 10 if pct > 8 else 9
        if pct >= 3:
            ax.text(
                x,
                y,
                f"{val_str}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )

    # Outside labels (segment names)
    for i, (wedge, label) in enumerate(zip(wedges, labels)):
        ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        x = 1.15 * np.cos(np.deg2rad(ang))
        y = 1.15 * np.sin(np.deg2rad(ang))
        ax.text(
            x, y, label, ha="center", va="center", fontsize=10, fontweight="bold", color="#333333"
        )

    ax.set_title(
        f"{ticker} -- {chart_type} Breakdown",
        fontsize=14,
        fontweight="bold",
        color="#333333",
        pad=20,
    )
    ax.set_aspect("equal")
    plt.tight_layout()

    filename = f"{ticker}_{chart_type.replace(' ', '_')}_Breakdown.png"
    output_path = output_dir / filename
    plt.savefig(output_path, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Chart saved: {filename}")
    return output_path


# ---------------------------------------------------------------------------
# DOCX generation
# ---------------------------------------------------------------------------


def generate_note_docx(
    ticker: str,
    company: str,
    markdown_text: str,
    chart_paths: list[Path],
    output_dir: Path,
) -> Optional[Path]:
    """Generate Word document from markdown with embedded charts.

    Uses Calibri 11pt, all black text (matching Tony's md_to_docx_v2.py style).
    """
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
    except ImportError:
        log.error("python-docx not installed -- skipping .docx generation")
        return None

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.font.color.rgb = RGBColor(0, 0, 0)

    # Title
    title = doc.add_heading(f"{ticker} -- {company}", level=1)
    for run in title.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)

    # Date line
    doc.add_paragraph(f"Equity Research Note -- {datetime.now().strftime('%B %Y')}")

    # Parse and render markdown
    lines = markdown_text.split("\n")
    charts_inserted = False
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("### "):
            h = doc.add_heading(line[4:].strip(), level=3)
            for r in h.runs:
                r.font.color.rgb = RGBColor(0, 0, 0)

        elif line.startswith("## "):
            section_title = line[3:].strip()
            h = doc.add_heading(section_title, level=2)
            for r in h.runs:
                r.font.color.rgb = RGBColor(0, 0, 0)
            # Insert charts after Business Overview section
            if not charts_inserted and (
                "business" in section_title.lower() or "segment" in section_title.lower()
            ):
                charts_inserted = True
                # We'll insert charts after this section's content ends
                # (at the next heading or after a blank line gap)

        elif line.startswith("# "):
            h = doc.add_heading(line[2:].strip(), level=1)
            for r in h.runs:
                r.font.color.rgb = RGBColor(0, 0, 0)

        elif line.startswith("- ") or line.startswith("* "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")

        elif re.match(r"^\d+\.\s", line):
            doc.add_paragraph(re.sub(r"^\d+\.\s", "", line).strip(), style="List Number")

        elif line.startswith("|") and "|" in line[1:]:
            # Simple markdown table handling
            # Collect all table lines
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[\s\-:|]+\|$", lines[i]):  # skip separator rows
                    table_lines.append(lines[i])
                i += 1
            i -= 1  # will be incremented below

            if table_lines:
                # Parse cells
                parsed_rows = []
                for tl in table_lines:
                    cells = [c.strip() for c in tl.strip("|").split("|")]
                    parsed_rows.append(cells)

                if parsed_rows:
                    num_cols = max(len(r) for r in parsed_rows)
                    table = doc.add_table(rows=len(parsed_rows), cols=num_cols)
                    table.style = "Table Grid"
                    for ri, row_cells in enumerate(parsed_rows):
                        for ci, cell_text in enumerate(row_cells):
                            if ci < num_cols:
                                cell = table.cell(ri, ci)
                                cell.text = cell_text
                                for p in cell.paragraphs:
                                    for r in p.runs:
                                        r.font.size = Pt(10)
                                        r.font.name = "Calibri"
                                        r.font.color.rgb = RGBColor(0, 0, 0)
                    # Bold header row
                    if len(parsed_rows) > 0:
                        for ci in range(num_cols):
                            for p in table.cell(0, ci).paragraphs:
                                for r in p.runs:
                                    r.bold = True

        elif line.strip():
            p = doc.add_paragraph()
            # Handle **bold** and *italic* inline formatting
            parts = re.split(r"(\*\*.*?\*\*|\*.*?\*)", line)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    run = p.add_run(part[2:-2])
                    run.bold = True
                elif part.startswith("*") and part.endswith("*"):
                    run = p.add_run(part[1:-1])
                    run.italic = True
                else:
                    p.add_run(part)

        i += 1

    # Embed charts at the end (or after Business Overview if we found it)
    if chart_paths:
        doc.add_heading("Charts", level=2)
        for cp in chart_paths:
            if cp and cp.exists():
                try:
                    doc.add_picture(str(cp), width=Inches(6))
                    doc.add_paragraph("")  # spacing
                except Exception as e:
                    log.warning(f"Could not embed chart {cp.name}: {e}")

    month = datetime.now().strftime("%b%Y")
    filename = f"{ticker}_Note_{month}.docx"
    output_path = output_dir / filename
    doc.save(str(output_path))
    log.info(f"  DOCX saved: {filename}")
    return output_path


# ---------------------------------------------------------------------------
# File organization (Prior Versions, Processed)
# ---------------------------------------------------------------------------


def organize_files(ticker_dir: Path, source_files: list[dict]) -> None:
    """Archive existing deliverables to Prior Versions/, move sources to Processed/.

    Follows Tony's memory preferences:
    - Prior versions get date suffix
    - Main folder should only contain latest deliverables
    - Source files go to Processed/
    """
    processed_dir = ticker_dir / "Processed"
    processed_dir.mkdir(exist_ok=True)

    prior_dir = ticker_dir / "Prior Versions"
    prior_dir.mkdir(exist_ok=True)

    date_suffix = datetime.now().strftime("%Y%m%d")

    # 1. Archive existing deliverables (notes, sources, changelogs, charts)
    for f in sorted(ticker_dir.iterdir()):
        if f.is_dir():
            continue
        name_lower = f.name.lower()
        ext = f.suffix.lower()

        is_deliverable = False
        # Note / Sources / Changelog markdown or docx
        if ext in (".md", ".docx") and any(
            kw in f.name for kw in ("_Note_", "_Sources_", "_Changelog")
        ):
            is_deliverable = True
        # Chart PNGs
        if ext == ".png" and any(
            kw in f.name for kw in ("Breakdown", "Revenue", "Profit")
        ):
            is_deliverable = True

        if is_deliverable:
            dest = prior_dir / f"{f.stem}_{date_suffix}{f.suffix}"
            # Avoid overwriting if archived today already
            if dest.exists():
                dest = prior_dir / f"{f.stem}_{date_suffix}_{uuid.uuid4().hex[:6]}{f.suffix}"
            shutil.move(str(f), str(dest))
            log.info(f"  Archived: {f.name} -> Prior Versions/")

    # 2. Move source files to Processed/
    for sf in source_files:
        src = Path(sf["path"])
        if src.exists() and src.parent == ticker_dir:
            dest = processed_dir / src.name
            if dest.exists():
                dest = processed_dir / f"{src.stem}_{date_suffix}{src.suffix}"
            shutil.move(str(src), str(dest))
            log.info(f"  Processed: {src.name}")


# ---------------------------------------------------------------------------
# Incremental update support
# ---------------------------------------------------------------------------


def find_existing_note(ticker_dir: Path, ticker: str) -> Optional[str]:
    """Find and read the most recent note markdown in the ticker folder.

    Used for incremental updates -- if a prior note exists, we pass it to
    Claude so it can update rather than rewrite from scratch.
    """
    candidates: list[Path] = []
    for f in ticker_dir.iterdir():
        if f.is_dir():
            continue
        if f.suffix.lower() == ".md" and "_Note_" in f.name:
            candidates.append(f)

    if not candidates:
        return None

    # Pick the most recently modified
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    try:
        return candidates[0].read_text(encoding="utf-8")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Version management
# ---------------------------------------------------------------------------


def determine_version(ticker_dir: Path, ticker: str, mode: str) -> str:
    """Determine the next version number based on existing notes."""
    if mode == "new":
        return "1.0"

    # Look for existing changelog to find latest version
    for f in ticker_dir.iterdir():
        if f.is_dir():
            continue
        if "_Changelog" in f.name and f.suffix.lower() == ".md":
            try:
                text = f.read_text(encoding="utf-8")
                versions = re.findall(r"## Version (\d+\.\d+)", text)
                if versions:
                    latest = versions[0]
                    parts = latest.split(".")
                    parts[-1] = str(int(parts[-1]) + 1)
                    return ".".join(parts)
            except Exception:
                pass

    return "1.0"


# ---------------------------------------------------------------------------
# Job orchestrator
# ---------------------------------------------------------------------------


def process_note_job(job: dict, api_key: str) -> None:
    """Process a single note generation job end-to-end."""
    job_id = job.get("id", str(uuid.uuid4()))
    ticker = job.get("ticker", "").upper()
    mode = "new"
    steps_detail = job.get("steps_detail", {})
    if isinstance(steps_detail, str):
        try:
            steps_detail = json.loads(steps_detail)
        except Exception:
            steps_detail = {}
    mode = steps_detail.get("mode", "new")

    log.info(f"=== Processing {ticker} (job {job_id[:12]}..., mode={mode}) ===")

    try:
        # Step 1: Claim and read files
        update_job_progress(job_id, "running", "Reading local files", 5)

        ticker_dir = STOCKS_DIR / ticker
        if not ticker_dir.exists():
            raise FileNotFoundError(f"No iCloud folder for {ticker} at {ticker_dir}")

        files = read_ticker_files(ticker)
        if not files:
            raise RuntimeError(f"No source files found in {ticker_dir}")

        log.info(f"  Found {len(files)} source files")
        update_job_progress(job_id, "running", f"Found {len(files)} files, calling Claude", 15)

        # Get company name from Charlie
        company = ticker
        try:
            resp = requests.get(
                f"{CHARLIE_API}/api/analysis/{ticker}",
                headers=_agent_headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                company = resp.json().get("company", ticker) or ticker
        except Exception:
            pass

        sector = get_sector(ticker)

        # Check for existing note (for incremental updates)
        existing_note = None
        if mode == "update":
            existing_note = find_existing_note(ticker_dir, ticker)
            if existing_note:
                log.info("  Found existing note for incremental update")

        # Step 2: Call Claude
        update_job_progress(job_id, "running", "Generating research note via Claude", 20)
        response_text = call_claude(
            files=files,
            ticker=ticker,
            company=company,
            sector=sector,
            existing_note=existing_note,
            mode=mode,
            api_key=api_key,
        )

        # Step 3: Parse response
        update_job_progress(job_id, "running", "Parsing response", 60)
        note_md, sources_md, revenue_data, profit_data = parse_response(response_text)

        if not note_md or len(note_md) < 500:
            log.warning(f"  Note seems too short ({len(note_md)} chars), using raw response")
            if len(response_text) > len(note_md):
                note_md = response_text

        log.info(f"  Note: {len(note_md):,} chars, Sources: {len(sources_md):,} chars")
        log.info(
            f"  Revenue segments: {len(revenue_data)}, Profit segments: {len(profit_data)}"
        )

        # Step 4: Generate charts
        update_job_progress(job_id, "running", "Generating charts", 70)
        chart_paths: list[Path] = []

        if revenue_data:
            cp = generate_donut_chart(ticker, "Revenue", revenue_data, "revenue", ticker_dir)
            if cp:
                chart_paths.append(cp)

        if profit_data:
            cp = generate_donut_chart(ticker, "Profit", profit_data, "profit", ticker_dir)
            if cp:
                chart_paths.append(cp)

        # Step 5: Generate DOCX
        update_job_progress(job_id, "running", "Building .docx", 80)
        docx_path = generate_note_docx(ticker, company, note_md, chart_paths, ticker_dir)

        # Step 6: Save markdown files locally
        month = datetime.now().strftime("%b%Y")
        version = determine_version(ticker_dir, ticker, mode)

        # Archive prior versions BEFORE saving new ones
        update_job_progress(job_id, "running", "Organizing files", 85)
        organize_files(ticker_dir, files)

        # Write new deliverables
        note_path = ticker_dir / f"{ticker}_Note_{month}.md"
        note_path.write_text(note_md, encoding="utf-8")
        log.info(f"  Saved: {note_path.name}")

        sources_path = ticker_dir / f"{ticker}_Sources_{month}.md"
        sources_path.write_text(sources_md, encoding="utf-8")
        log.info(f"  Saved: {sources_path.name}")

        # Changelog
        now_str = datetime.now().strftime("%Y-%m-%d")
        changelog_md = (
            f"# {ticker} Changelog\n\n"
            f"## Version {version} -- {now_str}\n"
            f"- {'Updated' if mode == 'update' else 'Initial'} research note generated\n"
            f"- {len(files)} source documents processed\n"
            f"- Generated via Charlie Local Agent\n"
        )

        # Append old changelog if this is an update
        if mode == "update":
            prior_dir = ticker_dir / "Prior Versions"
            for f in sorted(prior_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if "_Changelog" in f.name and f.suffix.lower() == ".md":
                    try:
                        old_text = f.read_text(encoding="utf-8")
                        # Strip the top-level heading and append
                        old_body = old_text.split("\n", 1)[1] if "\n" in old_text else old_text
                        changelog_md += "\n" + old_body
                    except Exception:
                        pass
                    break

        changelog_path = ticker_dir / f"{ticker}_Changelog.md"
        changelog_path.write_text(changelog_md, encoding="utf-8")
        log.info(f"  Saved: {changelog_path.name}")

        # Step 7: Upload to Charlie
        update_job_progress(job_id, "running", "Uploading to Charlie", 90)
        upload_note_to_charlie(
            ticker=ticker,
            note_md=note_md,
            sources_md=sources_md,
            changelog_md=changelog_md,
            docx_path=docx_path,
            chart_paths=chart_paths,
            version=version,
            metadata={
                "mode": mode,
                "documentsProcessed": len(files),
                "generatedLocally": True,
                "agentId": _agent_id(),
                "charCount": len(note_md),
            },
        )

        # Done
        update_job_progress(job_id, "complete", "Complete", 100)
        complete_job(job_id, {"ticker": ticker, "version": version, "charCount": len(note_md)})
        log.info(f"=== {ticker} complete (v{version}) ===\n")

    except Exception as e:
        log.error(f"Job {job_id[:12]}... failed: {e}")
        import traceback

        traceback.print_exc()
        update_job_progress(job_id, "failed", "Failed", None, error=str(e))


def process_ticker_directly(ticker: str, api_key: str, mode: str = "new") -> None:
    """Process a ticker directly without a backend job (--ticker flag)."""
    fake_job = {
        "id": str(uuid.uuid4()),
        "ticker": ticker.upper(),
        "job_type": "note",
        "steps_detail": json.dumps({"mode": mode}),
    }
    log.info(f"Direct processing mode for {ticker.upper()}")
    process_note_job(fake_job, api_key)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Charlie Local Agent -- polls for note jobs and processes them locally"
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one job and exit",
    )
    parser.add_argument(
        "--ticker",
        help="Process a specific ticker directly (no backend job needed)",
    )
    parser.add_argument(
        "--mode",
        choices=["new", "update"],
        default="new",
        help="Note generation mode (default: new)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY") or load_api_key_from_config()
    if not api_key:
        print("Error: Anthropic API key required.")
        print("  Set ANTHROPIC_API_KEY env var, use --api-key, or save to ~/.charlie_agent_config.json")
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate iCloud directory
    if not STOCKS_DIR.exists():
        log.error(f"iCloud STOCKS directory not found: {STOCKS_DIR}")
        sys.exit(1)

    log.info("Charlie Local Agent starting")
    log.info(f"  Stocks dir: {STOCKS_DIR}")
    log.info(f"  Backend: {CHARLIE_API}")

    # Direct ticker mode
    if args.ticker:
        process_ticker_directly(args.ticker, api_key, args.mode)
        return

    # Polling mode
    shutdown = threading.Event()

    def handle_signal(signum, frame):
        log.info(f"Received signal {signum}, shutting down...")
        shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log.info(f"Polling for jobs every {POLL_INTERVAL}s (Ctrl+C to stop)")

    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 10

    while not shutdown.is_set():
        try:
            jobs = poll_for_jobs()
            consecutive_errors = 0  # reset on successful poll

            if jobs:
                job = jobs[0]
                ticker = job.get("ticker", "???")
                log.info(f"Found job: {job['id'][:12]}... for {ticker}")

                # Claim the job
                claimed = claim_job(job["id"])
                if not claimed:
                    log.warning(f"Could not claim job {job['id'][:12]}..., skipping")
                    shutdown.wait(timeout=POLL_INTERVAL)
                    continue

                process_note_job(job, api_key)

                if args.once:
                    break
            else:
                log.debug("No pending jobs")

        except KeyboardInterrupt:
            break
        except Exception as e:
            consecutive_errors += 1
            log.error(f"Unexpected error: {e}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.error(f"Too many consecutive errors ({MAX_CONSECUTIVE_ERRORS}), exiting")
                sys.exit(1)

        shutdown.wait(timeout=POLL_INTERVAL)

    log.info("Agent stopped.")


if __name__ == "__main__":
    main()
