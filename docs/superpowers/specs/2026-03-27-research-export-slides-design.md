# Research Export to Slides/Infographics — Design Spec

**Date:** 2026-03-27
**Status:** Approved
**Location:** Agents tab, Research sub-view

---

## Overview

Add an "Export to Slides/Infographics" button to the Agents tab research results. When clicked, a modal allows the user to select multiple output formats with configurable slide counts, then queues independent background jobs. Results appear inline on the Agents tab AND as projects in the Slides tab.

## Requirements

1. Export button appears when research completes (alongside "Save to Research Tab" and "Download .md")
2. Modal wizard with checkboxes for 7 formats
3. Slide count picker for deck formats (Compact 1-3, Standard 4-6, Detailed 7-10, Comprehensive 11+, Custom)
4. Queue multiple exports simultaneously as independent background jobs
5. Progress shown inline on Agents tab in an "Exports" section below research results
6. Each export auto-creates a Slides tab project
7. Fire-and-forget: user can navigate away, results persist

## Output Formats

### Slide Decks (multi-slide, Gemini image gen)

| ID | Name | Description |
|----|------|-------------|
| `professional` | Professional Presentation | Clean, corporate, data-forward, structured headers |
| `sketchnote` | Sketchnote | Hand-drawn style, visual metaphors, icons, warm colors |
| `executive_brief` | Executive Brief | Dense text, key metrics highlighted, minimal decoration, serif fonts |
| `visual_storytelling` | Visual Storytelling | Hero images, one key point per slide, cinematic, large type |

### Single-Page Infographics (one image)

| ID | Name | Description |
|----|------|-------------|
| `research_summary` | Research Summary | Vertical flow, key findings/stats/sources |
| `competitive_landscape` | Competitive Landscape Map | Grid/quadrant layout, comparison matrix |
| `timeline` | Timeline/Evolution | Chronological view of findings, milestones |

## Slide Count Options

| Preset | Range | Outline Prompt Guidance |
|--------|-------|------------------------|
| Compact | 1-3 | Distill to the most critical findings only |
| Standard | 4-6 | Cover key themes with supporting evidence |
| Detailed | 7-10 | Comprehensive coverage with data, quotes, analysis |
| Comprehensive | 11+ | Full deep-dive with appendix-level detail |
| Custom | user enters number | Exact slide count honored |

## Frontend Changes (index.html)

### New State Variables

```javascript
const [exportModalOpen, setExportModalOpen] = useState(false);
const [exportSelections, setExportSelections] = useState({});
const [researchExports, setResearchExports] = useState({});  // keyed by research run ID
```

### Export Button

Added to the research results action row (lines ~20236-20248), next to existing buttons:
```
[Save to Research Tab] [Download .md] [Export to Slides/Infographics]
```

### Export Modal

Opens on button click. Two sections:

**Slide Decks** — each row: checkbox, format name, slide count dropdown (only visible when checked).
**Infographics** — each row: checkbox, format name. No slide count needed.

Footer: "Generate All" button (disabled until >= 1 selection). Shows count: "Generate 3 exports".

### Exports Section

Appears below research results when exports exist for the current research run. Each export rendered as a card:
- Format icon + name + slide count
- Status: queued / generating (with progress %) / complete / error
- When complete: thumbnail preview, "View in Slides" link, "Download" button
- Polls `/api/research/<run_id>/exports` every 3 seconds while any job is running

### Export History Persistence

`researchExports` state is populated on page load and when viewing a previous research run from history. Exports persist across navigation.

## Backend Changes (app_v3.py)

### New Table: `research_exports`

```sql
CREATE TABLE IF NOT EXISTS research_exports (
    id VARCHAR(100) PRIMARY KEY,
    research_run_id VARCHAR(100) NOT NULL,
    slide_project_id INTEGER,
    format VARCHAR(50) NOT NULL,
    type VARCHAR(20) NOT NULL,  -- 'deck' or 'infographic'
    slide_count INTEGER,
    status VARCHAR(20) DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);
```

### New Endpoint: `POST /api/research/<run_id>/export`

**Request:**
```json
{
  "exports": [
    {"format": "professional", "type": "deck", "slideCount": 6},
    {"format": "sketchnote", "type": "deck", "slideCount": 3},
    {"format": "research_summary", "type": "infographic"}
  ]
}
```

**Response:**
```json
{
  "exports": [
    {"id": "uuid1", "format": "professional", "status": "queued"},
    {"id": "uuid2", "format": "sketchnote", "status": "queued"},
    {"id": "uuid3", "format": "research_summary", "status": "queued"}
  ]
}
```

**Logic:**
1. Validate research run exists and is complete
2. For each export in the request:
   a. Create `research_exports` row with status='queued'
   b. Spawn background thread: `_run_research_export(export_id, run_id, format, type, slide_count)`
3. Return all export IDs

### New Endpoint: `GET /api/research/<run_id>/exports`

Returns all exports for a research run with current status/progress.

### Background Worker: `_run_research_export()`

1. Read research run data (synthesis + agent content + sources)
2. Build format-specific outline prompt:
   - Include research content
   - Specify format style guidelines
   - Specify target slide count
   - For infographics: single-image prompt
3. Call LLM (tier="standard") to generate outline JSON:
   ```json
   {
     "title": "...",
     "slides": [
       {"title": "...", "content": "...", "speakerNotes": "...", "visualDirection": "..."}
     ]
   }
   ```
4. Create `slide_projects` entry with name: "{Research Query} -- {Format Name} ({N} slides)"
5. For each slide in outline:
   a. Build Gemini image prompt using `_build_slide_prompt()` with format-specific style
   b. Call `_generate_slide_image()` (existing function)
   c. Create `slide_items` entry
   d. Update progress: `(i+1) / total_slides * 100`
6. Update `research_exports` row: status='complete', link slide_project_id
7. On error: status='error', error_message set

### Format-Specific Style Prompts

Each format gets injected into the Gemini image generation prompt:

- **professional**: "Clean corporate design. White/navy/accent color palette. Data tables and charts where appropriate. Sans-serif headers. Structured grid layout."
- **sketchnote**: "Hand-drawn visual style. Sketch-like icons and doodles. Warm earthy colors. Playful handwritten typography. Visual metaphors connecting ideas."
- **executive_brief**: "Dense, information-rich layout. Serif typography. Muted professional colors. Key metrics in large callout boxes. Minimal decoration."
- **visual_storytelling**: "Cinematic full-bleed imagery. One key insight per slide. Large bold typography. High contrast. Dramatic visual hierarchy."
- **research_summary**: "Vertical infographic flow. Sections for key findings, statistics, competitive data, and sources. Brand-consistent colors. Icons for each section."
- **competitive_landscape**: "Comparison matrix or quadrant layout. Side-by-side competitor columns. Logos/icons. Strengths highlighted in green, weaknesses in red."
- **timeline**: "Horizontal or vertical timeline. Date markers. Milestone icons. Progressive color gradient. Key events with brief descriptions."

## Data Flow

```
User clicks "Export to Slides/Infographics"
  -> Modal opens with format/count selections
  -> User checks formats, sets slide counts, clicks "Generate All"
  -> POST /api/research/<run_id>/export (array of selections)
  -> Backend creates research_exports rows + spawns threads
  -> Frontend closes modal, shows Exports section with progress cards
  -> Frontend polls GET /api/research/<run_id>/exports every 3s
  -> Each thread: LLM outline -> Gemini image gen -> slide_projects/items created
  -> On complete: thumbnail appears, "View in Slides" link active
  -> User can navigate away; exports persist in DB
```

## Error Handling

- If Gemini image gen fails for a slide, retry once, then mark that slide as "text-only" (content without image)
- If entire export fails, set status='error' with message; user can retry from the Exports section
- If research run is not complete, export button is disabled

## Testing

- Verify modal opens/closes correctly
- Verify multiple format selection with different slide counts
- Verify background jobs run independently
- Verify progress polling updates UI
- Verify Slides tab project creation
- Verify exports persist when navigating away and back
- Verify error states (Gemini failure, research not complete)
