"""Segment donut charts, in one place.

There were two implementations: one in the backend and one in the local agent,
drawn in the same style but drifting independently -- only the agent's refused
to draw a profit chart that duplicated the revenue chart, and only the agent's
labelled wedges with dollar values. A reader comparing a note generated on the
server with one generated on the laptop would have seen different charts from
the same data.

Returns PNG bytes; callers decide whether that becomes a file or base64.
"""

import io

# The note's palette. Ordered so adjacent wedges stay distinguishable.
PALETTE = [
    "#5DADE2", "#F7DC6F", "#F1948A", "#7DCEA0",
    "#BB8FCE", "#85C1E9", "#F8C471", "#82E0AA",
]


def _money(value_musd):
    """$M in, a human figure out."""
    return (f"${value_musd / 1000:.1f}B" if value_musd >= 1000
            else f"${value_musd:.0f}M")


def is_duplicate_series(a, b, tolerance=0.02):
    """Do two segment series say the same thing?

    Segment profit that merely repeats segment revenue is not a second data
    point, it is the same chart twice -- and it silently asserts every segment
    earns at the company margin. The agent checked this; the backend did not.
    """
    def shares(series):
        def _num(d):
            for key in ("value", "revenue", "profit"):
                if d.get(key) is not None:
                    try:
                        return float(d[key])
                    except (TypeError, ValueError):
                        continue
            return 0.0

        # "value" first: the parser normalises to it, and reading only
        # revenue/profit scored every normalised row as zero, which made this
        # return False for series that were in fact identical.
        pairs = [(str(d.get("segment", "")).strip().lower(), _num(d))
                 for d in (series or []) if isinstance(d, dict)]
        pairs = [(k, v) for k, v in pairs if v > 0]
        total = sum(v for _, v in pairs)
        return {k: v / total for k, v in pairs} if total else {}

    sa, sb = shares(a), shares(b)
    if not sa or not sb or set(sa) != set(sb):
        return False
    return all(abs(sa[k] - sb[k]) <= tolerance for k in sa)


def render_donut(ticker, chart_type, data, value_key, period=None):
    """A donut of one segment series. None if there is nothing to draw.

    `period` names the fiscal year the figures cover ("FY2025A", "FY2026E").
    It is part of the title because a segment mix is meaningless without it:
    the first charts shipped from a quarter's numbers and said only "Revenue
    Breakdown", so nothing on the page revealed which period was being shown.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy as np

    # A segment running an operating loss cannot be a slice of a pie; drawing
    # it would misstate every other share.
    # "value" is the normalised key the parser emits; "revenue"/"profit" are what
    # the model writes directly and what legacy notes stored. Accept all three --
    # reading only the latter silently scored every normalised row as zero, which
    # renders as no chart at all rather than as an error.
    def _num(d):
        for key in (value_key, "value", "revenue", "profit"):
            if key in d and d[key] is not None:
                try:
                    return float(d[key])
                except (TypeError, ValueError):
                    continue
        return 0.0

    pairs = [(d.get("segment", ""), _num(d)) for d in (data or []) if isinstance(d, dict)]
    pairs = [(label, val) for label, val in pairs if val > 0]
    if not pairs:
        return None

    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    wedges, _ = ax.pie(
        values, labels=None, colors=PALETTE[:len(values)], startangle=90,
        wedgeprops={"width": 0.58, "edgecolor": "none", "linewidth": 0},
    )
    ax.add_artist(plt.Circle((0, 0), 0.30, fc="white"))
    # "Segments", not "Total". The donut sums the reported segments only:
    # Corporate/Other is excluded (a loss cannot be a slice), and segment
    # revenue is gross of intersegment eliminations. Labelling that sum "Total"
    # invited it to be read as the consolidated figure -- a CVS profit chart
    # centred on $16.3B sat beside a note stating enterprise adjusted operating
    # income of $14.4B, with nothing explaining the gap.
    ax.text(0, 0.05, "Segments", ha="center", va="center", fontsize=12, color="#333333")
    ax.text(0, -0.08, _money(total), ha="center", va="center",
            fontsize=16, color="#333333", fontweight="bold")

    for wedge, value in zip(wedges, values):
        ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        x, y = 0.70 * np.cos(np.deg2rad(ang)), 0.70 * np.sin(np.deg2rad(ang))
        ax.text(x, y, f"{_money(value)}\n({value / total * 100:.1f}%)",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#00000055")])

    for wedge, label in zip(wedges, labels):
        ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        x, y = 1.18 * np.cos(np.deg2rad(ang)), 1.18 * np.sin(np.deg2rad(ang))
        ax.text(x, y, label, ha=("left" if x >= 0 else "right"), va="center",
                fontsize=11, fontweight="bold", color="#333333")

    title = (f"{ticker} — {period} {chart_type} by Segment" if period
             else f"{ticker} — {chart_type} Breakdown")
    ax.set_title(title, fontsize=14,
                 fontweight="bold", color="#333333", pad=20)
    ax.set_aspect("equal")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def render_series(ticker, specs):
    """Render one donut per requested series.

    `specs` is a list of {'kind': 'revenue'|'profit', 'period': 'FY2025A',
    'data': [...]}. Returns [{'type', 'label', 'filename', 'png'}], skipping any
    series with nothing drawable rather than failing the set -- a note missing
    next year's estimates should still get this year's charts.

    `label` exists because consumers used to build a heading from the type
    ("Revenue" -> "Revenue Breakdown"), which cannot express a period.
    """
    kind_titles = {'revenue': 'Revenue', 'profit': 'Operating Profit'}
    out = []
    for spec in (specs or []):
        kind = (spec.get('kind') or 'revenue').lower()
        period = (spec.get('period') or '').strip()
        data = spec.get('data')
        png = render_donut(ticker, kind_titles.get(kind, kind.title()),
                           data, kind, period=period or None)
        if not png:
            continue
        slug = ''.join(ch for ch in period if ch.isalnum()) or 'Latest'
        out.append({
            'type': f'{kind}_{slug.lower()}' if period else kind,
            'kind': kind,
            'period': period,
            'label': (f'{period} {kind_titles.get(kind, kind.title())}'
                      if period else kind_titles.get(kind, kind.title())),
            'filename': f'{ticker}_{slug}_{kind_titles.get(kind, kind).replace(" ", "_")}_Breakdown.png',
            'png': png,
        })
    return out


def render_pair(ticker, revenue_data, profit_data):
    """Both charts. A note is expected to carry revenue AND profit.

    This used to drop the profit chart when its shares matched revenue, on the
    reasoning that repeating revenue asserts every segment earns the company
    margin. That reasoning is sound about the DATA and wrong about the REMEDY:
    it turned a data problem into a missing chart, and a chart that is not there
    cannot be judged at all. Two donuts that look alike are visible and
    arguable; an absent one just looks like the feature is broken.

    The duplicate check still runs -- it is how the caller knows to go back for
    real segment profit before rendering -- but it no longer decides whether the
    chart exists. Callers should treat is_duplicate_series() as a signal to
    re-extract, and warn if it survives that.

    Returns [{'type', 'filename', 'png'}].
    """
    out = []
    rev = render_donut(ticker, "Revenue", revenue_data, "revenue")
    if rev:
        out.append({"type": "revenue", "filename": f"{ticker}_Revenue_Breakdown.png",
                    "png": rev})
    prof = render_donut(ticker, "Operating Profit", profit_data, "profit")
    if prof:
        out.append({"type": "profit", "filename": f"{ticker}_Profit_Breakdown.png",
                    "png": prof})
    return out
