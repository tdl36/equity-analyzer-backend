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
        pairs = [(str(d.get("segment", "")).strip().lower(),
                  float(d.get("revenue", d.get("profit", 0)) or 0))
                 for d in (series or [])]
        pairs = [(k, v) for k, v in pairs if v > 0]
        total = sum(v for _, v in pairs)
        return {k: v / total for k, v in pairs} if total else {}

    sa, sb = shares(a), shares(b)
    if not sa or not sb or set(sa) != set(sb):
        return False
    return all(abs(sa[k] - sb[k]) <= tolerance for k in sa)


def render_donut(ticker, chart_type, data, value_key):
    """A donut of one segment series. None if there is nothing to draw."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy as np

    # A segment running an operating loss cannot be a slice of a pie; drawing
    # it would misstate every other share.
    pairs = [(d.get("segment", ""),
              float(d.get(value_key, d.get("revenue", d.get("profit", 0))) or 0))
             for d in (data or [])]
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
    ax.text(0, 0.05, "Total", ha="center", va="center", fontsize=12, color="#333333")
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

    ax.set_title(f"{ticker} — {chart_type} Breakdown", fontsize=14,
                 fontweight="bold", color="#333333", pad=20)
    ax.set_aspect("equal")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def render_pair(ticker, revenue_data, profit_data):
    """Both charts, skipping a profit chart that only repeats revenue.

    Returns [{'type', 'filename', 'png'}] — the profit entry is absent when the
    model could not distinguish the two, which is the honest outcome.
    """
    out = []
    rev = render_donut(ticker, "Revenue", revenue_data, "revenue")
    if rev:
        out.append({"type": "revenue", "filename": f"{ticker}_Revenue_Breakdown.png",
                    "png": rev})
    if profit_data and not is_duplicate_series(revenue_data, profit_data):
        prof = render_donut(ticker, "Operating Profit", profit_data, "profit")
        if prof:
            out.append({"type": "profit", "filename": f"{ticker}_Profit_Breakdown.png",
                        "png": prof})
    return out
