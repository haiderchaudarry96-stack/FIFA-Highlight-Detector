"""Look and feel of the Streamlit app: event colors, badge HTML and the one custom CSS block."""
from __future__ import annotations

import html

# One color per event type, shared by the Plotly markers, the card badges and the legend.
EVENT_COLORS: dict[str, str] = {
    "Goal": "#16A34A",
    "Big Chance / Save": "#F59E0B",
    "Card / Foul": "#EF4444",
    "Exciting Moment": "#8B5CF6",
}
DARK_TEXT = "#1F2933"
PRIMARY = "#16A34A"
ACCENT = "#F59E0B"


def event_color(event_type: str) -> str:
    """Badge/marker color for an event type (unknown types fall back to purple)."""
    return EVENT_COLORS.get(event_type, EVENT_COLORS["Exciting Moment"])


def badge_text_color(event_type: str) -> str:
    """Dark text on amber, white on green/red/purple."""
    return DARK_TEXT if event_color(event_type) == EVENT_COLORS["Big Chance / Save"] else "#FFFFFF"


def badge_html(event_type: str) -> str:
    """A rounded colored label, e.g. a green 'Goal' pill."""
    return (f'<span class="hl-badge" style="background:{event_color(event_type)};'
            f'color:{badge_text_color(event_type)}">{html.escape(event_type)}</span>')


def legend_html() -> str:
    """A row of all event badges, used as the color legend under the timeline heading."""
    return '<div class="hl-legend">' + "".join(badge_html(t) for t in EVENT_COLORS) + "</div>"


CSS = f"""
<style>
:root {{
  --hl-green: {PRIMARY};
  --hl-green-dark: #15803D;
  --hl-amber: {ACCENT};
  --hl-text: {DARK_TEXT};
  --hl-muted: #52606D;
  --hl-card: #FBFDFB;
  --hl-line: #E3EEE1;
  --hl-radius: 16px;
  --hl-shadow: 0 4px 18px rgba(22, 101, 52, 0.08), 0 1px 3px rgba(31, 41, 51, 0.06);
}}

.stApp {{
  background:
    radial-gradient(1200px 380px at 12% -8%, rgba(22, 163, 74, 0.10), transparent 70%),
    radial-gradient(900px 320px at 95% -10%, rgba(245, 158, 11, 0.10), transparent 70%),
    #FFFFFF;
}}
/* top padding clears Streamlit's fixed header bar so the title is never clipped */
.block-container {{ padding-top: 4.5rem; max-width: 1280px; }}
html, body, .stApp {{ font-size: 16.5px; }}

/* ---------- header ---------- */
.hl-header {{ display: flex; align-items: center; gap: 18px; margin-bottom: 0.4rem; }}
.hl-logo {{
  width: 62px; height: 62px; border-radius: 18px; flex: none;
  display: grid; place-items: center; font-size: 34px;
  background: linear-gradient(135deg, #DCFCE7, #FEF3C7);
  box-shadow: var(--hl-shadow);
}}
.hl-title {{
  font-size: 2.35rem; font-weight: 800; line-height: 1.1; margin: 0; letter-spacing: -0.02em;
  background: linear-gradient(90deg, var(--hl-green-dark), var(--hl-green) 55%, var(--hl-amber));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.hl-tagline {{ color: var(--hl-muted); font-size: 1.05rem; margin: 4px 0 0 0; }}
.hl-section {{
  font-size: 1.35rem; font-weight: 750; color: var(--hl-text);
  margin: 1.6rem 0 0.5rem 0; display: flex; align-items: center; gap: 10px;
}}
.hl-section::before {{
  content: ""; width: 6px; height: 1.3rem; border-radius: 4px;
  background: linear-gradient(180deg, var(--hl-green), var(--hl-amber));
}}
.hl-section-sub {{ font-size: 0.92rem; font-weight: 600; color: var(--hl-muted); margin-left: 6px; }}
.hl-note {{ color: var(--hl-muted); font-size: 0.92rem; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ border-right: 1px solid var(--hl-line); }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0.5rem; }}
.hl-side-title {{
  font-size: 0.78rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--hl-green-dark); margin: 0.7rem 0 0.15rem 0;
  padding-bottom: 4px; border-bottom: 2px solid #D7EBD3;
}}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
  background: linear-gradient(135deg, #22C55E, var(--hl-green) 60%, var(--hl-green-dark));
  border: none; font-weight: 750; font-size: 1.05rem; padding: 0.7rem 1rem;
  border-radius: 14px; box-shadow: 0 6px 16px rgba(22, 163, 74, 0.28);
  transition: transform .15s ease, box-shadow .15s ease;
}}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {{
  transform: translateY(-1px); box-shadow: 0 10px 22px rgba(22, 163, 74, 0.34);
}}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
  border-radius: 14px; border: 1.5px solid #BFDDB9; background: #FFFFFF; font-weight: 600;
}}

/* ---------- cards, metrics, hero ---------- */
[data-testid="stMetric"] {{
  background: linear-gradient(160deg, #FFFFFF, #F3F8F2);
  border: 1px solid var(--hl-line) !important; border-radius: var(--hl-radius);
  box-shadow: var(--hl-shadow); padding: 8px 16px !important;
}}
[data-testid="stMetricLabel"] {{ min-height: 0; }}
[data-testid="stMetricLabel"] p {{ font-size: 0.85rem; }}
[data-testid="stMetricValue"] {{ font-weight: 800; color: var(--hl-text); font-size: 1.55rem; line-height: 1.25; }}
[class*="st-key-card_"], .st-key-hero {{
  background: var(--hl-card); border: 1px solid var(--hl-line) !important;
  border-radius: var(--hl-radius) !important; box-shadow: var(--hl-shadow);
  padding: 18px !important;
  animation: hl-fade-up .45s ease both;
  transition: box-shadow .2s ease, transform .2s ease;
}}
[class*="st-key-card_"]:hover {{ box-shadow: 0 10px 28px rgba(22, 101, 52, 0.14); transform: translateY(-2px); }}
.st-key-hero {{ background: linear-gradient(160deg, #FFFFFF 40%, #F0FAF0); }}
[data-testid="stImage"] img, [data-testid="stVideo"], video {{ border-radius: 12px; }}

.hl-card-head {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }}
.hl-rank {{
  width: 38px; height: 38px; border-radius: 12px; display: grid; place-items: center;
  font-weight: 800; color: #FFFFFF; background: linear-gradient(135deg, #22C55E, var(--hl-green-dark));
}}
.hl-time {{ font-size: 1.35rem; font-weight: 800; color: var(--hl-text); font-variant-numeric: tabular-nums; }}
.hl-score {{
  margin-left: auto; font-weight: 700; color: var(--hl-green-dark);
  background: #ECF7EA; border-radius: 999px; padding: 3px 10px; font-size: 0.9rem;
}}
.hl-badge {{
  display: inline-block; border-radius: 999px; padding: 3px 11px;
  font-size: 0.82rem; font-weight: 700; letter-spacing: 0.01em; white-space: nowrap;
}}
.hl-legend {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 2px 0 6px 0; }}
/* fixed-height, line-clamped text blocks keep both cards in a row the same height */
.hl-reason, .hl-snippet {{
  display: -webkit-box; -webkit-box-orient: vertical; overflow: hidden;
}}
.hl-reason {{
  color: var(--hl-text); font-size: 0.97rem; font-weight: 600; margin: 4px 0;
  line-height: 1.45; -webkit-line-clamp: 2; line-clamp: 2; height: 2.9em; cursor: help;
}}
.hl-snippet {{
  color: var(--hl-muted); font-style: italic; font-size: 0.92rem; line-height: 1.45;
  -webkit-line-clamp: 3; line-clamp: 3; height: calc(4.35em + 16px); box-sizing: border-box;
  border-left: 3px solid var(--hl-amber); background: #FFFBEB;
  padding: 8px 12px; border-radius: 0 10px 10px 0; margin: 6px 0 10px 0;
}}
.hl-empty {{
  text-align: center; padding: 48px 24px; border-radius: var(--hl-radius);
  border: 2px dashed #CDE6C8; background: #F7FBF6; color: var(--hl-muted);
}}
.hl-empty b {{ color: var(--hl-text); font-size: 1.2rem; }}

@keyframes hl-fade-up {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}
</style>
"""
