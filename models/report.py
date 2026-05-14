"""
Scout report PDF generator — fpdf2-based.
Generates a 1-page scout report for a given player.
"""

import io
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from fpdf import FPDF

DB_PATH = Path(__file__).parent.parent / "scout.db"

LEAGUE_SHORT = {
    "ENG-Premier League": "EPL",
    "ESP-La Liga":        "LaLiga",
    "GER-Bundesliga":     "Bundesliga",
    "ITA-Serie A":        "Serie A",
    "FRA-Ligue 1":        "Ligue 1",
}

DARK_BG  = (14, 17, 23)
ACCENT   = (126, 184, 247)
GREEN    = (126, 247, 168)
GOLD     = (247, 201, 126)
WHITE    = (255, 255, 255)
GRAY     = (160, 160, 160)
DARK_ROW = (30, 30, 46)
MID_ROW  = (24, 24, 38)


def _percentile_of(val: float, values: pd.Series) -> float:
    from scipy.stats import percentileofscore
    return round(percentileofscore(values.dropna().values, float(val), kind="rank"), 1)


def _build_radar_png(labels: list, percentiles: list) -> bytes:
    """Render a polar radar chart and return PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals = percentiles + [percentiles[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e1e2e")

    ax.plot(angles_closed, vals, color="#7EB8F7", linewidth=1.8)
    ax.fill(angles_closed, vals, color="#7EB8F7", alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color="#cccccc", fontsize=7)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color="#666666", fontsize=6)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", pad=6)
    ax.spines["polar"].set_color("#444444")
    ax.grid(color="#444444", linewidth=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_scout_pdf(player_name: str) -> bytes:
    """Return PDF bytes for a one-page scout report."""
    conn = sqlite3.connect(DB_PATH)
    master = pd.read_sql("SELECT * FROM players_master WHERE player = ? LIMIT 1",
                         conn, params=[player_name])
    vs = pd.read_sql(
        "SELECT market_value_eur, predicted_value_eur, undervalue_score "
        "FROM value_scouting WHERE player = ? LIMIT 1",
        conn, params=[player_name],
    )
    try:
        roles_df = pd.read_sql(
            "SELECT role_label FROM player_roles WHERE player = ? LIMIT 1",
            conn, params=[player_name],
        )
        role = roles_df.iloc[0]["role_label"] if not roles_df.empty else "—"
    except Exception:
        role = "—"

    pos_col = master.iloc[0]["pos"] if not master.empty else ""
    pos_group = "MF"
    if "FW" in str(pos_col).upper():   pos_group = "FW"
    elif "DF" in str(pos_col).upper(): pos_group = "DF"
    elif "GK" in str(pos_col).upper(): pos_group = "GK"

    peers = pd.read_sql(
        "SELECT * FROM players_master WHERE pos LIKE ?",
        conn, params=[f"%{pos_group}%"],
    )
    conn.close()

    if master.empty:
        raise ValueError(f"Player not found: {player_name}")

    p   = master.iloc[0]
    v   = vs.iloc[0] if not vs.empty else pd.Series(dtype=float)

    def num(col, default=0.0):
        return float(pd.to_numeric(p.get(col, default), errors="coerce") or 0)

    stats = [
        ("Goals / 90",    num("per_90_minutes_gls"), "per_90_minutes_gls"),
        ("xG / 90",       num("xg_p90"),              "xg_p90"),
        ("Assists / 90",  num("per_90_minutes_ast"),  "per_90_minutes_ast"),
        ("xA / 90",       num("xa_p90"),               "xa_p90"),
        ("Shots / 90",    num("standard_sh_90"),       "standard_sh_90"),
        ("G+A / 90",      num("per_90_minutes_g_a"),   "per_90_minutes_g_a"),
        ("Tackles Won",   num("performance_tklw"),     "performance_tklw"),
        ("Interceptions", num("performance_int"),      "performance_int"),
    ]

    age_raw = str(p.get("age", "")).split("-")[0]
    try:
        age = int(float(age_raw))
    except (ValueError, TypeError):
        age = "—"
    league  = LEAGUE_SHORT.get(str(p.get("league", "")), str(p.get("league", "")))
    minutes = int(float(p.get("playing_time_min", 0) or 0))
    actual_m  = float(v.get("market_value_eur",  0) or 0) / 1e6
    predict_m = float(v.get("predicted_value_eur", 0) or 0) / 1e6
    uv_score  = float(v.get("undervalue_score", 0) or 0)

    # ── Build PDF ─────────────────────────────────────────────
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)
    W, H = pdf.w, pdf.h
    M = 14

    # Background
    pdf.set_fill_color(*DARK_BG)
    pdf.rect(0, 0, W, H, "F")

    # Header bar
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 0, W, 28, "F")

    pdf.set_text_color(*DARK_BG)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(M, 5)
    pdf.cell(W - M * 2, 11, player_name, align="L")

    pdf.set_font("Helvetica", "", 9)
    subtitle = (
        f"{p.get('team', '—')}  ·  {league}  ·  "
        f"{p.get('pos', '—')}  ·  Age {age}  ·  {minutes:,} min"
    )
    pdf.set_xy(M, 16)
    pdf.cell(W - M * 2 - 55, 8, subtitle, align="L")

    # Role badge (top-right)
    pdf.set_fill_color(*GOLD)
    pdf.set_text_color(*DARK_BG)
    pdf.set_font("Helvetica", "B", 8)
    badge_w = 52
    pdf.set_xy(W - M - badge_w, 9)
    pdf.cell(badge_w, 8, role, border=0, fill=True, align="C")

    # ── Performance section ───────────────────────────────────
    y = 35
    pdf.set_text_color(*ACCENT)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_xy(M, y)
    pdf.cell(80, 7, "PERFORMANCE METRICS  (percentile vs same position)", align="L")
    y += 9

    BAR_MAX = 85

    for i, (label, val, col_key) in enumerate(stats):
        if col_key in peers.columns:
            pct = _percentile_of(val, pd.to_numeric(peers[col_key], errors="coerce"))
        else:
            pct = 50.0

        row_h = 9
        pdf.set_fill_color(*(DARK_ROW if i % 2 == 0 else MID_ROW))
        pdf.rect(M, y, W - M * 2, row_h, "F")

        # Label
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_xy(M + 2, y + 1.8)
        pdf.cell(50, row_h - 3, label)

        # Value
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*ACCENT)
        pdf.set_xy(M + 54, y + 1.8)
        pdf.cell(18, row_h - 3, f"{val:.3f}" if val < 10 else f"{val:.0f}", align="R")

        # Bar background + fill
        bar_x = M + 75
        bar_y = y + 3
        bar_filled = BAR_MAX * pct / 100
        pdf.set_fill_color(40, 40, 60)
        pdf.rect(bar_x, bar_y, BAR_MAX, 3.5, "F")
        bar_col = GREEN if pct >= 75 else ACCENT if pct >= 40 else (247, 126, 126)
        pdf.set_fill_color(*bar_col)
        if bar_filled > 0.5:
            pdf.rect(bar_x, bar_y, bar_filled, 3.5, "F")

        # Percentile label
        pdf.set_text_color(*GRAY)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_xy(bar_x + BAR_MAX + 2, y + 2)
        pdf.cell(12, row_h - 4, f"{pct:.0f}th", align="L")

        y += row_h

    # ── Radar chart (matplotlib polar PNG) ───────────────────
    try:
        from scipy.stats import percentileofscore as _pof
        from models.value_scouting import POS_RADAR_STATS, RADAR_STATS, INVERTED_RADAR_STATS

        radar_def = POS_RADAR_STATS.get(pos_group, RADAR_STATS)
        radar_labels, radar_vals = [], []
        for lbl, col in radar_def.items():
            if col in peers.columns:
                pct = round(_pof(
                    pd.to_numeric(peers[col], errors="coerce").dropna().values,
                    float(pd.to_numeric(p.get(col, 0), errors="coerce") or 0),
                    kind="rank",
                ), 1)
                if col in INVERTED_RADAR_STATS:
                    pct = round(100.0 - pct, 1)
            else:
                pct = 50.0
            radar_labels.append(lbl)
            radar_vals.append(pct)

        if radar_labels:
            radar_png = _build_radar_png(radar_labels, radar_vals)
            y += 6
            pdf.set_text_color(*ACCENT)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_xy(M, y)
            pdf.cell(80, 7, "PERCENTILE RADAR", align="L")
            y += 8
            radar_w = 68
            radar_h = 68
            radar_x = W / 2 - radar_w / 2
            pdf.image(io.BytesIO(radar_png), x=radar_x, y=y, w=radar_w, h=radar_h)
            y += radar_h + 4
    except Exception:
        pass

    # ── Value assessment ──────────────────────────────────────
    if actual_m > 0 and y < (H - 45):
        y += 2
        pdf.set_text_color(*ACCENT)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_xy(M, y)
        pdf.cell(80, 7, "MARKET VALUE ASSESSMENT", align="L")
        y += 9

        for lbl, val_str, color in [
            ("Actual market value", f"EUR{actual_m:.1f}M",  WHITE),
            ("Model estimate",      f"EUR{predict_m:.1f}M", GREEN if predict_m > actual_m else (247, 126, 126)),
            ("Undervalue score",    f"{uv_score:+.1f}%",  GREEN if uv_score > 0 else (247, 126, 126)),
        ]:
            pdf.set_fill_color(*MID_ROW)
            pdf.rect(M, y, 120, 8, "F")
            pdf.set_text_color(*GRAY)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_xy(M + 2, y + 2)
            pdf.cell(68, 4, lbl)
            pdf.set_text_color(*color)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_xy(M + 72, y + 2)
            pdf.cell(46, 4, val_str, align="R")
            y += 9

    # ── Footer ────────────────────────────────────────────────
    pdf.set_text_color(*GRAY)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_xy(M, H - 10)
    pdf.cell(
        W - M * 2, 5,
        "Football Scout AI  ·  Big 5 leagues  ·  FBref + Understat data  ·  XGBoost position-specific value model",
        align="C",
    )

    return bytes(pdf.output())
