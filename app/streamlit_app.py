import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import sqlite3
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── bootstrap: parquet → scout.db (Streamlit Cloud / fresh clone) ──────────
def _bootstrap_db():
    DB = Path(__file__).parent.parent / "scout.db"
    DATA = Path(__file__).parent.parent / "data"
    # always_refresh: rebuilt locally before every push
    always_refresh = {"players_master", "value_scouting", "player_roles",
                      "market_values", "league_factors", "model_metrics", "backtest_results"}
    tables = {
        "players_master":   DATA / "players_master.parquet",
        "value_scouting":   DATA / "value_scouting.parquet",
        "statsbomb_events": DATA / "statsbomb_events.parquet",
        "players_raw":      DATA / "players_raw.parquet",
        "understat_xg":     DATA / "understat_xg.parquet",
        "player_roles":     DATA / "player_roles.parquet",
        "market_values":    DATA / "market_values.parquet",
        "league_factors":   DATA / "league_factors.parquet",
        "model_metrics":    DATA / "model_metrics.parquet",
        "backtest_results": DATA / "backtest_results.parquet",
    }

    if not DB.exists():
        missing = [n for n, p in tables.items() if not p.exists()]
        if missing:
            st.error(f"Missing data files: {missing}")
            st.stop()
        conn = sqlite3.connect(DB)
        for table, path in tables.items():
            pd.read_parquet(path).to_sql(table, conn, if_exists="replace", index=False)
        conn.close()
    else:
        conn = sqlite3.connect(DB)
        existing = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        for table, path in tables.items():
            if (table not in existing or table in always_refresh) and path.exists():
                pd.read_parquet(path).to_sql(table, conn, if_exists="replace", index=False)
        conn.close()

_bootstrap_db()

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass  # local: key comes from .env via python-dotenv in models/search.py

DB_PATH = Path(__file__).parent.parent / "scout.db"


@st.cache_data(ttl=3600)
def _form_badges() -> dict:
    """25-26 vs 24-25 G+A/90 comparison → form badge per player."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT player, season, per_90_minutes_g_a, playing_time_min "
            "FROM players_raw WHERE season IN ('2425','2526') "
            "AND CAST(playing_time_min AS REAL) >= 450",
            conn,
        )
        conn.close()
    except Exception:
        return {}
    if df.empty:
        return {}
    df["g_a"] = pd.to_numeric(df["per_90_minutes_g_a"], errors="coerce").fillna(0)
    pivot = df.pivot_table(index="player", columns="season", values="g_a", aggfunc="mean")
    badges = {}
    for player, row in pivot.iterrows():
        v24 = row.get("2425")
        v25 = row.get("2526")
        if pd.isna(v24) or pd.isna(v25) or v24 == 0:
            continue
        chg = (v25 - v24) / v24
        if chg >= 0.20:
            badges[player] = "🔥 Hot"
        elif chg >= 0.05:
            badges[player] = "📈 Rising"
        elif chg >= -0.10:
            badges[player] = "→ Stable"
        else:
            badges[player] = "📉 Declining"
    return badges

from models.search import parse_query, search_players, get_player_detail
from models.form import get_form_trend, get_season_trend
from models.value_scouting import (
    get_undervalued, get_similar_players,
    get_player_percentiles, get_player_role, RADAR_STATS, POS_RADAR_STATS,
    get_team_fit_players, get_all_teams, get_feature_importance,
    get_league_factors, get_model_metrics, get_age_curve_data,
)
from models.report import generate_scout_pdf


def _build_radar(pct_df: pd.DataFrame, height: int = 380) -> go.Figure:
    """Render a polar radar from get_player_percentiles() output.
    Each player's row may have different column sets (position-specific labels).
    """
    fig = go.Figure()
    for _, row in pct_df.iterrows():
        pg = row.get("pos_group", "MF")
        radar = POS_RADAR_STATS.get(pg, RADAR_STATS)
        labels = list(radar.keys())
        vals = [float(row.get(lbl, 0)) for lbl in labels]
        vals_closed = vals + [vals[0]]
        labels_closed = labels + [labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=labels_closed,
            fill="toself", name=row["player"], opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(
            visible=True, range=[0, 100],
            ticktext=["0", "25", "50", "75", "100"],
            tickvals=[0, 25, 50, 75, 100],
        )),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="#ffffff", height=height,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=30, b=60),
    )
    return fig

st.set_page_config(
    page_title="Football Scout AI",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
<style>
.title  { font-size: 2rem; font-weight: 800; margin-bottom: 0; }
.subtitle { color: #888; font-size: 0.95rem; margin-top: 0; }
.stat-box {
    background: #1e1e2e; border-radius: 10px;
    padding: 16px 20px; text-align: center;
}
.stat-val { font-size: 1.6rem; font-weight: 700; color: #7EB8F7; }
.stat-lbl { font-size: 0.75rem; color: #aaa; margin-top: 2px; }
.tag-under { background:#1a3a1a; color:#7EF7A8; padding:3px 10px;
             border-radius:20px; font-size:0.8rem; font-weight:600; }
.tag-over  { background:#3a1a1a; color:#F77E7E; padding:3px 10px;
             border-radius:20px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">⚽ Football Scout AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Big 5 leagues · Multi-season weighted data (22/23 – 25/26) · XGBoost value model + xG/xA · NL search</p>',
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Player Search", "💰 Value Scouting", "🎯 Team Fit"])


# ════════════════════════════════════════════════════════════
# TAB 1 — Player Search
# ════════════════════════════════════════════════════════════
with tab1:
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            label="Search",
            placeholder="e.g. young pressing midfielder in the Bundesliga / clinical striker under 25 in Ligue 1",
            label_visibility="collapsed",
            key="search_query",
        )
    with col_btn:
        search_clicked = st.button("Search", use_container_width=True, type="primary", key="search_btn")

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.filters = None
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []

    if search_clicked and query.strip():
        with st.spinner("Parsing query with AI..."):
            try:
                filters = parse_query(query)
                results = search_players(filters)
                st.session_state.results = results
                st.session_state.filters = filters
            except Exception as e:
                st.error(f"Search error: {e}")

    if st.session_state.results is not None:
        results: pd.DataFrame = st.session_state.results
        filters = st.session_state.filters

        with st.expander("Parsed filters", expanded=False):
            st.code(json.dumps(filters, ensure_ascii=False, indent=2), language="json")

        if results.empty:
            st.warning("No players found. Try relaxing the filters.")
            st.stop()

        st.markdown(f"**{len(results)} players** found")

        badges = _form_badges()
        display = results.copy()
        display["Form"] = display["player"].map(badges).fillna("")
        display["player"] = display.apply(
            lambda r: f"{r['player']} ⚠" if pd.to_numeric(r.get("minutes", 900), errors="coerce") < 700 else r["player"],
            axis=1,
        )
        display = display.rename(columns={
            "player": "Player", "team": "Club", "pos": "Position",
            "age": "Age", "league": "League", "minutes": "Minutes",
            "goals_p90": "Goals/90", "assists_p90": "Assists/90",
            "total_goals": "Goals", "total_assists": "Assists",
            "tackles_won": "Tackles", "interceptions": "Interceptions", "fouls": "Fouls",
        })
        st.dataframe(
            display, use_container_width=True, hide_index=True,
            column_config={
                "Goals/90": st.column_config.ProgressColumn("Goals/90", min_value=0, max_value=1.5, format="%.2f"),
                "Assists/90": st.column_config.ProgressColumn("Assists/90", min_value=0, max_value=1.0, format="%.2f"),
                "Form": st.column_config.TextColumn("Form", width="small"),
            },
        )

        dl_col, wl_col = st.columns([2, 3])
        with dl_col:
            st.download_button(
                label="⬇ Download results as CSV",
                data=display.to_csv(index=False).encode("utf-8"),
                file_name="scout_results.csv",
                mime="text/csv",
            )
        with wl_col:
            wl_add = st.multiselect(
                "★ Add to Watchlist",
                options=results["player"].tolist(),
                default=[],
                key="wl_add_select",
            )
            if wl_add:
                new_players = [p for p in wl_add if p not in st.session_state.watchlist]
                st.session_state.watchlist.extend(new_players)
                if new_players:
                    st.toast(f"Added {len(new_players)} player(s) to watchlist")

        # ── Percentile radar comparison ──────────────────────
        st.divider()
        st.subheader("Percentile Profile Comparison")
        st.caption("Percentile rank vs same position across Big 5 leagues")
        player_names = results["player"].tolist()
        default = player_names[:2] if len(player_names) >= 2 else player_names
        selected = st.multiselect("Select players to compare (max 3)", player_names, default=default, max_selections=3, key="compare_select")

        if len(selected) >= 2:
            pct_df = get_player_percentiles(selected)
            if not pct_df.empty:
                st.plotly_chart(_build_radar(pct_df, height=450), use_container_width=True)

            # Side-by-side stat comparison table
            COMPARE_STATS = [
                ("Goals/90",     "per_90_minutes_gls"),
                ("xG/90",        "xg_p90"),
                ("npxG/90",      "npxg_p90"),
                ("Assists/90",   "per_90_minutes_ast"),
                ("xA/90",        "xa_p90"),
                ("G+A/90",       "per_90_minutes_g_a"),
                ("Shots/90",     "standard_sh_90"),
                ("SoT/90",       "standard_sot_90"),
                ("Tackles Won",  "performance_tklw"),
                ("Interceptions","performance_int"),
                ("Fouls Drawn",  "performance_fld"),
            ]
            try:
                conn = sqlite3.connect(DB_PATH)
                sel_df = pd.read_sql(
                    f"SELECT * FROM players_master WHERE player IN ({','.join(['?']*len(selected))})",
                    conn, params=selected,
                )
                vs_df = pd.read_sql(
                    f"SELECT player, market_value_eur, predicted_value_eur, undervalue_score "
                    f"FROM value_scouting WHERE player IN ({','.join(['?']*len(selected))})",
                    conn, params=selected,
                )
                conn.close()

                sel_df = sel_df.drop_duplicates("player").set_index("player")
                vs_df  = vs_df.drop_duplicates("player").set_index("player")

                comp_rows = []
                for label, col in COMPARE_STATS:
                    row = {"Metric": label}
                    for p in selected:
                        if p in sel_df.index and col in sel_df.columns:
                            v = pd.to_numeric(sel_df.loc[p, col], errors="coerce")
                            row[p] = round(float(v), 3) if pd.notna(v) else "—"
                        else:
                            row[p] = "—"
                    comp_rows.append(row)

                # Value rows
                for label, col in [("Value (M€)", "market_value_eur"), ("Predicted (M€)", "predicted_value_eur"), ("Undervalue (%)", "undervalue_score")]:
                    row = {"Metric": label}
                    for p in selected:
                        if p in vs_df.index:
                            v = pd.to_numeric(vs_df.loc[p, col], errors="coerce")
                            if col in ("market_value_eur", "predicted_value_eur"):
                                row[p] = f"{v/1e6:.1f}" if pd.notna(v) and v > 0 else "—"
                            else:
                                row[p] = f"{v:+.1f}%" if pd.notna(v) else "—"
                        else:
                            row[p] = "—"
                    comp_rows.append(row)

                comp_table = pd.DataFrame(comp_rows).set_index("Metric")

                st.divider()
                st.subheader("Side-by-Side Comparison")
                # Highlight the best value in each row (numeric only)
                def highlight_best(row):
                    styles = [""] * len(row)
                    try:
                        nums = [float(v) for v in row if str(v).replace(".","").replace("-","").replace("+","").isnumeric() or (isinstance(v, float) and not pd.isna(v))]
                        if not nums: return styles
                        best = max(nums)
                        for i, v in enumerate(row):
                            try:
                                if float(v) == best:
                                    styles[i] = "background-color: #1a3a1a; color: #7EF7A8; font-weight: bold"
                            except (ValueError, TypeError):
                                pass
                    except Exception:
                        pass
                    return styles

                st.dataframe(
                    comp_table.style.apply(highlight_best, axis=1),
                    use_container_width=True,
                )
            except Exception as e:
                st.info(f"Comparison table unavailable: {e}")

        # ── Player detail ────────────────────────────────────
        st.divider()
        st.subheader("Player Detail")
        detail_name = st.selectbox("Select player", player_names, key="detail_select")
        if detail_name:
            row = results[results["player"] == detail_name].iloc[0]
            try:
                _conn = sqlite3.connect(DB_PATH)
                _xg = pd.read_sql(
                    "SELECT xg_p90, npxg_p90, xa_p90 FROM players_master WHERE player = ? LIMIT 1",
                    _conn, params=[detail_name]
                )
                _conn.close()
                _xg_row = _xg.iloc[0] if not _xg.empty else None
            except Exception:
                _xg_row = None

            role_info = get_player_role(detail_name)
            role_label = role_info.get("role_label", "")

            # Header: stat boxes + role badge
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            for col, (lbl, val) in zip(
                [c1, c2, c3, c4, c5, c6],
                [("Club", row["team"]), ("Position", row["pos"]), ("Age", str(row["age"])),
                 ("Minutes", f"{int(row['minutes']):,}"), ("Goals/90", f"{row['goals_p90']:.2f}"),
                 ("Assists/90", f"{row['assists_p90']:.2f}")],
            ):
                with col:
                    st.markdown(
                        f'<div class="stat-box"><div class="stat-val">{val}</div>'
                        f'<div class="stat-lbl">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            if role_label:
                form_badge = _form_badges().get(detail_name, "")
                badge_html = f'<span style="background:#2a2a1e;color:#F7C97E;padding:4px 12px;border-radius:20px;font-size:0.85rem;font-weight:600">{role_label}</span>'
                if form_badge:
                    badge_html += f' &nbsp;<span style="background:#1a2a1a;color:#7EF7A8;padding:4px 12px;border-radius:20px;font-size:0.85rem;font-weight:600">{form_badge}</span>'
                st.markdown(badge_html, unsafe_allow_html=True)

            st.write("")
            if _xg_row is not None and pd.notna(_xg_row.get("xg_p90", None)):
                xg_c1, xg_c2, xg_c3 = st.columns(3)
                xg_c1.metric("xG/90", f"{_xg_row['xg_p90']:.3f}")
                xg_c2.metric("npxG/90", f"{_xg_row['npxg_p90']:.3f}")
                xg_c3.metric("xA/90", f"{_xg_row['xa_p90']:.3f}")

            # Percentile radar (position-specific)
            pct_df = get_player_percentiles([detail_name])
            if not pct_df.empty:
                st.divider()
                pg_label = pct_df.iloc[0].get("pos_group", "")
                st.subheader(f"Percentile Profile — {pg_label} metrics vs Big 5 peers")
                st.plotly_chart(_build_radar(pct_df, height=400), use_container_width=True)

            # PDF export
            st.divider()
            if st.button("Download Scout Report (PDF)", key="pdf_btn"):
                try:
                    pdf_bytes = generate_scout_pdf(detail_name)
                    st.download_button(
                        label="Save PDF",
                        data=pdf_bytes,
                        file_name=f"{detail_name.replace(' ', '_')}_scout_report.pdf",
                        mime="application/pdf",
                        key="pdf_dl",
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

            detail = get_player_detail(detail_name)
            sb = detail.get("statsbomb")
            if sb and sb.get("matches_played", 0) > 0:
                st.markdown(f"\n**StatsBomb event data ({int(sb['matches_played'])} matches)**")
                s1, s2, s3, s4, s5 = st.columns(5)
                for col, (lbl, key) in zip(
                    [s1, s2, s3, s4, s5],
                    [("Shots","total_shots"),("Passes","total_passes"),
                     ("Pressures","total_pressures"),("Dribbles","total_dribbles"),("Tackles","total_tackles")],
                ):
                    with col:
                        st.metric(lbl, int(sb.get(key) or 0))

            # Form trend
            st.divider()
            with st.spinner("Analysing form..."):
                form = get_form_trend(detail_name)
                season_data = get_season_trend(detail_name)

            if form.get("has_data"):
                st.subheader("Match-by-Match Form (LSTM)")
                t1, t2, t3 = st.columns(3)
                t1.metric("Trend", form["trend"])
                t2.metric("Last 5 slope", form["slope"])
                t3.metric("Next match prediction", form["prediction"] or "N/A")

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=form["match_ids"], y=form["scores"],
                    mode="lines+markers", name="Form score",
                    line=dict(color="#7EB8F7", width=2), marker=dict(size=6),
                ))
                scores_arr = np.array(form["scores"])
                if len(scores_arr) >= 3:
                    ma = np.convolve(scores_arr, np.ones(3)/3, mode="valid")
                    fig2.add_trace(go.Scatter(
                        x=list(range(3, len(scores_arr)+1)), y=ma.tolist(),
                        mode="lines", name="3-match MA",
                        line=dict(color="#F7C97E", width=2, dash="dash"),
                    ))
                if form["prediction"] is not None:
                    fig2.add_trace(go.Scatter(
                        x=[len(form["match_ids"])+1], y=[form["prediction"]],
                        mode="markers", name="Next match (LSTM)",
                        marker=dict(color="#7EF7A8", size=12, symbol="star"),
                    ))
                fig2.update_layout(
                    xaxis_title="Match", yaxis_title="Form Score", height=350,
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#ffffff", legend=dict(orientation="h", y=-0.2),
                    margin=dict(t=20, b=60),
                )
                fig2.update_xaxes(gridcolor="#333")
                fig2.update_yaxes(gridcolor="#333")
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Form score = Shots×2 + Dribbles×1.5 + Tackles×1 + Interceptions×1 + Pressures×0.5 + Passes×0.3")

            if season_data.get("has_data"):
                st.subheader("Multi-Season Performance Trend (22/23 – 25/26)")
                s1, s2 = st.columns(2)
                with s1:
                    st.metric("Trend", season_data["trend"])
                with s2:
                    st.caption("Seasons: " + " → ".join(season_data["seasons"]))

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=season_data["seasons"], y=season_data["gls_p90"],
                    mode="lines+markers", name="Goals/90",
                    line=dict(color="#7EB8F7", width=2), marker=dict(size=8),
                ))
                fig3.add_trace(go.Scatter(
                    x=season_data["seasons"], y=season_data["ast_p90"],
                    mode="lines+markers", name="Assists/90",
                    line=dict(color="#F7C97E", width=2), marker=dict(size=8),
                ))
                fig3.add_trace(go.Scatter(
                    x=season_data["seasons"], y=season_data["g_a_p90"],
                    mode="lines+markers", name="G+A/90",
                    line=dict(color="#7EF7A8", width=2, dash="dot"), marker=dict(size=6),
                ))
                fig3.update_layout(
                    xaxis_title="Season", yaxis_title="per 90 min", height=320,
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#ffffff", legend=dict(orientation="h", y=-0.25),
                    margin=dict(t=20, b=70),
                )
                fig3.update_xaxes(gridcolor="#333")
                fig3.update_yaxes(gridcolor="#333")
                st.plotly_chart(fig3, use_container_width=True)

                min_data = dict(zip(season_data["seasons"], season_data["minutes"]))
                st.caption("Minutes: " + "  |  ".join(f"{s}: {m}" for s, m in min_data.items()))

            elif not form.get("has_data"):
                st.info("No trend data available for this player.")


        # ── Watchlist section ─────────────────────────────────
        if st.session_state.watchlist:
            st.divider()
            st.subheader("★ Watchlist")
            wl_players = st.session_state.watchlist

            wl_col1, wl_col2 = st.columns([4, 1])
            with wl_col1:
                st.caption(f"{len(wl_players)} player(s) saved")
            with wl_col2:
                if st.button("Clear watchlist", key="wl_clear"):
                    st.session_state.watchlist = []
                    st.rerun()

            remove_p = st.multiselect("Remove from watchlist", wl_players, default=[], key="wl_remove")
            if remove_p:
                st.session_state.watchlist = [p for p in wl_players if p not in remove_p]
                st.rerun()

            if len(st.session_state.watchlist) >= 2:
                wl_pct = get_player_percentiles(st.session_state.watchlist[:5])
                if not wl_pct.empty:
                    st.plotly_chart(_build_radar(wl_pct, height=430), use_container_width=True)

            # Side-by-side stat table for watchlist players
            if st.session_state.watchlist:
                WATCH_STATS = [
                    ("Goals/90",      "per_90_minutes_gls"),
                    ("xG/90",         "xg_p90"),
                    ("Assists/90",    "per_90_minutes_ast"),
                    ("xA/90",         "xa_p90"),
                    ("G+A/90",        "per_90_minutes_g_a"),
                    ("Shots/90",      "standard_sh_90"),
                    ("Tackles Won",   "performance_tklw"),
                    ("Interceptions", "performance_int"),
                ]
                try:
                    wl_names = st.session_state.watchlist[:6]
                    conn = sqlite3.connect(DB_PATH)
                    wl_master = pd.read_sql(
                        f"SELECT * FROM players_master WHERE player IN ({','.join(['?']*len(wl_names))})",
                        conn, params=wl_names,
                    )
                    wl_vs = pd.read_sql(
                        f"SELECT player, market_value_eur, predicted_value_eur, undervalue_score "
                        f"FROM value_scouting WHERE player IN ({','.join(['?']*len(wl_names))})",
                        conn, params=wl_names,
                    )
                    conn.close()
                    wl_master = wl_master.drop_duplicates("player").set_index("player")
                    wl_vs = wl_vs.drop_duplicates("player").set_index("player")

                    wl_rows = []
                    for lbl, col in WATCH_STATS:
                        row = {"Metric": lbl}
                        for p in wl_names:
                            if p in wl_master.index and col in wl_master.columns:
                                v = pd.to_numeric(wl_master.loc[p, col], errors="coerce")
                                row[p] = round(float(v), 3) if pd.notna(v) else "—"
                            else:
                                row[p] = "—"
                        wl_rows.append(row)

                    for lbl, col in [("Value (M€)", "market_value_eur"), ("Predicted (M€)", "predicted_value_eur"), ("Undervalue (%)", "undervalue_score")]:
                        row = {"Metric": lbl}
                        for p in wl_names:
                            if p in wl_vs.index:
                                v = pd.to_numeric(wl_vs.loc[p, col], errors="coerce")
                                if col in ("market_value_eur", "predicted_value_eur"):
                                    row[p] = f"{v/1e6:.1f}" if pd.notna(v) and v > 0 else "—"
                                else:
                                    row[p] = f"{v:+.1f}%" if pd.notna(v) else "—"
                            else:
                                row[p] = "—"
                        wl_rows.append(row)

                    wl_table = pd.DataFrame(wl_rows).set_index("Metric")
                    st.dataframe(wl_table, use_container_width=True)
                except Exception as e:
                    st.info(f"Watchlist comparison unavailable: {e}")


# ════════════════════════════════════════════════════════════
# TAB 2 — Value Scouting
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Undervalued Player Detection")
    st.caption("XGBoost model predicts performance-based fair value — players where predicted > actual market value are flagged as undervalued")

    # ── League difficulty context ─────────────────────────────
    with st.expander("League Scoring Difficulty", expanded=False):
        try:
            lf_df = get_league_factors()
            if not lf_df.empty:
                st.caption(
                    "Goals/90 scoring rate per league vs Big 5 average — "
                    "factor < 1 means this league scores more than average (stats slightly deflated in adj_* columns); "
                    "factor > 1 means harder to score here (stats boosted)"
                )
                lf_display = lf_df.copy()
                lf_display["League"] = lf_display["league"].str.replace(r"^.*?-", "", regex=True)
                lf_display["Goals/90 (league avg)"] = lf_display["league_mean"].round(3)
                lf_display["Goals/90 (global avg)"] = lf_display["global_mean"].round(3)
                lf_display["Adj Factor"] = lf_display["factor"].round(3)
                lf_display["Difficulty"] = lf_display["factor"].apply(
                    lambda f: "Harder to score" if f > 1.03 else ("Easier to score" if f < 0.97 else "Average")
                )
                st.dataframe(
                    lf_display[["League", "Goals/90 (league avg)", "Goals/90 (global avg)", "Adj Factor", "Difficulty"]],
                    use_container_width=True, hide_index=True,
                )
        except Exception:
            pass

    # ── Filters ──────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        budget = st.slider("Max budget (M€)", min_value=1, max_value=150, value=30, step=1)
    with f2:
        pos_filter = st.selectbox("Position", ["All", "FW", "MF", "DF", "GK"])
    with f3:
        league_filter = st.selectbox("League", [
            "All", "ENG-Premier League", "ESP-La Liga",
            "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1",
        ])
    with f4:
        age_filter = st.slider("Max age", min_value=16, max_value=38, value=28)

    top_n = st.slider("Results to show", min_value=5, max_value=50, value=20)

    if st.button("Run analysis", type="primary", key="value_btn"):
        with st.spinner("Scanning for undervalued players..."):
            try:
                df_val = get_undervalued(
                    max_value_eur=budget * 1_000_000,
                    position=None if pos_filter == "All" else pos_filter,
                    league=None if league_filter == "All" else league_filter,
                    age_max=age_filter,
                    top_n=top_n,
                )
                st.session_state.value_results = df_val
            except Exception as e:
                st.error(f"Error: {e}")

    if "value_results" not in st.session_state:
        st.session_state.value_results = None

    if st.session_state.value_results is not None:
        df_val: pd.DataFrame = st.session_state.value_results

        if df_val.empty:
            st.warning("No players found. Try adjusting the filters.")
        else:
            st.markdown(f"**{len(df_val)} players** identified")
            if pos_filter == "GK":
                st.info("⚠️ GK model has higher uncertainty (CV RMSE 0.75 vs 0.54–0.55 for outfield). Results are re-ranked by performance percentile composite (Save% + CS% + GA90 inverted + Saves/90).")
                # Re-rank GKs by percentile composite instead of undervalue_score
                try:
                    conn_gk = sqlite3.connect(DB_PATH)
                    gk_stats = pd.read_sql(
                        """SELECT vs.player, vs.gk_save_pct, vs.gk_ga_p90,
                                  vs.gk_cs_pct, vs.gk_saves_p90, vs.gk_pksave_pct
                           FROM value_scouting vs
                           WHERE vs.gk_save_pct IS NOT NULL AND vs.gk_save_pct > 0
                             AND vs.playing_time_min >= 1500""",
                        conn_gk,
                    )
                    conn_gk.close()
                    if not gk_stats.empty:
                        from scipy.stats import percentileofscore as _pof
                        for col in ["gk_save_pct", "gk_cs_pct", "gk_saves_p90", "gk_pksave_pct"]:
                            gk_stats[f"pct_{col}"] = gk_stats[col].apply(
                                lambda v: _pof(gk_stats[col].dropna().values, v, kind="rank")
                            )
                        # GA/90: lower is better → invert
                        gk_stats["pct_gk_ga_p90"] = gk_stats["gk_ga_p90"].apply(
                            lambda v: 100 - _pof(gk_stats["gk_ga_p90"].dropna().values, v, kind="rank")
                        )
                        gk_stats["gk_perf_score"] = (
                            gk_stats["pct_gk_save_pct"] * 0.35
                            + gk_stats["pct_gk_cs_pct"] * 0.25
                            + gk_stats["pct_gk_ga_p90"] * 0.25
                            + gk_stats["pct_gk_saves_p90"] * 0.15
                        ).round(1)
                        df_val = df_val.merge(
                            gk_stats[["player", "gk_perf_score"]], on="player", how="left"
                        )
                        df_val = df_val.sort_values("gk_perf_score", ascending=False)
                except Exception:
                    pass

            # ── Scatter: actual vs predicted value ───────────
            st.divider()
            st.subheader("Actual vs Predicted Market Value")
            st.caption("Above the line = undervalued (predicted > actual) | Below = overvalued")

            conn = sqlite3.connect(DB_PATH)
            scatter_df = pd.read_sql(
                """SELECT player, team, league, pos, age,
                          market_value_eur, predicted_value_eur, undervalue_score,
                          per_90_minutes_gls
                   FROM value_scouting
                   WHERE market_value_eur > 0 AND undervalue_score IS NOT NULL
                     AND market_value_eur <= ?""",
                conn, params=[budget * 1_000_000]
            )
            conn.close()

            scatter_df["Actual (M€)"]    = scatter_df["market_value_eur"] / 1e6
            scatter_df["Predicted (M€)"] = scatter_df["predicted_value_eur"] / 1e6
            scatter_df["Undervalue (%)"] = scatter_df["undervalue_score"].round(1)
            scatter_df["Goals/90"]       = pd.to_numeric(scatter_df["per_90_minutes_gls"], errors="coerce").fillna(0).round(2)

            fig_s = px.scatter(
                scatter_df,
                x="Actual (M€)", y="Predicted (M€)",
                color="league",
                hover_name="player",
                hover_data={"team": True, "pos": True, "age": True,
                            "Goals/90": True, "Undervalue (%)": True,
                            "Actual (M€)": ":.1f", "Predicted (M€)": ":.1f"},
                color_discrete_map={
                    "ENG-Premier League": "#7EB8F7",
                    "ESP-La Liga":        "#F7C97E",
                    "GER-Bundesliga":     "#7EF7A8",
                    "ITA-Serie A":        "#F77E7E",
                    "FRA-Ligue 1":        "#C47EF7",
                },
                height=500,
            )

            max_val = max(scatter_df["Actual (M€)"].max(), scatter_df["Predicted (M€)"].max())
            fig_s.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", name="Fair value line",
                line=dict(color="#555", dash="dash", width=1),
            ))
            fig_s.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font_color="#ffffff", legend_title="League",
                margin=dict(t=20, b=40),
            )
            fig_s.update_xaxes(gridcolor="#222", title="Actual market value (M€)")
            fig_s.update_yaxes(gridcolor="#222", title="Predicted fair value (M€)")
            st.plotly_chart(fig_s, use_container_width=True)

            # ── Undervalued table ─────────────────────────────
            st.divider()
            st.subheader(f"Undervalued TOP {len(df_val)}")

            st.dataframe(
                df_val,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Undervalue (%)": st.column_config.ProgressColumn(
                        "Undervalue (%)", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "Goals/90": st.column_config.NumberColumn("Goals/90", format="%.2f"),
                    "Assists/90": st.column_config.NumberColumn("Assists/90", format="%.2f"),
                    "Actual (M€)": st.column_config.NumberColumn("Actual (M€)", format="%.1f"),
                    "Predicted (M€)": st.column_config.NumberColumn("Predicted (M€)", format="%.1f"),
                    "xG/90": st.column_config.NumberColumn("xG/90", format="%.3f"),
                    "npxG/90": st.column_config.NumberColumn("npxG/90", format="%.3f"),
                    "xA/90": st.column_config.NumberColumn("xA/90", format="%.3f"),
                },
            )

            # ── Player detail card ────────────────────────────
            st.divider()
            st.subheader("Player Detail")
            val_players = df_val["player"].tolist()
            selected_val = st.selectbox("Select player", val_players, key="val_detail")

            if selected_val:
                row = df_val[df_val["player"] == selected_val].iloc[0]
                actual  = row["Actual (M€)"]
                predict = row["Predicted (M€)"]
                score   = row["Undervalue (%)"]
                tag     = "tag-under" if score > 0 else "tag-over"
                label   = f"Undervalued +{score:.1f}%" if score > 0 else f"Overvalued {score:.1f}%"

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Club", row["team"])
                m2.metric("Position", row["pos"])
                m3.metric("Age", str(row["age"]))
                m4.metric("Market value", f"EUR{actual:.1f}M")
                m5.metric("Model estimate", f"EUR{predict:.1f}M", delta=f"{score:+.1f}%")

                val_role = get_player_role(selected_val).get("role_label", "")
                badge_row = f'<span class="{tag}">{label}</span>'
                if val_role:
                    badge_row += f' &nbsp;<span style="background:#2a2a1e;color:#F7C97E;padding:3px 10px;border-radius:20px;font-size:0.8rem;font-weight:600">{val_role}</span>'
                st.markdown(badge_row, unsafe_allow_html=True)

                # Get CV RMSE for CI bands
                try:
                    _metrics = get_model_metrics()
                    _pos_grp_ci = "MF"
                    _pos_raw_ci = str(row.get("pos", ""))
                    if "FW" in _pos_raw_ci.upper():   _pos_grp_ci = "FW"
                    elif "DF" in _pos_raw_ci.upper(): _pos_grp_ci = "DF"
                    elif "GK" in _pos_raw_ci.upper(): _pos_grp_ci = "GK"
                    _rmse_row = _metrics[_metrics["pos_group"] == _pos_grp_ci]
                    _rmse_val = float(_rmse_row.iloc[0]["rmse"]) if not _rmse_row.empty else 0.0
                    # CI in value space: predict * (e^rmse - 1)
                    _ci_m = predict * (np.exp(_rmse_val) - 1) if _rmse_val > 0 else 0.0
                except Exception:
                    _ci_m, _rmse_val = 0.0, 0.0

                _ci_kwargs = {}
                if _ci_m > 0:
                    _ci_kwargs["error_y"] = dict(
                        type="data",
                        array=[float(_ci_m)],
                        arrayminus=[float(min(float(_ci_m), float(predict) * 0.9))],
                        color="#aaa", thickness=2, width=6,
                    )

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    name="Actual market value",
                    x=["Value comparison"],
                    y=[actual],
                    marker_color="#F77E7E",
                    width=0.3,
                ))
                fig_bar.add_trace(go.Bar(
                    name="Model estimate (±1σ CI)",
                    x=["Value comparison"],
                    y=[predict],
                    marker_color="#7EF7A8",
                    width=0.3,
                    **_ci_kwargs,
                ))
                fig_bar.update_layout(
                    barmode="group", height=300,
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#ffffff", yaxis_title="M€",
                    margin=dict(t=20, b=20),
                    legend=dict(orientation="h", y=-0.3),
                )
                fig_bar.update_yaxes(gridcolor="#222")
                if _rmse_val > 0:
                    st.caption(f"Error bars: ±1σ confidence interval (CV RMSE = {_rmse_val:.3f} log-scale, {_pos_grp_ci} model)")
                st.plotly_chart(fig_bar, use_container_width=True)

                # ── Feature importance for this position model ──
                try:
                    pos_grp = "MF"
                    pos_raw = str(row.get("pos", ""))
                    if "FW" in pos_raw.upper():   pos_grp = "FW"
                    elif "DF" in pos_raw.upper(): pos_grp = "DF"
                    elif "GK" in pos_raw.upper(): pos_grp = "GK"

                    fi_df = get_feature_importance(pos_grp)
                    if not fi_df.empty:
                        st.divider()
                        st.subheader(f"What Drives {pos_grp} Value? — Model Feature Importance")
                        st.caption(f"XGBoost gain importance for the {pos_grp} position model (trained on {pos_grp} players with TM market values)")

                        FEAT_LABELS = {
                            "age": "Age", "age_factor": "Age Factor (peak multiplier)",
                            "age_sq": "Age² (non-linear peak)", "league_tier": "League Tier",
                            "seasons_count": "Seasons in Big 5", "playing_time_min": "Minutes Played",
                            "playing_time_90s": "90s Played",
                            "per_90_minutes_gls": "Goals/90", "per_90_minutes_ast": "Assists/90",
                            "per_90_minutes_g_a": "G+A/90", "performance_gls": "Total Goals",
                            "performance_ast": "Total Assists",
                            "standard_sh_90": "Shots/90", "standard_sot_90": "SoT/90",
                            "standard_g_sh": "Goals per Shot",
                            "xg_p90": "xG/90", "npxg_p90": "npxG/90", "xa_p90": "xA/90",
                            "performance_tklw": "Tackles Won", "performance_int": "Interceptions",
                            "performance_fld": "Fouls Drawn", "performance_fls": "Fouls Committed",
                            "performance_crdy": "Yellow Cards",
                            "gk_save_pct": "Save %", "gk_ga_p90": "GA/90 (lower=better)",
                            "gk_cs_pct": "Clean Sheet %", "gk_saves_p90": "Saves/90",
                        }
                        fi_df["label"] = fi_df["feature"].map(FEAT_LABELS).fillna(fi_df["feature"])
                        fi_df = fi_df.sort_values("importance", ascending=True)

                        fig_fi = go.Figure(go.Bar(
                            x=fi_df["importance"],
                            y=fi_df["label"],
                            orientation="h",
                            marker_color=[
                                "#7EF7A8" if v >= fi_df["importance"].quantile(0.75)
                                else "#7EB8F7" if v >= fi_df["importance"].quantile(0.40)
                                else "#555"
                                for v in fi_df["importance"]
                            ],
                        ))
                        fig_fi.update_layout(
                            height=max(300, len(fi_df) * 24),
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            font_color="#ffffff", xaxis_title="Importance (gain)",
                            margin=dict(t=10, b=20, l=10, r=10),
                        )
                        fig_fi.update_xaxes(gridcolor="#222")
                        st.plotly_chart(fig_fi, use_container_width=True)
                except Exception:
                    pass

                # ── Similar player recommendations ────────────
                st.divider()
                st.subheader("Similar Players — Cheaper Alternatives")
                st.caption("Same statistical profile · cosine similarity · lower market value")
                with st.spinner("Finding similar players..."):
                    similar = get_similar_players(
                        selected_val,
                        max_value_eur=actual * 1_000_000 * 0.85,
                        top_n=8,
                    )
                if similar.empty:
                    st.info("No similar players found with sufficient data.")
                else:
                    st.dataframe(
                        similar,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Similarity (%)": st.column_config.ProgressColumn(
                                "Similarity (%)", min_value=0, max_value=100, format="%.1f%%"
                            ),
                            "Undervalue (%)": st.column_config.NumberColumn("Undervalue (%)", format="%.1f"),
                            "Goals/90": st.column_config.NumberColumn("Goals/90", format="%.2f"),
                            "Assists/90": st.column_config.NumberColumn("Assists/90", format="%.2f"),
                            "Actual (M€)": st.column_config.NumberColumn("Actual (M€)", format="%.1f"),
                            "Predicted (M€)": st.column_config.NumberColumn("Predicted (M€)", format="%.1f"),
                            "xG/90": st.column_config.NumberColumn("xG/90", format="%.3f"),
                            "npxG/90": st.column_config.NumberColumn("npxG/90", format="%.3f"),
                            "xA/90": st.column_config.NumberColumn("xA/90", format="%.3f"),
                        },
                    )

    # ── Model Validation ─────────────────────────────────────
    st.divider()
    with st.expander("Model Validation — Backtest & CV RMSE", expanded=False):
        st.subheader("Model Accuracy: Full 4-Season vs Historical (2022-23 Only)")
        st.caption(
            "Historical model trained on 2022-23 stats only (5-fold OOF cross_val_predict). "
            "Full model trained on 4-season weighted averages. "
            "Lower RMSE = better accuracy (log-scale)."
        )
        try:
            metrics_df = get_model_metrics()
            if not metrics_df.empty:
                # RMSE comparison bar chart
                try:
                    _conn = sqlite3.connect(DB_PATH)
                    bt_df = pd.read_sql(
                        "SELECT pos, hist_pred_eur, full_pred_eur, market_value_eur "
                        "FROM backtest_results WHERE market_value_eur > 0 "
                        "AND hist_pred_eur IS NOT NULL AND full_pred_eur IS NOT NULL",
                        _conn,
                    )
                    _conn.close()
                except Exception:
                    bt_df = pd.DataFrame()

                rmse_rows = []
                for _, mrow in metrics_df.iterrows():
                    pos = mrow["pos_group"]
                    full_rmse = mrow["rmse"]
                    # Compute historical RMSE from backtest results
                    hist_rmse = np.nan
                    if not bt_df.empty:
                        sub = bt_df[bt_df["pos"].str.upper().str.contains(pos, na=False)].copy()
                        sub["hist_pred_eur"] = pd.to_numeric(sub["hist_pred_eur"], errors="coerce")
                        sub["full_pred_eur"] = pd.to_numeric(sub["full_pred_eur"], errors="coerce")
                        sub = sub.dropna(subset=["hist_pred_eur", "full_pred_eur"])
                        if len(sub) >= 5:
                            log_true  = np.log1p(sub["market_value_eur"].values)
                            log_hist  = np.log1p(sub["hist_pred_eur"].values)
                            hist_rmse = float(np.sqrt(np.mean((log_hist - log_true)**2)))
                    rmse_rows.append({"Position": pos, "Historical (2223)": round(hist_rmse, 4) if not np.isnan(hist_rmse) else None, "Full 4-Season": round(full_rmse, 4), "N train": mrow["n_train"]})

                rmse_display = pd.DataFrame(rmse_rows)

                fig_rmse = go.Figure()
                fig_rmse.add_trace(go.Bar(
                    name="Historical (2022-23 only)",
                    x=rmse_display["Position"],
                    y=rmse_display["Historical (2223)"],
                    marker_color="#F77E7E",
                ))
                fig_rmse.add_trace(go.Bar(
                    name="Full 4-Season Model",
                    x=rmse_display["Position"],
                    y=rmse_display["Full 4-Season"],
                    marker_color="#7EF7A8",
                ))
                fig_rmse.update_layout(
                    barmode="group", height=320,
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#ffffff", yaxis_title="RMSE (log-scale)",
                    margin=dict(t=20, b=30),
                    legend=dict(orientation="h", y=-0.25),
                )
                fig_rmse.update_yaxes(gridcolor="#222")
                st.plotly_chart(fig_rmse, use_container_width=True)

                # Summary improvement
                valid = [(r["Historical (2223)"], r["Full 4-Season"]) for _, r in rmse_display.iterrows()
                         if r["Historical (2223)"] is not None and r["Historical (2223)"] > 0]
                if valid:
                    avg_improvement = np.mean([(h - f) / h * 100 for h, f in valid])
                    st.metric(
                        "Average RMSE improvement from multi-season weighting",
                        f"{avg_improvement:.1f}%",
                        help="Positive = full model is more accurate than single-season model",
                    )

                st.dataframe(rmse_display, use_container_width=True, hide_index=True)

            if not bt_df.empty:
                st.divider()
                st.subheader("Actual vs Predicted — Backtest Scatter")
                bt_plot = bt_df.copy()
                bt_plot["Actual (M€)"]     = bt_plot["market_value_eur"] / 1e6
                bt_plot["Historical (M€)"] = pd.to_numeric(bt_plot["hist_pred_eur"], errors="coerce") / 1e6
                bt_plot["Full (M€)"]       = pd.to_numeric(bt_plot["full_pred_eur"], errors="coerce") / 1e6
                bt_plot = bt_plot.dropna(subset=["Historical (M€)", "Full (M€)"])

                bt_col1, bt_col2 = st.columns(2)
                with bt_col1:
                    st.caption("Historical model (2022-23 only)")
                    fig_bt1 = px.scatter(
                        bt_plot, x="Actual (M€)", y="Historical (M€)", color="pos",
                        opacity=0.55, height=340,
                        color_discrete_map={"FW": "#7EB8F7", "MF": "#F7C97E", "DF": "#7EF7A8", "GK": "#F77E7E"},
                    )
                    _mx1 = max(bt_plot["Actual (M€)"].max(), bt_plot["Historical (M€)"].max())
                    fig_bt1.add_trace(go.Scatter(x=[0, _mx1], y=[0, _mx1], mode="lines",
                                                  line=dict(color="#555", dash="dash", width=1), showlegend=False))
                    fig_bt1.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                           font_color="#ffffff", margin=dict(t=10, b=30))
                    fig_bt1.update_xaxes(gridcolor="#222")
                    fig_bt1.update_yaxes(gridcolor="#222")
                    st.plotly_chart(fig_bt1, use_container_width=True)
                with bt_col2:
                    st.caption("Full 4-season model")
                    fig_bt2 = px.scatter(
                        bt_plot, x="Actual (M€)", y="Full (M€)", color="pos",
                        opacity=0.55, height=340,
                        color_discrete_map={"FW": "#7EB8F7", "MF": "#F7C97E", "DF": "#7EF7A8", "GK": "#F77E7E"},
                    )
                    _mx2 = max(bt_plot["Actual (M€)"].max(), bt_plot["Full (M€)"].max())
                    fig_bt2.add_trace(go.Scatter(x=[0, _mx2], y=[0, _mx2], mode="lines",
                                                  line=dict(color="#555", dash="dash", width=1), showlegend=False))
                    fig_bt2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                           font_color="#ffffff", margin=dict(t=10, b=30))
                    fig_bt2.update_xaxes(gridcolor="#222")
                    fig_bt2.update_yaxes(gridcolor="#222")
                    st.plotly_chart(fig_bt2, use_container_width=True)

        except Exception as e:
            st.info(f"Model validation data unavailable: {e}")

    # ── Age Curve ─────────────────────────────────────────────
    st.divider()
    with st.expander("Age-Value Curve — Position-Specific", expanded=False):
        st.subheader("Player Value Peak Age by Position")
        st.caption("Polynomial regression (degree 2) of market value vs age, fitted on players with TM valuations")

        ac_pos = st.selectbox("Position group", ["FW", "MF", "DF", "GK"], key="age_curve_pos")

        try:
            ac_data = get_age_curve_data(ac_pos)
            if ac_data.get("has_data"):
                fig_ac = go.Figure()
                # Scatter: individual players (background)
                fig_ac.add_trace(go.Scatter(
                    x=ac_data["scatter_age"], y=ac_data["scatter_value_m"],
                    mode="markers", name="Individual players",
                    text=ac_data["scatter_player"],
                    marker=dict(color="#7EB8F7", opacity=0.30, size=4),
                    hovertemplate="%{text}<br>Age: %{x}<br>Value: €%{y:.1f}M<extra></extra>",
                ))
                # Median values per age
                if "med_age" in ac_data and "med_value_m" in ac_data:
                    fig_ac.add_trace(go.Scatter(
                        x=ac_data["med_age"], y=ac_data["med_value_m"],
                        mode="markers+lines", name="Median value by age",
                        marker=dict(color="#F77E7E", size=7),
                        line=dict(color="#F77E7E", width=1.5, dash="dot"),
                        hovertemplate="Age %{x}: median €%{y:.1f}M<extra></extra>",
                    ))
                # Polynomial trend curve
                fig_ac.add_trace(go.Scatter(
                    x=ac_data["curve_age"], y=ac_data["curve_value_m"],
                    mode="lines", name="Polynomial trend",
                    line=dict(color="#F7C97E", width=2.5),
                ))
                # Peak age marker on median line
                peak = ac_data["peak_age"]
                fig_ac.add_vline(
                    x=peak, line_dash="dash", line_color="#7EF7A8", line_width=1.5,
                    annotation_text=f"Median peak: {peak}", annotation_position="top right",
                    annotation_font_color="#7EF7A8",
                )
                fig_ac.update_layout(
                    xaxis_title="Age", yaxis_title="Market Value (M€)",
                    height=420,
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#ffffff",
                    legend=dict(orientation="h", y=-0.18),
                    margin=dict(t=20, b=60),
                )
                fig_ac.update_xaxes(gridcolor="#222")
                fig_ac.update_yaxes(gridcolor="#222")
                st.plotly_chart(fig_ac, use_container_width=True)
                st.caption(
                    f"n = {ac_data['n_players']} {ac_pos} players with TM valuations. "
                    f"Median peak age = {peak}. "
                    "Note: youth premium inflates values for young players — market value peak reflects transfer market dynamics, not performance peak."
                )
            else:
                st.info(f"Not enough {ac_pos} data for age curve.")
        except Exception as e:
            st.info(f"Age curve unavailable: {e}")


# ════════════════════════════════════════════════════════════
# TAB 3 — Team Fit Scoring
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Team Fit Scoring")
    st.caption(
        "Find players whose statistical profile best matches a target team's playing style. "
        "Team DNA = mean tactical stats of current players in the same position group."
    )

    col_tf1, col_tf2, col_tf3, col_tf4 = st.columns([3, 2, 2, 2])

    with col_tf1:
        all_teams = get_all_teams()
        selected_tf_team = st.selectbox(
            "Target Team",
            options=all_teams,
            index=all_teams.index("Arsenal") if "Arsenal" in all_teams else 0,
            key="tf_team",
        )
    with col_tf2:
        tf_position = st.selectbox(
            "Position Filter",
            options=["All", "FW", "MF", "DF"],
            key="tf_pos",
        )
    with col_tf3:
        tf_budget = st.number_input(
            "Max Value (M€)",
            min_value=0.0, max_value=300.0, value=50.0, step=5.0,
            key="tf_budget",
        )
    with col_tf4:
        tf_topn = st.number_input("Top N", min_value=5, max_value=30, value=15, key="tf_topn")

    if st.button("Find Team Fits", type="primary", key="tf_btn"):
        with st.spinner(f"Analysing {selected_tf_team}'s DNA..."):
            fit_df = get_team_fit_players(
                target_team=selected_tf_team,
                position=None if tf_position == "All" else tf_position,
                max_value_eur=float(tf_budget) * 1_000_000 if tf_budget > 0 else None,
                top_n=int(tf_topn),
            )

        if fit_df.empty:
            st.warning(f"No fit results for {selected_tf_team}. Try a different position filter.")
        else:
            st.success(f"Top {len(fit_df)} players matching {selected_tf_team}'s style")
            st.dataframe(
                fit_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Fit Score (%)": st.column_config.ProgressColumn(
                        "Fit Score (%)", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "Undervalue (%)": st.column_config.NumberColumn("Undervalue (%)", format="%.1f"),
                    "Value (M€)": st.column_config.NumberColumn("Value (M€)", format="%.1f"),
                    "Goals/90": st.column_config.NumberColumn("Goals/90", format="%.2f"),
                    "Assists/90": st.column_config.NumberColumn("Assists/90", format="%.2f"),
                    "xG/90": st.column_config.NumberColumn("xG/90", format="%.3f"),
                    "xA/90": st.column_config.NumberColumn("xA/90", format="%.3f"),
                },
            )

            # Radar comparison: team DNA vs top 3 fits
            st.divider()
            st.subheader(f"{selected_tf_team} DNA vs Top Fits")

            fit_player_names = fit_df["player"].tolist()[:3]
            conn = sqlite3.connect(DB_PATH)
            team_dna_raw = pd.read_sql(
                f"SELECT * FROM players_master WHERE team = ?",
                conn, params=[selected_tf_team],
            )
            conn.close()

            if not team_dna_raw.empty and fit_player_names:
                pcts = get_player_percentiles(fit_player_names)
                if not pcts.empty:
                    st.plotly_chart(_build_radar(pcts, height=420), use_container_width=True)
