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
    if DB.exists():
        return
    DATA = Path(__file__).parent.parent / "data"
    tables = {
        "players_master":   DATA / "players_master.parquet",
        "value_scouting":   DATA / "value_scouting.parquet",
        "statsbomb_events": DATA / "statsbomb_events.parquet",
        "players_raw":      DATA / "players_raw.parquet",
    }
    missing = [name for name, path in tables.items() if not path.exists()]
    if missing:
        st.error(f"Missing data files: {missing}")
        st.stop()
    conn = sqlite3.connect(DB)
    for table, path in tables.items():
        pd.read_parquet(path).to_sql(table, conn, if_exists="replace", index=False)
    conn.close()

_bootstrap_db()

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass  # local: key comes from .env via python-dotenv in models/search.py

from models.search import parse_query, search_players, get_player_detail
from models.form import get_form_trend, get_season_trend
from models.value_scouting import get_undervalued, get_similar_players

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
    '<p class="subtitle">Big 5 leagues · Multi-season weighted data (22/23 – 25/26) · XGBoost value model · NL search</p>',
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2 = st.tabs(["🔍 Player Search", "💰 Value Scouting"])


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

        display = results.rename(columns={
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
            },
        )

        # ── Player comparison radar ──────────────────────────
        st.divider()
        st.subheader("Player Comparison")
        player_names = results["player"].tolist()
        default = player_names[:2] if len(player_names) >= 2 else player_names
        selected = st.multiselect("Select players to compare (max 3)", player_names, default=default, max_selections=3, key="compare_select")

        if len(selected) >= 2:
            raw_cols = ["goals_p90", "assists_p90", "tackles_won", "interceptions", "minutes"]
            labels   = ["Goals/90", "Assists/90", "Tackles", "Interceptions", "Minutes"]
            colors   = ["#7EB8F7", "#F7C97E", "#7EF7A8"]
            norm = results[["player"] + raw_cols].copy()
            for col in raw_cols:
                mx = norm[col].max()
                norm[col] = norm[col] / mx if mx > 0 else 0

            fig = go.Figure()
            for i, player in enumerate(selected):
                row = norm[norm["player"] == player]
                if row.empty: continue
                vals = row[raw_cols].iloc[0].tolist()
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=labels + [labels[0]],
                    fill="toself", name=player,
                    line_color=colors[i], fillcolor=colors[i], opacity=0.3,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=450,
                paper_bgcolor="#0e1117", font_color="#ffffff",
                margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Player detail ────────────────────────────────────
        st.divider()
        st.subheader("Player Detail")
        detail_name = st.selectbox("Select player", player_names, key="detail_select")
        if detail_name:
            row = results[results["player"] == detail_name].iloc[0]
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


# ════════════════════════════════════════════════════════════
# TAB 2 — Value Scouting
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Undervalued Player Detection")
    st.caption("XGBoost model predicts performance-based fair value — players where predicted > actual market value are flagged as undervalued")

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

            # ── Scatter: actual vs predicted value ───────────
            st.divider()
            st.subheader("Actual vs Predicted Market Value")
            st.caption("Above the line = undervalued (predicted > actual) | Below = overvalued")

            DB_PATH = Path(__file__).parent.parent / "scout.db"
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
                    "저평가점수(%)": st.column_config.ProgressColumn(
                        "Undervalue (%)", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "골/90": st.column_config.NumberColumn("Goals/90", format="%.2f"),
                    "어시스트/90": st.column_config.NumberColumn("Assists/90", format="%.2f"),
                    "실제몸값(M€)": st.column_config.NumberColumn("Actual (M€)", format="%.1f"),
                    "예측몸값(M€)": st.column_config.NumberColumn("Predicted (M€)", format="%.1f"),
                },
            )

            # ── Player detail card ────────────────────────────
            st.divider()
            st.subheader("Player Detail")
            val_players = df_val["player"].tolist()
            selected_val = st.selectbox("Select player", val_players, key="val_detail")

            if selected_val:
                row = df_val[df_val["player"] == selected_val].iloc[0]
                actual  = row["실제몸값(M€)"]
                predict = row["예측몸값(M€)"]
                score   = row["저평가점수(%)"]
                tag     = "tag-under" if score > 0 else "tag-over"
                label   = f"Undervalued +{score:.1f}%" if score > 0 else f"Overvalued {score:.1f}%"

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Club", row["team"])
                m2.metric("Position", row["pos"])
                m3.metric("Age", str(row["age"]))
                m4.metric("Market value", f"€{actual:.1f}M")
                m5.metric("Model estimate", f"€{predict:.1f}M", delta=f"{score:+.1f}%")

                st.markdown(f'<span class="{tag}">{label}</span>', unsafe_allow_html=True)

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    name="Actual market value",
                    x=["Value comparison"],
                    y=[actual],
                    marker_color="#F77E7E",
                    width=0.3,
                ))
                fig_bar.add_trace(go.Bar(
                    name="Model estimate",
                    x=["Value comparison"],
                    y=[predict],
                    marker_color="#7EF7A8",
                    width=0.3,
                ))
                fig_bar.update_layout(
                    barmode="group", height=300,
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#ffffff", yaxis_title="M€",
                    margin=dict(t=20, b=20),
                    legend=dict(orientation="h", y=-0.3),
                )
                fig_bar.update_yaxes(gridcolor="#222")
                st.plotly_chart(fig_bar, use_container_width=True)

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
                        },
                    )
