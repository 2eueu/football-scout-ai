import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from models.search import parse_query, search_players, get_player_detail
from models.form import get_form_trend
from models.value_scouting import get_undervalued

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
    '<p class="subtitle">Big 5 리그 멀티시즌(22-23 ~ 25-26) 가중 평균 데이터 기반</p>',
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2 = st.tabs(["🔍 선수 검색", "💰 Value Scouting"])


# ════════════════════════════════════════════════════════════
# TAB 1 — 선수 검색
# ════════════════════════════════════════════════════════════
with tab1:
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            label="검색",
            placeholder="예: 골 잘 넣는 20대 초반 공격수 / 프리미어리그 압박 강한 미드필더",
            label_visibility="collapsed",
            key="search_query",
        )
    with col_btn:
        search_clicked = st.button("검색", use_container_width=True, type="primary", key="search_btn")

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.filters = None

    if search_clicked and query.strip():
        with st.spinner("AI가 조건을 분석하는 중..."):
            try:
                filters = parse_query(query)
                results = search_players(filters)
                st.session_state.results = results
                st.session_state.filters = filters
            except Exception as e:
                st.error(f"검색 오류: {e}")

    if st.session_state.results is not None:
        results: pd.DataFrame = st.session_state.results
        filters = st.session_state.filters

        with st.expander("AI 파싱 결과", expanded=False):
            st.code(json.dumps(filters, ensure_ascii=False, indent=2), language="json")

        if results.empty:
            st.warning("조건에 맞는 선수가 없어요. 조건을 완화해보세요.")
            st.stop()

        st.markdown(f"**{len(results)}명** 검색됨")

        display = results.rename(columns={
            "player": "선수", "team": "팀", "pos": "포지션",
            "age": "나이", "league": "리그", "minutes": "출전(분)",
            "goals_p90": "골/90", "assists_p90": "어시스트/90",
            "total_goals": "총골", "total_assists": "총어시스트",
            "tackles_won": "태클", "interceptions": "인터셉트", "fouls": "파울",
        })
        st.dataframe(
            display, use_container_width=True, hide_index=True,
            column_config={
                "골/90": st.column_config.ProgressColumn("골/90", min_value=0, max_value=1.5, format="%.2f"),
                "어시스트/90": st.column_config.ProgressColumn("어시스트/90", min_value=0, max_value=1.0, format="%.2f"),
            },
        )

        # ── 선수 비교 레이더 차트 ────────────────────────────
        st.divider()
        st.subheader("선수 비교")
        player_names = results["player"].tolist()
        default = player_names[:2] if len(player_names) >= 2 else player_names
        selected = st.multiselect("비교할 선수 선택 (최대 3명)", player_names, default=default, max_selections=3, key="compare_select")

        if len(selected) >= 2:
            raw_cols = ["goals_p90", "assists_p90", "tackles_won", "interceptions", "minutes"]
            labels   = ["골/90", "어시스트/90", "태클", "인터셉트", "출전시간"]
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

        # ── 선수 상세 ────────────────────────────────────────
        st.divider()
        st.subheader("선수 상세")
        detail_name = st.selectbox("선수 선택", player_names, key="detail_select")
        if detail_name:
            row = results[results["player"] == detail_name].iloc[0]
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            for col, (lbl, val) in zip(
                [c1, c2, c3, c4, c5, c6],
                [("팀", row["team"]), ("포지션", row["pos"]), ("나이", str(row["age"])),
                 ("출전(분)", f"{int(row['minutes']):,}"), ("골/90", f"{row['goals_p90']:.2f}"),
                 ("어시스트/90", f"{row['assists_p90']:.2f}")],
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
                st.markdown(f"\n**StatsBomb 이벤트 ({int(sb['matches_played'])}경기)**")
                s1, s2, s3, s4, s5 = st.columns(5)
                for col, (lbl, key) in zip(
                    [s1, s2, s3, s4, s5],
                    [("슈팅","total_shots"),("패스","total_passes"),
                     ("압박","total_pressures"),("드리블","total_dribbles"),("태클","total_tackles")],
                ):
                    with col:
                        st.metric(lbl, int(sb.get(key) or 0))

            # 폼 트렌드
            st.divider()
            st.subheader("폼 트렌드 (LSTM)")
            with st.spinner("폼 분석 중..."):
                form = get_form_trend(detail_name)

            if not form.get("has_data"):
                st.info("StatsBomb 이벤트 데이터가 없는 선수입니다.")
            else:
                t1, t2, t3 = st.columns(3)
                t1.metric("트렌드", form["trend"])
                t2.metric("최근 5경기 기울기", form["slope"])
                t3.metric("다음 경기 예측", form["prediction"] or "N/A")

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=form["match_ids"], y=form["scores"],
                    mode="lines+markers", name="폼 스코어",
                    line=dict(color="#7EB8F7", width=2), marker=dict(size=6),
                ))
                scores_arr = np.array(form["scores"])
                if len(scores_arr) >= 3:
                    ma = np.convolve(scores_arr, np.ones(3)/3, mode="valid")
                    fig2.add_trace(go.Scatter(
                        x=list(range(3, len(scores_arr)+1)), y=ma.tolist(),
                        mode="lines", name="3경기 이동평균",
                        line=dict(color="#F7C97E", width=2, dash="dash"),
                    ))
                if form["prediction"] is not None:
                    fig2.add_trace(go.Scatter(
                        x=[len(form["match_ids"])+1], y=[form["prediction"]],
                        mode="markers", name="다음 경기 예측 (LSTM)",
                        marker=dict(color="#7EF7A8", size=12, symbol="star"),
                    ))
                fig2.update_layout(
                    xaxis_title="경기", yaxis_title="폼 스코어", height=350,
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#ffffff", legend=dict(orientation="h", y=-0.2),
                    margin=dict(t=20, b=60),
                )
                fig2.update_xaxes(gridcolor="#333")
                fig2.update_yaxes(gridcolor="#333")
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("폼 스코어 = 슈팅×2 + 드리블×1.5 + 태클×1 + 인터셉트×1 + 압박×0.5 + 패스×0.3")


# ════════════════════════════════════════════════════════════
# TAB 2 — Value Scouting
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 저평가 선수 발굴")
    st.caption("XGBoost 모델이 퍼포먼스 기반 적정 몸값을 예측 — 실제 시장가치보다 예측값이 높은 선수가 저평가 선수")

    # ── 필터 ─────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        budget = st.slider("최대 예산 (M€)", min_value=1, max_value=150, value=30, step=1)
    with f2:
        pos_filter = st.selectbox("포지션", ["전체", "FW", "MF", "DF", "GK"])
    with f3:
        league_filter = st.selectbox("리그", [
            "전체", "ENG-Premier League", "ESP-La Liga",
            "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1",
        ])
    with f4:
        age_filter = st.slider("최대 나이", min_value=16, max_value=38, value=28)

    top_n = st.slider("표시 인원", min_value=5, max_value=50, value=20)

    if st.button("분석 실행", type="primary", key="value_btn"):
        with st.spinner("저평가 선수 분석 중..."):
            try:
                df_val = get_undervalued(
                    max_value_eur=budget * 1_000_000,
                    position=None if pos_filter == "전체" else pos_filter,
                    league=None if league_filter == "전체" else league_filter,
                    age_max=age_filter,
                    top_n=top_n,
                )
                st.session_state.value_results = df_val
            except Exception as e:
                st.error(f"오류: {e}")

    if "value_results" not in st.session_state:
        st.session_state.value_results = None

    if st.session_state.value_results is not None:
        df_val: pd.DataFrame = st.session_state.value_results

        if df_val.empty:
            st.warning("조건에 맞는 선수가 없어요.")
        else:
            st.markdown(f"**{len(df_val)}명** 발굴됨")

            # ── 산점도: 실제 vs 예측 몸값 ────────────────────
            st.divider()
            st.subheader("실제 몸값 vs 예측 몸값")
            st.caption("대각선 위 = 저평가 (예측 > 실제) | 대각선 아래 = 고평가")

            # raw 데이터 다시 로드 (scatter용)
            import sqlite3, os
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

            scatter_df["실제(M€)"]  = scatter_df["market_value_eur"] / 1e6
            scatter_df["예측(M€)"]  = scatter_df["predicted_value_eur"] / 1e6
            scatter_df["저평가(%)"] = scatter_df["undervalue_score"].round(1)
            scatter_df["골/90"]     = pd.to_numeric(scatter_df["per_90_minutes_gls"], errors="coerce").fillna(0).round(2)

            fig_s = px.scatter(
                scatter_df,
                x="실제(M€)", y="예측(M€)",
                color="league",
                hover_name="player",
                hover_data={"team": True, "pos": True, "age": True,
                            "골/90": True, "저평가(%)": True,
                            "실제(M€)": ":.1f", "예측(M€)": ":.1f"},
                color_discrete_map={
                    "ENG-Premier League": "#7EB8F7",
                    "ESP-La Liga":        "#F7C97E",
                    "GER-Bundesliga":     "#7EF7A8",
                    "ITA-Serie A":        "#F77E7E",
                    "FRA-Ligue 1":        "#C47EF7",
                },
                height=500,
            )

            # 등가선 (y = x)
            max_val = max(scatter_df["실제(M€)"].max(), scatter_df["예측(M€)"].max())
            fig_s.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", name="등가선 (적정가)",
                line=dict(color="#555", dash="dash", width=1),
            ))

            fig_s.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font_color="#ffffff", legend_title="리그",
                margin=dict(t=20, b=40),
            )
            fig_s.update_xaxes(gridcolor="#222", title="실제 시장가치 (M€)")
            fig_s.update_yaxes(gridcolor="#222", title="예측 적정가치 (M€)")
            st.plotly_chart(fig_s, use_container_width=True)

            # ── 저평가 선수 테이블 ────────────────────────────
            st.divider()
            st.subheader(f"저평가 TOP {len(df_val)}")

            st.dataframe(
                df_val,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "저평가점수(%)": st.column_config.ProgressColumn(
                        "저평가점수(%)", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "골/90": st.column_config.NumberColumn("골/90", format="%.2f"),
                    "어시스트/90": st.column_config.NumberColumn("어시스트/90", format="%.2f"),
                    "실제몸값(M€)": st.column_config.NumberColumn("실제 (M€)", format="%.1f"),
                    "예측몸값(M€)": st.column_config.NumberColumn("예측 (M€)", format="%.1f"),
                },
            )

            # ── 선수 상세 카드 ────────────────────────────────
            st.divider()
            st.subheader("선수 상세")
            val_players = df_val["player"].tolist()
            selected_val = st.selectbox("선수 선택", val_players, key="val_detail")

            if selected_val:
                row = df_val[df_val["player"] == selected_val].iloc[0]
                actual  = row["실제몸값(M€)"]
                predict = row["예측몸값(M€)"]
                score   = row["저평가점수(%)"]
                tag     = "tag-under" if score > 0 else "tag-over"
                label   = f"저평가 +{score:.1f}%" if score > 0 else f"고평가 {score:.1f}%"

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("팀", row["team"])
                m2.metric("포지션", row["pos"])
                m3.metric("나이", str(row["age"]))
                m4.metric("실제 몸값", f"€{actual:.1f}M")
                m5.metric("예측 몸값", f"€{predict:.1f}M", delta=f"{score:+.1f}%")

                st.markdown(
                    f'<span class="{tag}">{label}</span>',
                    unsafe_allow_html=True,
                )

                # 가치 비교 바 차트
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    name="실제 시장가치",
                    x=["시장가치 비교"],
                    y=[actual],
                    marker_color="#F77E7E",
                    width=0.3,
                ))
                fig_bar.add_trace(go.Bar(
                    name="예측 적정가치",
                    x=["시장가치 비교"],
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
