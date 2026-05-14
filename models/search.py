"""
자연어 → 구조화된 필터 파싱 + 선수 DB 검색
"""

import os
import json
import sqlite3
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(__file__).parent.parent / "scout.db"

GEMINI_SYSTEM_PROMPT = """
You are a query parser for a football scouting AI. Accept input in English or Korean.
Convert the natural language input into the JSON format below.
Set missing fields to null. Return JSON only — no explanation.

Output format:
{
  "position": ["FW"] | ["MF"] | ["DF"] | ["GK"] | ["FW","MF"] etc | null,
  "age_min": number | null,
  "age_max": number | null,
  "league": ["ENG-Premier League"] | ["ESP-La Liga"] | ["GER-Bundesliga"] | ["ITA-Serie A"] | ["FRA-Ligue 1"] | null,
  "min_goals_per90": number | null,
  "min_assists_per90": number | null,
  "min_pressures": number | null,
  "min_tackles": number | null,
  "min_interceptions": number | null,
  "min_minutes": number | null,
  "sort_by": "goals" | "assists" | "pressures" | "tackles" | "form" | "undervalue" | null,
  "limit": number (default 10)
}

sort_by rules:
- "form": "in form", "hot", "best form", "현재 폼 좋은", "요즘 잘하는", "이번 시즌"
- "undervalue": "undervalued", "hidden gems", "bargain", "저평가", "가성비"
- "goals": goal-related queries
- "tackles" / "pressures": defensive / pressing queries

Position mapping:
- striker / forward / FW / 공격수 / 스트라이커 → ["FW"]
- midfielder / MF / 미드필더 → ["MF"]
- defender / DF / 수비수 → ["DF"]
- goalkeeper / GK / 골키퍼 → ["GK"]
- winger / 윙어 → ["FW", "MF"]
- defensive mid / CDM / 수비형미드 → ["MF"]

League mapping:
- Premier League / EPL / England / 프리미어리그 / 잉글랜드 → ENG-Premier League
- La Liga / Spain / 라리가 / 스페인 → ESP-La Liga
- Bundesliga / Germany / 분데스리가 / 독일 → GER-Bundesliga
- Serie A / Italy / 세리에A / 이탈리아 → ITA-Serie A
- Ligue 1 / France / 리그앙 / 프랑스 → FRA-Ligue 1
"""


def parse_query(text: str) -> dict:
    """자연어 → 구조화된 필터 (Groq API)"""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(".env 파일에 GROQ_API_KEY를 설정해주세요")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": GEMINI_SYSTEM_PROMPT},
            {"role": "user", "content": f"입력: {text}"},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json\s*|\s*```", "", raw).strip()

    filters = json.loads(raw)
    return filters


def build_query(filters: dict) -> tuple[str, list]:
    """필터 dict → SQL WHERE 절 + 파라미터"""
    conditions = []
    params = []

    if filters.get("position"):
        pos_conditions = [f"pos LIKE ?" for _ in filters["position"]]
        conditions.append(f"({' OR '.join(pos_conditions)})")
        params.extend([f"%{p}%" for p in filters["position"]])

    if filters.get("age_min") is not None:
        conditions.append("CAST(age AS INTEGER) >= ?")
        params.append(int(filters["age_min"]))

    if filters.get("age_max") is not None:
        conditions.append("CAST(age AS INTEGER) <= ?")
        params.append(int(filters["age_max"]))

    if filters.get("league"):
        placeholders = ",".join(["?" for _ in filters["league"]])
        conditions.append(f"league IN ({placeholders})")
        params.extend(filters["league"])

    if filters.get("min_goals_per90") is not None:
        conditions.append("CAST(per_90_minutes_gls AS REAL) >= ?")
        params.append(float(filters["min_goals_per90"]))

    if filters.get("min_assists_per90") is not None:
        conditions.append("CAST(per_90_minutes_ast AS REAL) >= ?")
        params.append(float(filters["min_assists_per90"]))

    if filters.get("min_minutes") is not None:
        conditions.append("CAST(playing_time_min AS INTEGER) >= ?")
        params.append(int(filters["min_minutes"]))

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    return where_clause, params


SORT_COLUMN_MAP = {
    "goals":     "CAST(m.per_90_minutes_gls AS REAL)",
    "assists":   "CAST(m.per_90_minutes_ast AS REAL)",
    "pressures": "CAST(m.performance_fld AS REAL)",
    "tackles":   "CAST(m.performance_tklw AS REAL)",
}


def _prefix_where(where_clause: str) -> str:
    """Prefix players_master columns with 'm.' alias for JOIN queries."""
    w = where_clause
    w = w.replace("pos LIKE", "m.pos LIKE")
    w = w.replace("CAST(age", "CAST(m.age")
    w = w.replace("league IN", "m.league IN")
    w = w.replace("CAST(per_90", "CAST(m.per_90")
    w = w.replace("CAST(playing_time", "CAST(m.playing_time")
    return w


def search_players(filters: dict) -> pd.DataFrame:
    """필터 dict → 선수 검색 결과 DataFrame"""
    where_clause, params = build_query(filters)
    sort_by = filters.get("sort_by") or ""
    limit = int(filters.get("limit") or 10)
    where_master = _prefix_where(where_clause)

    if sort_by == "form":
        sql = f"""
            SELECT
                m.player, m.team, m.pos,
                CAST(SUBSTR(CAST(m.age AS TEXT), 1,
                     CASE WHEN INSTR(CAST(m.age AS TEXT), '-') > 0
                          THEN INSTR(CAST(m.age AS TEXT), '-') - 1
                          ELSE LENGTH(CAST(m.age AS TEXT)) END) AS INTEGER) AS age,
                m.league,
                CAST(m.playing_time_min AS INTEGER) AS minutes,
                ROUND(CAST(m.per_90_minutes_gls AS REAL), 2) AS goals_p90,
                ROUND(CAST(m.per_90_minutes_ast AS REAL), 2) AS assists_p90,
                CAST(m.performance_gls AS INTEGER) AS total_goals,
                CAST(m.performance_ast AS INTEGER) AS total_assists,
                MAX(CAST(m.performance_tklw AS INTEGER)) AS tackles_won,
                MAX(CAST(m.performance_int AS INTEGER)) AS interceptions,
                CAST(m.performance_fls AS INTEGER) AS fouls,
                ROUND(COALESCE(CAST(r.per_90_minutes_gls AS REAL), 0) +
                      COALESCE(CAST(r.per_90_minutes_ast AS REAL), 0), 3) AS form_score
            FROM players_master m
            LEFT JOIN players_raw r
                ON m.player = r.player AND r.season = '2526'
                   AND CAST(r.playing_time_min AS INTEGER) >= 90
            WHERE {where_master}
              AND m.player IS NOT NULL AND m.player != ''
              AND CAST(m.playing_time_min AS INTEGER) > 0
            GROUP BY m.player, m.team, m.league
            ORDER BY form_score DESC
            LIMIT ?
        """
    elif sort_by == "undervalue":
        sql = f"""
            SELECT
                m.player, m.team, m.pos,
                CAST(SUBSTR(CAST(m.age AS TEXT), 1,
                     CASE WHEN INSTR(CAST(m.age AS TEXT), '-') > 0
                          THEN INSTR(CAST(m.age AS TEXT), '-') - 1
                          ELSE LENGTH(CAST(m.age AS TEXT)) END) AS INTEGER) AS age,
                m.league,
                CAST(m.playing_time_min AS INTEGER) AS minutes,
                ROUND(CAST(m.per_90_minutes_gls AS REAL), 2) AS goals_p90,
                ROUND(CAST(m.per_90_minutes_ast AS REAL), 2) AS assists_p90,
                CAST(m.performance_gls AS INTEGER) AS total_goals,
                CAST(m.performance_ast AS INTEGER) AS total_assists,
                MAX(CAST(m.performance_tklw AS INTEGER)) AS tackles_won,
                MAX(CAST(m.performance_int AS INTEGER)) AS interceptions,
                CAST(m.performance_fls AS INTEGER) AS fouls,
                ROUND(COALESCE(CAST(v.undervalue_score AS REAL), 0), 1) AS undervalue_score
            FROM players_master m
            LEFT JOIN value_scouting v ON m.player = v.player
            WHERE {where_master}
              AND m.player IS NOT NULL AND m.player != ''
              AND CAST(m.playing_time_min AS INTEGER) > 0
              AND CAST(v.undervalue_score AS REAL) > 0
            GROUP BY m.player, m.team, m.league
            ORDER BY undervalue_score DESC
            LIMIT ?
        """
    else:
        sort_col = SORT_COLUMN_MAP.get(sort_by, "CAST(m.per_90_minutes_gls AS REAL)")
        sql = f"""
            SELECT
                m.player, m.team, m.pos,
                CAST(SUBSTR(CAST(m.age AS TEXT), 1,
                     CASE WHEN INSTR(CAST(m.age AS TEXT), '-') > 0
                          THEN INSTR(CAST(m.age AS TEXT), '-') - 1
                          ELSE LENGTH(CAST(m.age AS TEXT)) END) AS INTEGER) AS age,
                m.league,
                CAST(m.playing_time_min AS INTEGER) AS minutes,
                ROUND(CAST(m.per_90_minutes_gls AS REAL), 2) AS goals_p90,
                ROUND(CAST(m.per_90_minutes_ast AS REAL), 2) AS assists_p90,
                CAST(m.performance_gls AS INTEGER) AS total_goals,
                CAST(m.performance_ast AS INTEGER) AS total_assists,
                MAX(CAST(m.performance_tklw AS INTEGER)) AS tackles_won,
                MAX(CAST(m.performance_int AS INTEGER)) AS interceptions,
                CAST(m.performance_fls AS INTEGER) AS fouls
            FROM players_master m
            WHERE {where_master}
              AND m.player IS NOT NULL AND m.player != ''
              AND CAST(m.playing_time_min AS INTEGER) > 0
            GROUP BY m.player, m.team, m.league
            ORDER BY {sort_col} DESC
            LIMIT ?
        """

    params.append(limit)
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


def get_player_detail(player_name: str) -> dict:
    """선수 이름으로 전체 스탯 조회"""
    conn = sqlite3.connect(DB_PATH)

    master = pd.read_sql(
        "SELECT * FROM players_master WHERE player = ? LIMIT 1",
        conn, params=[player_name]
    )

    statsbomb = pd.read_sql(
        """
        SELECT
            COUNT(*) as matches_played,
            SUM(shots) as total_shots,
            SUM(passes) as total_passes,
            SUM(pressures) as total_pressures,
            SUM(dribbles) as total_dribbles,
            SUM(tackles) as total_tackles,
            SUM(interceptions) as total_interceptions
        FROM statsbomb_events
        WHERE player = ?
        """,
        conn, params=[player_name]
    )

    conn.close()

    if master.empty:
        return {}

    result = master.iloc[0].to_dict()
    if not statsbomb.empty and statsbomb.iloc[0]["matches_played"] > 0:
        result["statsbomb"] = statsbomb.iloc[0].to_dict()

    return result


def search(text: str) -> pd.DataFrame:
    """자연어 입력 → 선수 검색 결과 (원스텝)"""
    filters = parse_query(text)
    print(f"[파싱 결과] {json.dumps(filters, ensure_ascii=False, indent=2)}")
    return search_players(filters)
