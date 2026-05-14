"""
Data pipeline: FBref + StatsBomb → SQLite
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "scout.db"


# ── FBref 수집 ────────────────────────────────────────────────

def fetch_fbref_players(season: str = "2324") -> pd.DataFrame:
    """
    soccerdata로 FBref 시즌 스탯 수집.
    season: '2324' = 2023-24 시즌
    리그: Big 5 유럽 리그
    """
    try:
        import soccerdata as sd
    except ImportError:
        raise ImportError("pip install soccerdata 먼저 실행해주세요")

    leagues = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]
    frames = []

    for league in leagues:
        try:
            fbref = sd.FBref(leagues=league, seasons=season)
            stats = fbref.read_player_season_stats(stat_type="standard")
            stats = stats.reset_index()
            stats = _flatten_columns(stats)
            stats = stats.loc[:, ~stats.columns.duplicated()]
            stats["league"] = league
            frames.append(stats)
            print(f"[FBref] {league} 완료: {len(stats)}명")
        except Exception as e:
            print(f"[FBref] {league} 실패: {e}")

    if not frames:
        raise RuntimeError("FBref 데이터 수집 실패 — 네트워크 또는 soccerdata 버전 확인")

    df = pd.concat(frames, ignore_index=True)
    return df


def fetch_fbref_stat(stat_type: str, season: str = "2324") -> pd.DataFrame:
    """지정된 stat_type의 FBref 데이터 수집. 지원 타입: standard, keeper, shooting, playing_time, misc"""
    try:
        import soccerdata as sd
    except ImportError:
        raise ImportError("pip install soccerdata 먼저 실행해주세요")

    leagues = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]
    frames = []

    for league in leagues:
        try:
            fbref = sd.FBref(leagues=league, seasons=season)
            stats = fbref.read_player_season_stats(stat_type=stat_type)
            stats = stats.reset_index()
            stats = _flatten_columns(stats)
            stats = stats.loc[:, ~stats.columns.duplicated()]
            stats["league"] = league
            frames.append(stats)
            print(f"[FBref {stat_type}] {league} 완료: {len(stats)}명")
        except Exception as e:
            print(f"[FBref {stat_type}] {league} 실패: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── StatsBomb 수집 ────────────────────────────────────────────

def fetch_statsbomb_players() -> pd.DataFrame:
    """
    StatsBomb 공개 데이터에서 선수별 이벤트 집계 스탯 추출.
    무료 공개 경기 기준 (La Liga Messi era, WSL 등)
    """
    try:
        from statsbombpy import sb
    except ImportError:
        raise ImportError("pip install statsbombpy 먼저 실행해주세요")

    competitions = sb.competitions()
    # 공개 데이터 중 La Liga (competition_id=11) 사용
    la_liga = competitions[
        (competitions["competition_id"] == 11) &
        (competitions["season_id"] == 90)  # 2020-21
    ]

    if la_liga.empty:
        available = competitions[["competition_id", "competition_name", "season_id", "season_name"]].head(10)
        print("[StatsBomb] La Liga 2020-21 없음. 사용 가능한 데이터:")
        print(available.to_string())
        return pd.DataFrame()

    matches = sb.matches(competition_id=11, season_id=90)
    print(f"[StatsBomb] La Liga 경기 수: {len(matches)}")

    records = []
    for _, match in matches.iterrows():
        try:
            events = sb.events(match_id=match["match_id"])
            player_stats = _aggregate_events(events, match["match_id"])
            records.extend(player_stats)
        except Exception as e:
            print(f"[StatsBomb] match {match['match_id']} 실패: {e}")

    return pd.DataFrame(records) if records else pd.DataFrame()


def _aggregate_events(events: pd.DataFrame, match_id: int) -> list[dict]:
    """경기 이벤트 → 선수별 집계"""
    if events.empty:
        return []

    players = events[events["player"].notna()]["player"].unique()
    records = []

    for player in players:
        p_events = events[events["player"] == player]
        team = p_events["team"].iloc[0] if "team" in p_events.columns else None
        position = p_events["position"].dropna().iloc[0] if "position" in p_events.columns and not p_events["position"].dropna().empty else None

        record = {
            "match_id": match_id,
            "player": player,
            "team": team,
            "position": str(position) if position else None,
            "shots": len(p_events[p_events["type"] == "Shot"]),
            "passes": len(p_events[p_events["type"] == "Pass"]),
            "pressures": len(p_events[p_events["type"] == "Pressure"]),
            "dribbles": len(p_events[p_events["type"] == "Dribble"]),
            "tackles": len(p_events[p_events["type"] == "Tackle"]),
            "interceptions": len(p_events[p_events["type"] == "Interception"]),
        }
        records.append(record)

    return records


# ── DB 저장 ───────────────────────────────────────────────────

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """FBref 멀티레벨 컬럼 → 단일 문자열 컬럼명으로 평탄화"""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            parts = [str(p).strip() for p in col if str(p).strip() and str(p) != "''"]
            new_cols.append("_".join(parts) if parts else "unnamed")
        df.columns = new_cols
    else:
        df.columns = [str(c) for c in df.columns]

    # 공백, 특수문자 정리
    df.columns = (
        pd.Index(df.columns)
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def save_to_db(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    """DataFrame → SQLite 저장"""
    if df.empty:
        print(f"[DB] {table} 저장 스킵 (빈 DataFrame)")
        return

    df = _flatten_columns(df)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table, conn, if_exists=if_exists, index=False)
    conn.close()
    print(f"[DB] {table} 저장 완료: {len(df)}행")


def load_from_db(table: str) -> pd.DataFrame:
    """SQLite → DataFrame"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


# ── 전처리 ────────────────────────────────────────────────────

def preprocess_players(df: pd.DataFrame) -> pd.DataFrame:
    """기본 전처리: 타입 정리, 결측값 처리, per90 계산"""
    if df.empty:
        return df

    df = df.copy()

    # 숫자형 컬럼 변환
    numeric_cols = df.select_dtypes(include=["object"]).columns
    for col in numeric_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > len(df) * 0.3:
            df[col] = converted

    # 출전 시간 기준 per90 계산
    if "minutes_played" in df.columns and df["minutes_played"].notna().any():
        minutes = pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0)
        for col in ["goals", "assists", "shots", "pressures", "tackles"]:
            if col in df.columns:
                df[f"{col}_per90"] = np.where(
                    minutes > 0,
                    pd.to_numeric(df[col], errors="coerce") / minutes * 90,
                    np.nan
                )

    df = df.fillna(0)
    return df


# ── 시즌별 수집 ───────────────────────────────────────────────

# 시즌 가중치: 최신일수록 높게 (25-26은 시즌 진행 중이라 약간 낮게)
SEASON_WEIGHTS = {
    "2223": 0.15,
    "2324": 0.25,
    "2425": 0.35,
    "2526": 0.25,
}

def run_season(season: str) -> None:
    """단일 시즌 FBref 수집 → players_raw 에 append"""
    print(f"\n  [{season}] 수집 중...")

    try:
        standard = fetch_fbref_players(season)
        standard = preprocess_players(standard)
        standard["season"] = season
    except Exception as e:
        print(f"  [{season}] standard 실패: {e}")
        return

    merged = standard.copy()
    for stat_type in ["shooting", "playing_time", "misc"]:
        try:
            df = fetch_fbref_stat(stat_type, season)
            df = preprocess_players(df)
            df["season"] = season
            keys = [k for k in ["player", "team", "league", "season"]
                    if k in df.columns and k in merged.columns]
            if keys:
                merged = merged.merge(df, on=keys, how="left", suffixes=("", f"_{stat_type}"))
        except Exception as e:
            print(f"  [{season}] {stat_type} 실패: {e}")

    merged = merged.loc[:, ~merged.columns.duplicated()]
    conn = sqlite3.connect(DB_PATH)
    merged.to_sql("players_raw", conn, if_exists="append", index=False)
    conn.close()
    print(f"  [{season}] 저장 완료: {len(merged)}행")


def build_weighted_master() -> None:
    """
    멀티시즌 가중 평균 → players_master 생성
    - 시즌별 가중치 적용 (최신 시즌 우선)
    - 최소 출전시간 500분 이상 시즌만 포함
    - 선수별 최신 팀/포지션 기록
    """
    conn = sqlite3.connect(DB_PATH)
    raw = pd.read_sql("SELECT * FROM players_raw", conn)
    conn.close()

    if raw.empty:
        print("[통합] 데이터 없음")
        return

    # 출전시간 컬럼 통일
    min_col = next((c for c in raw.columns if "min" in c and "90" not in c
                    and "nation" not in c), None)
    raw["minutes_clean"] = pd.to_numeric(raw[min_col], errors="coerce").fillna(0) if min_col else 0

    # 최소 출전 필터
    raw = raw[raw["minutes_clean"] >= 500].copy()

    # 가중치 컬럼
    raw["weight"] = raw["season"].map(SEASON_WEIGHTS).fillna(0.1)

    id_cols = {"player", "team", "league", "season", "pos", "nation",
               "age", "born", "weight", "minutes_clean"}
    stat_cols = [c for c in raw.columns
                 if c not in id_cols
                 and pd.to_numeric(raw[c], errors="coerce").notna().sum() > 0]

    records = []
    for player, grp in raw.groupby("player"):
        total_w = grp["weight"].sum()
        if total_w == 0:
            continue

        latest = grp.sort_values("season").iloc[-1]
        rec = {
            "player": player,
            "team": latest.get("team", ""),
            "league": latest.get("league", ""),
            "pos": latest.get("pos", ""),
            "nation": latest.get("nation", ""),
            "age": latest.get("age", 0),
            "seasons_count": int(grp["season"].nunique()),
            "latest_season": grp["season"].max(),
        }

        for col in stat_cols:
            vals = pd.to_numeric(grp[col], errors="coerce")
            valid = vals.notna()
            if valid.sum() > 0:
                w = grp.loc[valid, "weight"]
                rec[col] = round(float((vals[valid] * w).sum() / w.sum()), 4)

        records.append(rec)

    master = pd.DataFrame(records)
    conn = sqlite3.connect(DB_PATH)
    master.to_sql("players_master", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[통합] players_master 완성: {len(master)}명, {len(master.columns)}컬럼")


# ── 메인 실행 ─────────────────────────────────────────────────

SEASONS = ["2223", "2324", "2425", "2526"]

def run_pipeline(seasons: list[str] = None) -> None:
    if seasons is None:
        seasons = SEASONS

    print("=" * 50)
    print(f"멀티시즌 파이프라인 시작: {seasons}")
    print("=" * 50)

    # players_raw 초기화
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS players_raw")
    conn.commit()
    conn.close()

    for season in seasons:
        run_season(season)

    print("\n[통합] 가중 평균 마스터 테이블 생성 중...")
    build_weighted_master()

    print("\n[StatsBomb] 이벤트 데이터 수집 중...")
    try:
        sb = fetch_statsbomb_players()
        save_to_db(sb, "statsbomb_events")
    except Exception as e:
        print(f"StatsBomb 실패: {e}")

    print("\n파이프라인 완료!")
    print(f"DB 경로: {DB_PATH}")


def build_master_table() -> None:
    """여러 스탯 테이블 → 선수별 통합 테이블"""
    conn = sqlite3.connect(DB_PATH)

    tables = [t[0] for t in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if "players_standard" not in tables:
        print("[통합] players_standard 없음 — 스킵")
        conn.close()
        return

    standard = pd.read_sql("SELECT * FROM players_standard", conn)

    merge_keys = ["player", "league"]
    master = standard.copy()

    for table in ["players_shooting", "players_playing_time", "players_misc"]:
        if table in tables:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            shared_keys = [k for k in merge_keys if k in df.columns and k in master.columns]
            if shared_keys:
                suffix = f"_{table.split('_')[1]}"
                master = master.merge(df, on=shared_keys, how="left", suffixes=("", suffix))

    master.to_sql("players_master", conn, if_exists="replace", index=False)
    print(f"[통합] players_master 생성 완료: {len(master)}행, {len(master.columns)}컬럼")
    conn.close()


if __name__ == "__main__":
    run_pipeline()
