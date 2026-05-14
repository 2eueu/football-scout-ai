"""
Value Scouting — 저평가 선수 발굴
1. Transfermarkt 이적료 크롤링
2. FBref 가중 평균 스탯 피처 엔지니어링
3. XGBoost 회귀로 "퍼포먼스 기반 적정 몸값" 예측
4. 실제 몸값 vs 예측 몸값 → 저평가/고평가 분류
"""

import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

DB_PATH = Path(__file__).parent.parent / "scout.db"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

TM_LEAGUES = {
    "ENG-Premier League": ("GB1", "premier-league"),
    "ESP-La Liga":        ("ES1", "laliga"),
    "GER-Bundesliga":     ("L1",  "bundesliga"),
    "ITA-Serie A":        ("IT1", "serie-a"),
    "FRA-Ligue 1":        ("FR1", "ligue-1"),
}


# ── Transfermarkt 크롤링 ──────────────────────────────────────

def _scrape_tm_league(league_id: str, league_slug: str) -> list[dict]:
    url = f"https://www.transfermarkt.com/{league_slug}/marktwerte/wettbewerb/{league_id}"
    records = []
    page = 1

    while page <= 20:
        try:
            r = requests.get(
                url, headers=HEADERS,
                params={"ajax": "yw1", "page": page},
                timeout=15
            )
            if r.status_code != 200:
                break

            soup = BeautifulSoup(r.text, "lxml")
            rows = soup.select("table.items tbody tr:not(.spacer):not(.bg_blau_20)")
            if not rows:
                break

            for row in rows:
                try:
                    name_el = row.select_one("td.hauptlink a")
                    val_el  = row.select_one("td.rechts.hauptlink")
                    pos_el  = row.select_one("td:nth-child(2) tr:nth-child(2) td")
                    if not name_el or not val_el:
                        continue
                    records.append({
                        "player_tm": name_el.text.strip(),
                        "market_value_eur": _parse_value(val_el.text.strip()),
                        "position_tm": pos_el.text.strip() if pos_el else None,
                    })
                except Exception:
                    continue

            page += 1
            time.sleep(1.2)

        except Exception as e:
            print(f"  [TM] page {page} 실패: {e}")
            break

    return records


def _parse_value(text: str) -> float:
    """'€45.00m' → 45_000_000, '€500k' → 500_000"""
    text = text.replace(",", ".").lower().strip()
    m = re.search(r"[\d.]+", text)
    if not m:
        return 0.0
    val = float(m.group())
    if "m" in text:
        return val * 1_000_000
    if "k" in text:
        return val * 1_000
    return val


def fetch_market_values() -> pd.DataFrame:
    all_records = []
    for fbref_name, (league_id, slug) in TM_LEAGUES.items():
        print(f"  [TM] {fbref_name} 수집 중...")
        rows = _scrape_tm_league(league_id, slug)
        for r in rows:
            r["league"] = fbref_name
        all_records.extend(rows)
        print(f"  [TM] {fbref_name} 완료: {len(rows)}명")
        time.sleep(2)

    df = pd.DataFrame(all_records)
    if not df.empty:
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("market_values", conn, if_exists="replace", index=False)
        conn.close()
        print(f"\n[DB] market_values 저장: {len(df)}명")
    return df


# ── 피처 엔지니어링 ───────────────────────────────────────────

BASE_FEATURES = [
    "age", "seasons_count",
    "playing_time_min", "playing_time_90s",
    "per_90_minutes_gls", "per_90_minutes_ast", "per_90_minutes_g_a",
    "performance_gls", "performance_ast",
    "standard_sh_90", "standard_sot_90",
    "performance_tklw", "performance_int",
    "performance_fld", "performance_fls",
    "performance_crdy",
    "xg_p90", "npxg_p90", "xa_p90",
]


def _pos_group(pos: str) -> str:
    if not isinstance(pos, str):
        return "MF"
    pos = pos.upper()
    if "GK" in pos: return "GK"
    if "FW" in pos: return "FW"
    if "DF" in pos: return "DF"
    return "MF"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pos_group"] = df["pos"].apply(_pos_group)
    # FBref age 형식: "23-338" → 23
    df["age"] = df["age"].astype(str).str.split("-").str[0]
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(25)

    # 잠재력 프리미엄: 어릴수록 높게
    df["age_factor"] = np.where(df["age"] <= 23, 1.5,
                        np.where(df["age"] <= 27, 1.2,
                        np.where(df["age"] <= 30, 1.0, 0.75)))

    # 리그 티어 가중치
    tier = {
        "ENG-Premier League": 1.3, "ESP-La Liga": 1.2,
        "GER-Bundesliga": 1.1, "ITA-Serie A": 1.05, "FRA-Ligue 1": 1.0,
    }
    df["league_tier"] = df["league"].map(tier).fillna(1.0)

    for col in BASE_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ── 모델 학습 & 예측 ──────────────────────────────────────────

def train_value_model(df: pd.DataFrame):
    try:
        from xgboost import XGBRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        raise ImportError("pip install xgboost scikit-learn")

    train_df = df[df["market_value_eur"] > 0].copy()
    if len(train_df) < 50:
        raise RuntimeError(f"학습 데이터 부족: {len(train_df)}명")

    train_df["log_value"] = np.log1p(train_df["market_value_eur"])

    feat_cols = [c for c in BASE_FEATURES + ["age_factor", "league_tier"]
                 if c in train_df.columns]
    pos_dummies = pd.get_dummies(train_df["pos_group"], prefix="pos")
    X = pd.concat([train_df[feat_cols], pos_dummies], axis=1).fillna(0)
    y = train_df["log_value"]

    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(X, y)

    cv = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    print(f"[모델] 5-fold CV RMSE (log): {-cv.mean():.4f} ± {cv.std():.4f}")

    return model, X.columns.tolist(), pos_dummies.columns.tolist()


def predict_values(df: pd.DataFrame, model, feature_cols, pos_cols) -> pd.DataFrame:
    pos_dummies = pd.get_dummies(df["pos_group"], prefix="pos")
    for col in pos_cols:
        if col not in pos_dummies.columns:
            pos_dummies[col] = 0

    base_feats = [c for c in feature_cols if not c.startswith("pos_")]
    X_all = pd.concat([df[base_feats].fillna(0), pos_dummies[pos_cols]], axis=1)

    df = df.copy()
    df["predicted_value_eur"] = np.expm1(model.predict(X_all))

    has_val = df["market_value_eur"] > 0
    df["undervalue_score"] = np.where(
        has_val,
        (df["predicted_value_eur"] - df["market_value_eur"]) / df["market_value_eur"] * 100,
        np.nan,
    )
    return df


# ── 메인 파이프라인 ───────────────────────────────────────────

def run_value_scouting(use_cached_tm: bool = False) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    master = pd.read_sql("SELECT * FROM players_master", conn)
    tables = [t[0] for t in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    conn.close()

    if use_cached_tm and "market_values" in tables:
        conn = sqlite3.connect(DB_PATH)
        tm = pd.read_sql("SELECT * FROM market_values", conn)
        conn.close()
        print(f"[TM] 캐시 사용: {len(tm)}명")
    else:
        print("[TM] Transfermarkt 크롤링 시작...")
        tm = fetch_market_values()

    if tm.empty:
        raise RuntimeError("Transfermarkt 데이터 없음")

    tm["player_key"] = tm["player_tm"].str.lower().str.strip()
    master["player_key"] = master["player"].str.lower().str.strip()

    merged = master.merge(
        tm[["player_key", "market_value_eur"]],
        on="player_key", how="left",
    )
    merged["market_value_eur"] = pd.to_numeric(
        merged["market_value_eur"], errors="coerce"
    ).fillna(0)

    matched = (merged["market_value_eur"] > 0).sum()
    print(f"[매칭] 시장가치 매칭 선수: {matched}명 / {len(merged)}명")

    df = build_features(merged)

    print("[모델] XGBoost 학습 중...")
    model, feat_cols, pos_cols = train_value_model(df)

    result = predict_values(df, model, feat_cols, pos_cols)

    save_cols = ["player", "team", "league", "pos", "age",
                 "market_value_eur", "predicted_value_eur", "undervalue_score",
                 "per_90_minutes_gls", "per_90_minutes_ast",
                 "performance_tklw", "performance_int",
                 "playing_time_min", "seasons_count", "latest_season"]
    save_cols = [c for c in save_cols if c in result.columns]

    conn = sqlite3.connect(DB_PATH)
    result[save_cols].to_sql("value_scouting", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[DB] value_scouting 저장: {len(result)}명")

    return result[save_cols]


def get_undervalued(
    max_value_eur: float = None,
    position: str = None,
    league: str = None,
    age_max: int = None,
    top_n: int = 20,
) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM value_scouting WHERE undervalue_score IS NOT NULL AND market_value_eur > 0",
        conn
    )
    conn.close()

    if max_value_eur:
        df = df[df["market_value_eur"] <= max_value_eur]
    if position:
        df = df[df["pos"].str.upper().str.contains(position.upper(), na=False)]
    if league:
        df = df[df["league"] == league]
    if age_max:
        df = df[pd.to_numeric(df["age"], errors="coerce") <= age_max]

    df = df.drop_duplicates(subset=["player"]).sort_values("undervalue_score", ascending=False).head(top_n)
    df["market_value_m"]    = (df["market_value_eur"] / 1e6).round(1)
    df["predicted_value_m"] = (df["predicted_value_eur"] / 1e6).round(1)
    df["undervalue_score"]  = df["undervalue_score"].round(1)

    out_cols = [
        "player", "team", "league", "pos", "age",
        "market_value_m", "predicted_value_m", "undervalue_score",
        "per_90_minutes_gls", "per_90_minutes_ast",
        "xg_p90", "npxg_p90", "xa_p90",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols].rename(columns={
        "market_value_m":     "Actual (M€)",
        "predicted_value_m":  "Predicted (M€)",
        "undervalue_score":   "Undervalue (%)",
        "per_90_minutes_gls": "Goals/90",
        "per_90_minutes_ast": "Assists/90",
        "xg_p90":             "xG/90",
        "npxg_p90":           "npxG/90",
        "xa_p90":             "xA/90",
    })


SIMILARITY_FEATURES = [
    "per_90_minutes_gls", "per_90_minutes_ast", "per_90_minutes_g_a",
    "standard_sh_90", "performance_tklw", "performance_int",
    "playing_time_min", "age_factor", "league_tier",
    "xg_p90", "npxg_p90", "xa_p90",
]


def get_similar_players(
    player_name: str,
    max_value_eur: float = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """코사인 유사도 기반 유사 선수 추천 — 더 저렴한 선수 우선."""
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler

    conn = sqlite3.connect(DB_PATH)
    master = pd.read_sql("SELECT * FROM players_master", conn)
    try:
        vs = pd.read_sql(
            """SELECT player, market_value_eur, predicted_value_eur, undervalue_score
               FROM value_scouting
               WHERE market_value_eur > 0""",
            conn,
        )
    except Exception:
        vs = pd.DataFrame()
    conn.close()

    # deduplicate before any merge — value_scouting can have multiple rows per player
    master = master.drop_duplicates(subset=["player"])
    if not vs.empty:
        vs = vs.drop_duplicates(subset=["player"])

    df = build_features(master)
    if not vs.empty:
        df = df.merge(vs, on="player", how="left")
    else:
        df["market_value_eur"] = 0
        df["predicted_value_eur"] = np.nan
        df["undervalue_score"] = np.nan

    feat_cols = [c for c in SIMILARITY_FEATURES if c in df.columns]
    X = df[feat_cols].fillna(0).values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    target_idx = df[df["player"].str.lower() == player_name.lower()].index
    if target_idx.empty:
        target_idx = df[df["player"].str.contains(player_name, case=False, na=False)].index
    if target_idx.empty:
        return pd.DataFrame()

    idx = target_idx[0]
    target_vec = X_scaled[idx].reshape(1, -1)
    sims = cosine_similarity(target_vec, X_scaled)[0]
    df["similarity"] = sims

    result = df[df.index != idx].copy()

    target_val = df.loc[idx, "market_value_eur"] if "market_value_eur" in df.columns else 0
    if max_value_eur:
        result = result[result["market_value_eur"] <= max_value_eur]
    elif target_val > 0:
        result = result[result["market_value_eur"] <= target_val * 0.85]

    result = result[result["market_value_eur"] > 0]
    result = result.drop_duplicates(subset=["player"])
    result = result.sort_values("similarity", ascending=False).head(top_n)

    result["similarity_pct"] = (result["similarity"] * 100).round(1)
    result["market_value_m"] = (result["market_value_eur"] / 1e6).round(1)
    result["predicted_value_m"] = (result.get("predicted_value_eur", 0) / 1e6).round(1)

    cols = ["player", "team", "league", "pos", "age",
            "market_value_m", "predicted_value_m", "undervalue_score",
            "similarity_pct", "per_90_minutes_gls", "per_90_minutes_ast",
            "xg_p90", "npxg_p90", "xa_p90"]
    cols = [c for c in cols if c in result.columns]
    return result[cols].rename(columns={
        "market_value_m":     "Actual (M€)",
        "predicted_value_m":  "Predicted (M€)",
        "undervalue_score":   "Undervalue (%)",
        "similarity_pct":     "Similarity (%)",
        "per_90_minutes_gls": "Goals/90",
        "per_90_minutes_ast": "Assists/90",
        "xg_p90":             "xG/90",
        "npxg_p90":           "npxG/90",
        "xa_p90":             "xA/90",
    })


if __name__ == "__main__":
    print("=== Value Scouting 파이프라인 ===\n")
    run_value_scouting(use_cached_tm=False)

    print("\n[결과] €20M 이하 저평가 공격수 TOP 10:")
    top = get_undervalued(max_value_eur=20_000_000, position="FW", top_n=10)
    print(top.to_string(index=False))
