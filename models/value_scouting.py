"""
Value Scouting — 저평가 선수 발굴
1. Transfermarkt 이적료 크롤링
2. FBref 가중 평균 스탯 피처 엔지니어링
3. XGBoost 회귀로 "퍼포먼스 기반 적정 몸값" 예측
4. 실제 몸값 vs 예측 몸값 → 저평가/고평가 분류
5. 나이 통제 잔차 (performance_premium), 리그 보정 계수
"""

import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

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

def _get_league_teams(league_id: str, league_slug: str) -> dict:
    """Return {team_id: team_slug} for all teams in the league."""
    import requests
    from bs4 import BeautifulSoup
    url = f"https://www.transfermarkt.com/{league_slug}/startseite/wettbewerb/{league_id}/saison_id/2025"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return {}
        soup = BeautifulSoup(r.text, "lxml")
        teams = {}
        for a in soup.find_all("a", href=re.compile(r"/startseite/verein/\d+")):
            href = a.get("href", "")
            m = re.match(r"^/([^/]+)/startseite/verein/(\d+)", href)
            if m:
                teams[m.group(2)] = m.group(1)  # {team_id: slug}
        return teams
    except Exception as e:
        print(f"  [TM] league teams error for {league_slug}: {e}")
        return {}


def _scrape_team_squad(team_id: str, team_slug: str, league_name: str) -> list[dict]:
    """Scrape market values for all players in a team squad."""
    import requests
    from bs4 import BeautifulSoup
    url = f"https://www.transfermarkt.com/{team_slug}/kader/verein/{team_id}/saison_id/2025"
    records = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        rows = soup.select("table.items tr.odd, table.items tr.even")
        for row in rows:
            try:
                name_el = row.select_one("td.hauptlink a")
                val_el  = row.select_one("td.rechts.hauptlink")
                if not name_el or not val_el:
                    continue
                name = name_el.text.strip()
                val  = _parse_value(val_el.text.strip())
                if val <= 0 or not name:
                    continue
                records.append({
                    "player_tm":       name,
                    "market_value_eur": val,
                    "league":          league_name,
                })
            except Exception:
                continue
    except Exception as e:
        print(f"  [TM] squad error {team_slug}: {e}")
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
    """Scrape market values via squad pages for full coverage (~25 players/team)."""
    all_records = []
    seen_global: set = set()

    for fbref_name, (league_id, slug) in TM_LEAGUES.items():
        print(f"  [TM] {fbref_name} — fetching teams...")
        teams = _get_league_teams(league_id, slug)
        print(f"  [TM] {fbref_name} — {len(teams)} teams found")
        time.sleep(1)

        league_count = 0
        for team_id, team_slug in teams.items():
            rows = _scrape_team_squad(team_id, team_slug, fbref_name)
            for row in rows:
                key = (row["player_tm"].lower().strip(), fbref_name)
                if key in seen_global:
                    continue
                seen_global.add(key)
                all_records.append(row)
                league_count += 1
            time.sleep(1.2)

        print(f"  [TM] {fbref_name} 완료: {league_count}명")
        time.sleep(2)

    df = pd.DataFrame(all_records)
    if not df.empty:
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("market_values", conn, if_exists="replace", index=False)
        conn.close()
        print(f"\n[DB] market_values 저장: {len(df)}명 ({df['player_tm'].nunique()} unique)")
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

POS_FEATURES = {
    "FW": BASE_FEATURES,
    "MF": BASE_FEATURES,
    "DF": [
        "age", "seasons_count", "playing_time_min", "playing_time_90s",
        "performance_tklw", "performance_int", "performance_fld", "performance_fls",
        "performance_crdy", "per_90_minutes_g_a",
        "standard_sh_90", "xg_p90", "xa_p90",
    ],
    "GK": ["age", "seasons_count", "playing_time_min",
           "gk_save_pct", "gk_ga_p90", "gk_cs_pct", "gk_saves_p90"],
}

# Position-agnostic fallback (used by report.py)
RADAR_STATS = {
    "Goals/90":      "per_90_minutes_gls",
    "xG/90":         "xg_p90",
    "Assists/90":    "per_90_minutes_ast",
    "xA/90":         "xa_p90",
    "Shots/90":      "standard_sh_90",
    "G+A/90":        "per_90_minutes_g_a",
    "Tackles":       "performance_tklw",
    "Interceptions": "performance_int",
}

# Position-specific radar definitions
POS_RADAR_STATS = {
    "FW": {
        "Goals/90":    "per_90_minutes_gls",
        "xG/90":       "xg_p90",
        "npxG/90":     "npxg_p90",
        "Shots/90":    "standard_sh_90",
        "SoT/90":      "standard_sot_90",
        "Assists/90":  "per_90_minutes_ast",
        "xA/90":       "xa_p90",
        "G+A/90":      "per_90_minutes_g_a",
    },
    "MF": {
        "xA/90":         "xa_p90",
        "Assists/90":    "per_90_minutes_ast",
        "xG/90":         "xg_p90",
        "Goals/90":      "per_90_minutes_gls",
        "Tackles Won":   "performance_tklw",
        "Interceptions": "performance_int",
        "Fouls Drawn":   "performance_fld",
        "G+A/90":        "per_90_minutes_g_a",
    },
    "DF": {
        "Tackles Won":   "performance_tklw",
        "Interceptions": "performance_int",
        "Fouls Drawn":   "performance_fld",
        "xG/90":         "xg_p90",
        "xA/90":         "xa_p90",
        "Goals/90":      "per_90_minutes_gls",
        "Assists/90":    "per_90_minutes_ast",
        "G+A/90":        "per_90_minutes_g_a",
    },
    "GK": {
        "Save %":        "gk_save_pct",
        "GA/90":         "gk_ga_p90",
        "Clean Sheet %": "gk_cs_pct",
        "PK Save %":     "gk_pksave_pct",
        "Saves/90":      "gk_saves_p90",
        "Win %":         "gk_win_pct",
    },
}

# Stats where a lower value is better — percentile is inverted (100 - rank)
INVERTED_RADAR_STATS = {"gk_ga_p90"}


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


# ── 모델 학습 & 예측 (position-specific) ─────────────────────

def _train_one_model(train_df: pd.DataFrame, feat_cols: list):
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score

    train_df = train_df.copy()
    # age² captures the non-linear value peak around mid-20s
    if "age" in train_df.columns and "age_sq" not in feat_cols:
        train_df["age_sq"] = train_df["age"] ** 2
        feat_cols = feat_cols + ["age_sq"]

    train_df["log_value"] = np.log1p(train_df["market_value_eur"])
    X = train_df[feat_cols].fillna(0)
    y = train_df["log_value"]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,   # prevents overfitting on rare high-value players
        reg_alpha=0.1,        # L1
        reg_lambda=2.0,       # L2
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    rmse = float("nan")
    if len(train_df) >= 10:
        cv = cross_val_score(model, X, y, cv=min(5, len(train_df) // 5),
                             scoring="neg_root_mean_squared_error")
        rmse = float(-cv.mean())
        print(f"  CV RMSE: {rmse:.4f} ± {cv.std():.4f}  (n={len(train_df)})")

    return model, feat_cols, rmse


# Structural features: age/league/experience only — no performance stats
STRUCTURAL_FEATURES = ["age", "age_sq", "age_factor", "league_tier", "seasons_count", "playing_time_min"]


def train_value_model(df: pd.DataFrame):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("pip install xgboost scikit-learn")

    train_df = df[df["market_value_eur"] > 0].copy()
    if len(train_df) < 30:
        raise RuntimeError(f"Training data too small: {len(train_df)}")

    models = {}
    structural_models = {}
    model_rmse = {}

    for pos, feat_list in POS_FEATURES.items():
        subset = train_df[train_df["pos_group"] == pos]
        feat_cols = [c for c in feat_list + ["age_factor", "league_tier"]
                     if c in subset.columns]
        if len(subset) < 10:
            continue
        print(f"[Model {pos}] training on {len(subset)} players...")
        m, fc, rmse = _train_one_model(subset, feat_cols)
        models[pos] = (m, fc)
        model_rmse[pos] = {"rmse": round(rmse, 4), "n_train": len(subset)}

        struct_cols = [c for c in STRUCTURAL_FEATURES if c in subset.columns]
        sm, sfc, _ = _train_one_model(subset, struct_cols)
        structural_models[pos] = (sm, sfc)

    return models, structural_models, model_rmse


def predict_values(df: pd.DataFrame, models: dict,
                   structural_models: dict = None) -> pd.DataFrame:
    df = df.copy()
    if "age" in df.columns:
        df["age_sq"] = pd.to_numeric(df["age"], errors="coerce").fillna(25) ** 2
    df["predicted_value_eur"] = np.nan
    df["structural_value_eur"] = np.nan

    for pos, (model, feat_cols) in models.items():
        mask = df["pos_group"] == pos
        if not mask.any():
            continue
        present = [c for c in feat_cols if c in df.columns]
        X = df.loc[mask, present].fillna(0)
        df.loc[mask, "predicted_value_eur"] = np.expm1(model.predict(X))

        # Structural prediction (age/league baseline)
        if structural_models and pos in structural_models:
            sm, sc = structural_models[pos]
            sc_present = [c for c in sc if c in df.columns]
            Xs = df.loc[mask, sc_present].fillna(0)
            df.loc[mask, "structural_value_eur"] = np.expm1(sm.predict(Xs))

    # Fallback for unmatched positions
    fallback_mask = df["predicted_value_eur"].isna()
    if fallback_mask.any() and not df["predicted_value_eur"].dropna().empty:
        df.loc[fallback_mask, "predicted_value_eur"] = df["predicted_value_eur"].median()

    # Performance premium: how much skill adds ABOVE age/league baseline
    # Positive = player outperforms their age/league profile
    sv = df["structural_value_eur"].replace(0, np.nan)
    df["performance_premium"] = np.where(
        sv.notna() & (sv > 0),
        (df["predicted_value_eur"] - sv) / sv * 100,
        np.nan,
    )

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

    import unicodedata

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s).lower().strip())
        return "".join(c for c in s if not unicodedata.combining(c))

    tm["player_key"] = tm["player_tm"].apply(_norm)
    master["player_key"] = master["player"].apply(_norm)

    # Exact (normalised) match first
    merged = master.merge(
        tm[["player_key", "market_value_eur"]].drop_duplicates("player_key"),
        on="player_key", how="left",
    )

    # Fuzzy fallback for unmatched players
    unmatched_mask = merged["market_value_eur"].isna()
    unmatched_count = unmatched_mask.sum()
    if unmatched_count > 0:
        try:
            from rapidfuzz import process, fuzz
            tm_keys = tm["player_key"].unique().tolist()
            tm_val_map = tm.drop_duplicates("player_key").set_index("player_key")["market_value_eur"].to_dict()
            unmatched_keys = merged.loc[unmatched_mask, "player_key"].tolist()
            fuzzy_vals = []
            for key in unmatched_keys:
                match, score, _ = process.extractOne(key, tm_keys, scorer=fuzz.token_sort_ratio)
                fuzzy_vals.append(tm_val_map.get(match, 0) if score >= 88 else 0)
            merged.loc[unmatched_mask, "market_value_eur"] = fuzzy_vals
            fuzzy_matched = sum(v > 0 for v in fuzzy_vals)
            print(f"[매칭] 퍼지 매칭 추가: {fuzzy_matched}명")
        except ImportError:
            pass

    merged["market_value_eur"] = pd.to_numeric(
        merged["market_value_eur"], errors="coerce"
    ).fillna(0)

    matched = (merged["market_value_eur"] > 0).sum()
    print(f"[매칭] 시장가치 매칭 선수: {matched}명 / {len(merged)}명")

    df = build_features(merged)

    print("[Model] Training position-specific XGBoost models...")
    models, structural_models, model_rmse = train_value_model(df)

    result = predict_values(df, models, structural_models)

    # Save model metrics (CV RMSE per position)
    if model_rmse:
        metrics_rows = [{"pos_group": pos, **v} for pos, v in model_rmse.items()]
        conn = sqlite3.connect(DB_PATH)
        pd.DataFrame(metrics_rows).to_sql("model_metrics", conn, if_exists="replace", index=False)
        conn.close()

    # Save feature importances
    fi_records = []
    for pos, (model, feat_cols) in models.items():
        importances = model.feature_importances_
        for feat, imp in sorted(zip(feat_cols, importances), key=lambda x: -x[1]):
            fi_records.append({"pos_group": pos, "feature": feat, "importance": float(imp)})
    if fi_records:
        conn = sqlite3.connect(DB_PATH)
        pd.DataFrame(fi_records).to_sql("feature_importance", conn, if_exists="replace", index=False)
        conn.close()

    save_cols = ["player", "team", "league", "pos", "age",
                 "market_value_eur", "predicted_value_eur", "undervalue_score",
                 "structural_value_eur", "performance_premium",
                 "per_90_minutes_gls", "per_90_minutes_ast",
                 "xg_p90", "npxg_p90", "xa_p90",
                 "performance_tklw", "performance_int",
                 "playing_time_min", "seasons_count", "latest_season",
                 "gk_save_pct", "gk_ga_p90", "gk_cs_pct",
                 "gk_pksave_pct", "gk_saves_p90", "gk_win_pct"]
    save_cols = [c for c in save_cols if c in result.columns]

    conn = sqlite3.connect(DB_PATH)
    result[save_cols].to_sql("value_scouting", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[DB] value_scouting saved: {len(result)} players")

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


# ── 팀 피트 스코어링 ──────────────────────────────────────────

TEAM_FIT_FEATURES = [
    "per_90_minutes_gls", "per_90_minutes_ast", "per_90_minutes_g_a",
    "xg_p90", "xa_p90", "standard_sh_90", "standard_sot_90",
    "performance_tklw", "performance_int", "performance_fld",
]


def get_team_fit_players(
    target_team: str,
    position: str = None,
    max_value_eur: float = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Find players whose stats best fit a target team's playing style.
    Team DNA = mean of tactical features for same-position players already on that team.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler

    conn = sqlite3.connect(DB_PATH)
    master = pd.read_sql("SELECT * FROM players_master", conn)
    try:
        vs = pd.read_sql(
            "SELECT player, market_value_eur, undervalue_score FROM value_scouting WHERE market_value_eur > 0",
            conn,
        )
    except Exception:
        vs = pd.DataFrame()
    conn.close()

    master = master.drop_duplicates("player")
    master["pos_group"] = master["pos"].apply(_pos_group)
    if not vs.empty:
        vs = vs.drop_duplicates("player")

    feat_cols = [c for c in TEAM_FIT_FEATURES if c in master.columns]
    for col in feat_cols:
        master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0)

    # Merge market values
    if not vs.empty:
        master = master.merge(vs, on="player", how="left")
    else:
        master["market_value_eur"] = 0
        master["undervalue_score"] = np.nan

    # Filter position if given
    pos_groups = []
    if position:
        p = position.upper()
        if "FW" in p: pos_groups = ["FW"]
        elif "MF" in p: pos_groups = ["MF"]
        elif "DF" in p: pos_groups = ["DF"]
        elif "GK" in p: pos_groups = ["GK"]
    if not pos_groups:
        pos_groups = ["FW", "MF", "DF"]

    results = []
    for pg in pos_groups:
        team_players = master[(master["team"] == target_team) & (master["pos_group"] == pg)]
        if team_players.empty:
            continue

        # Team DNA: mean stats of current players in this position group
        team_dna = team_players[feat_cols].mean().values.reshape(1, -1)

        # Candidates: all players NOT on this team, same position group
        candidates = master[(master["team"] != target_team) & (master["pos_group"] == pg)].copy()
        if max_value_eur:
            candidates = candidates[candidates["market_value_eur"] <= max_value_eur]

        if candidates.empty:
            continue

        X = candidates[feat_cols].values
        scaler = StandardScaler()
        # Fit on all players for this position group for consistent scaling
        all_pos = master[master["pos_group"] == pg][feat_cols].values
        scaler.fit(all_pos)

        X_scaled = scaler.transform(X)
        dna_scaled = scaler.transform(team_dna)

        sims = cosine_similarity(dna_scaled, X_scaled)[0]
        candidates = candidates.copy()
        candidates["fit_score"] = sims
        candidates["fit_pct"] = (sims * 100).round(1)
        results.append(candidates)

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    df = df.drop_duplicates("player")
    df = df.sort_values("fit_score", ascending=False).head(top_n)

    df["market_value_m"] = (df["market_value_eur"].fillna(0) / 1e6).round(1)
    df["age_int"] = df["age"].astype(str).str.split("-").str[0]
    df["age_int"] = pd.to_numeric(df["age_int"], errors="coerce").fillna(0).astype(int)

    out_cols = ["player", "team", "league", "pos", "age_int",
                "market_value_m", "undervalue_score", "fit_pct",
                "per_90_minutes_gls", "per_90_minutes_ast", "xg_p90", "xa_p90"]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols].rename(columns={
        "age_int":            "age",
        "market_value_m":     "Value (M€)",
        "undervalue_score":   "Undervalue (%)",
        "fit_pct":            "Fit Score (%)",
        "per_90_minutes_gls": "Goals/90",
        "per_90_minutes_ast": "Assists/90",
        "xg_p90":             "xG/90",
        "xa_p90":             "xA/90",
    })


def get_all_teams() -> list[str]:
    """Return sorted list of all team names in the DB."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT DISTINCT team FROM players_master WHERE team IS NOT NULL ORDER BY team", conn)
    conn.close()
    return df["team"].tolist()


def get_league_factors() -> pd.DataFrame:
    """Return league scoring difficulty factors for goals/90 (one row per league)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT league, league_mean, global_mean, factor FROM league_factors "
            "WHERE stat = 'per_90_minutes_gls' ORDER BY factor DESC",
            conn,
        )
    except Exception:
        df = pd.DataFrame(columns=["league", "league_mean", "global_mean", "factor"])
    conn.close()
    return df


def get_model_metrics() -> pd.DataFrame:
    """Return CV RMSE per position for the current full model."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM model_metrics ORDER BY pos_group", conn)
    except Exception:
        df = pd.DataFrame(columns=["pos_group", "rmse", "n_train"])
    conn.close()
    return df


def get_age_curve_data(pos_group: str = "FW") -> dict:
    """
    Polynomial age-value curve for a position group.
    Returns scatter points, fitted curve, and peak age.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT player, age, pos, market_value_eur FROM value_scouting "
        "WHERE market_value_eur > 0",
        conn,
    )
    conn.close()

    df["pos_group"] = df["pos"].apply(_pos_group)
    df["age_num"] = pd.to_numeric(
        df["age"].astype(str).str.split("-").str[0], errors="coerce"
    )
    df = df[(df["pos_group"] == pos_group) & df["age_num"].between(16, 40)].copy()

    if len(df) < 20:
        return {"has_data": False}

    # Filter to 19-38 to avoid extreme youth-premium distortion
    df = df[df["age_num"].between(19, 38)].copy()
    if len(df) < 20:
        return {"has_data": False}

    df["value_m"] = df["market_value_eur"] / 1e6

    # Median-smoothed curve: groupby age → median value → rolling smooth
    age_med = df.groupby("age_num")["market_value_eur"].median()
    ages_int = age_med.index.values.astype(float)
    med_vals = age_med.values / 1e6

    # Peak age = age with highest median market value in the data
    peak_age = int(age_med.idxmax())
    peak_age = max(19, min(peak_age, 35))

    # Polynomial fit to median values for smooth curve
    coeffs = np.polyfit(ages_int, np.log1p(age_med.values), 2)
    age_range = np.linspace(19, 38, 100)
    curve_m = np.clip(np.expm1(np.polyval(coeffs, age_range)) / 1e6, 0, None)

    return {
        "has_data":        True,
        "scatter_age":     df["age_num"].tolist(),
        "scatter_value_m": df["value_m"].round(1).tolist(),
        "scatter_player":  df["player"].tolist(),
        "curve_age":       age_range.tolist(),
        "curve_value_m":   curve_m.round(2).tolist(),
        "med_age":         ages_int.tolist(),
        "med_value_m":     [round(float(v), 2) for v in med_vals],
        "peak_age":        peak_age,
        "n_players":       len(df),
        "pos_group":       pos_group,
        "coeffs":          [float(c) for c in coeffs],
    }


def run_backtest_model() -> pd.DataFrame:
    """
    Temporal backtest: train on 2022-23 season stats only, evaluate via 5-fold OOF.
    Compares historical-model accuracy vs full 4-season model on current TM values.
    Saves results to scout.db backtest_results table.
    """
    import unicodedata
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_predict, KFold

    conn = sqlite3.connect(DB_PATH)
    raw_hist = pd.read_sql("SELECT * FROM players_raw WHERE season = '2223'", conn)
    tm        = pd.read_sql("SELECT * FROM market_values", conn)
    try:
        cur_preds = pd.read_sql(
            "SELECT player, predicted_value_eur AS full_pred_eur FROM value_scouting",
            conn,
        ).drop_duplicates("player")
    except Exception:
        cur_preds = pd.DataFrame()
    conn.close()

    if raw_hist.empty or tm.empty:
        return pd.DataFrame()

    def _norm(s):
        s = unicodedata.normalize("NFKD", str(s).lower().strip())
        return "".join(c for c in s if not unicodedata.combining(c))

    tm["player_key"]       = tm["player_tm"].apply(_norm)
    raw_hist["player_key"] = raw_hist["player"].apply(_norm)
    raw_hist = raw_hist.drop_duplicates(subset=["player"]).copy()

    merged = raw_hist.merge(
        tm[["player_key", "market_value_eur"]].drop_duplicates("player_key"),
        on="player_key", how="left",
    )
    merged["market_value_eur"] = pd.to_numeric(
        merged["market_value_eur"], errors="coerce"
    ).fillna(0)

    df_hist = build_features(merged)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    records = []
    for pos, feat_list in POS_FEATURES.items():
        subset = df_hist[
            (df_hist["pos_group"] == pos) & (df_hist["market_value_eur"] > 0)
        ].copy()
        if len(subset) < 20:
            continue

        feat_cols = [c for c in feat_list + ["age_factor", "league_tier"]
                     if c in subset.columns]
        if "age" in subset.columns:
            subset["age_sq"] = subset["age"] ** 2
            if "age_sq" not in feat_cols:
                feat_cols = feat_cols + ["age_sq"]

        subset["log_value"] = np.log1p(subset["market_value_eur"])
        X = subset[feat_cols].fillna(0)
        y = subset["log_value"].values

        model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=2.0, random_state=42, verbosity=0,
        )
        oof_log = cross_val_predict(model, X, y, cv=kf)

        out = subset[["player", "pos", "team", "league", "age",
                       "market_value_eur"]].copy()
        out["hist_pred_eur"] = np.expm1(oof_log).clip(min=100_000)
        records.append(out)

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)

    if not cur_preds.empty:
        result = result.merge(cur_preds.drop_duplicates("player"), on="player", how="left")
    else:
        result["full_pred_eur"] = result["hist_pred_eur"]

    result["hist_error_pct"] = (
        (result["hist_pred_eur"] - result["market_value_eur"])
        / result["market_value_eur"] * 100
    ).round(1)
    if "full_pred_eur" in result.columns:
        full = pd.to_numeric(result["full_pred_eur"], errors="coerce").fillna(0)
        result["full_error_pct"] = (
            (full - result["market_value_eur"]) / result["market_value_eur"] * 100
        ).round(1)

    save_cols = [c for c in ["player", "pos", "team", "league", "age",
                              "market_value_eur", "hist_pred_eur", "hist_error_pct",
                              "full_pred_eur", "full_error_pct"] if c in result.columns]

    conn = sqlite3.connect(DB_PATH)
    result[save_cols].to_sql("backtest_results", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[Backtest] {len(result)} players evaluated and saved")
    return result[save_cols]


def get_feature_importance(pos_group: str) -> pd.DataFrame:
    """Return feature importances for a position model, sorted descending."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT feature, importance FROM feature_importance WHERE pos_group = ? ORDER BY importance DESC",
            conn, params=[pos_group],
        )
    except Exception:
        df = pd.DataFrame(columns=["feature", "importance"])
    conn.close()
    return df


# ── 퍼센타일 레이더 ────────────────────────────────────────────

def get_player_percentiles(player_names: list[str]) -> pd.DataFrame:
    """
    Returns 0-100 percentile ranks using position-specific radar stats.
    Columns: player, pos_group + one column per POS_RADAR_STATS[pos_group] key.
    Players with different positions may have different column sets.
    """
    from scipy.stats import percentileofscore

    conn = sqlite3.connect(DB_PATH)
    master = pd.read_sql("SELECT * FROM players_master", conn)
    conn.close()

    master = master.drop_duplicates("player")
    master["pos_group"] = master["pos"].apply(_pos_group)

    # Pre-cast all possible stat columns
    all_cols = set(c for stats in POS_RADAR_STATS.values() for c in stats.values())
    all_cols |= set(RADAR_STATS.values())
    for col in all_cols:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0)

    records = []
    for name in player_names:
        row = master[master["player"].str.lower() == name.lower()]
        if row.empty:
            row = master[master["player"].str.contains(name, case=False, na=False)]
        if row.empty:
            continue
        row = row.iloc[0]
        pg = row["pos_group"]
        peers = master[master["pos_group"] == pg]
        radar = POS_RADAR_STATS.get(pg, RADAR_STATS)

        rec = {"player": row["player"], "pos_group": pg}
        for label, col in radar.items():
            if col in peers.columns:
                pct = round(percentileofscore(peers[col].values, row[col], kind="rank"), 1)
                if col in INVERTED_RADAR_STATS:
                    pct = round(100.0 - pct, 1)
                rec[label] = pct
            else:
                rec[label] = 0.0
        records.append(rec)

    return pd.DataFrame(records)


# ── 역할 클러스터링 ────────────────────────────────────────────

CLUSTER_FEATURES = [
    "per_90_minutes_gls", "per_90_minutes_ast", "per_90_minutes_g_a",
    "xg_p90", "xa_p90", "standard_sh_90",
    "performance_tklw", "performance_int", "performance_fld",
    "playing_time_min",
]

ROLE_LABELS = {
    "FW": {
        0: "Target Forward",   1: "False 9 / Link-Up",   2: "Clinical Striker",
        3: "Wide Forward",     4: "Pressing Forward",    5: "Withdrawn Striker",
    },
    "MF": {
        0: "Deep-Lying Playmaker", 1: "Box-to-Box",       2: "Attacking Midfielder",
        3: "Defensive Midfielder", 4: "Wide Midfielder",  5: "Pressing Midfielder",
    },
    "DF": {
        0: "Ball-Playing CB",  1: "Defensive CB",         2: "Fullback (Attacking)",
        3: "Fullback (Defensive)", 4: "Sweeper",          5: "Wing-Back",
    },
    "GK": {0: "Goalkeeper"},
}


def compute_role_clusters(n_clusters: int = 6, min_minutes: int = 500) -> pd.DataFrame:
    """
    K-means cluster players within each position group.
    Returns DataFrame with player + role_cluster + role_label columns.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    conn = sqlite3.connect(DB_PATH)
    master = pd.read_sql("SELECT * FROM players_master", conn)
    conn.close()

    master = master.drop_duplicates("player")
    master["pos_group"] = master["pos"].apply(_pos_group)
    master["playing_time_min"] = pd.to_numeric(master["playing_time_min"], errors="coerce").fillna(0)
    master = master[master["playing_time_min"] >= min_minutes].copy()

    for col in CLUSTER_FEATURES:
        master[col] = pd.to_numeric(master.get(col, 0), errors="coerce").fillna(0)

    records = []
    for pg in ["FW", "MF", "DF", "GK"]:
        subset = master[master["pos_group"] == pg].copy()
        if len(subset) < n_clusters:
            subset["role_cluster"] = 0
            subset["role_label"] = ROLE_LABELS.get(pg, {}).get(0, pg)
            records.append(subset[["player", "role_cluster", "role_label"]])
            continue

        feat_cols = [c for c in CLUSTER_FEATURES if c in subset.columns]
        X = subset[feat_cols].fillna(0).values
        X_scaled = StandardScaler().fit_transform(X)

        k = min(n_clusters, len(subset) // 10)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        subset["role_cluster"] = km.fit_predict(X_scaled)

        # Sort clusters by goal contribution to assign consistent labels
        cluster_means = subset.groupby("role_cluster")["per_90_minutes_gls"].mean().sort_values(ascending=False)
        cluster_rank = {old: new for new, old in enumerate(cluster_means.index)}
        subset["role_cluster"] = subset["role_cluster"].map(cluster_rank)
        labels = ROLE_LABELS.get(pg, {})
        subset["role_label"] = subset["role_cluster"].map(labels).fillna(pg)
        records.append(subset[["player", "role_cluster", "role_label"]])
        print(f"  [{pg}] {len(subset)} players → {k} clusters")

    return pd.concat(records, ignore_index=True)


def get_player_role(player_name: str) -> dict:
    """Return role_cluster and role_label for a single player."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT player, role_cluster, role_label FROM player_roles WHERE player = ? LIMIT 1",
            conn, params=[player_name]
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


if __name__ == "__main__":
    print("=== Value Scouting 파이프라인 ===\n")
    run_value_scouting(use_cached_tm=False)

    print("\n[결과] €20M 이하 저평가 공격수 TOP 10:")
    top = get_undervalued(max_value_eur=20_000_000, position="FW", top_n=10)
    print(top.to_string(index=False))
