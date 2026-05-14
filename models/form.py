"""
선수 폼 트렌드 분석 — LSTM 기반 시계열 예측
인천공항 인턴 LSTM 구조 재활용 (재무 시계열 → 선수 퍼포먼스 시계열)
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

DB_PATH = Path(__file__).parent.parent / "scout.db"
MODEL_PATH = Path(__file__).parent.parent / "form_lstm.pt"

FEATURES = ["shots", "passes", "pressures", "dribbles", "tackles", "interceptions"]

WEIGHTS = {"shots": 2.0, "passes": 0.3, "pressures": 0.5,
           "dribbles": 1.5, "tackles": 1.0, "interceptions": 1.0}

WINDOW = 5


# ── 모델 정의 ─────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class FormLSTM(nn.Module):
        def __init__(self, input_size: int = 6, hidden: int = 32, layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True)
            self.fc   = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)


# ── 데이터 준비 ───────────────────────────────────────────────

def _load_player_series(player_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT {', '.join(FEATURES)}, match_id FROM statsbomb_events "
        "WHERE player = ? ORDER BY match_id ASC",
        conn, params=[player_name]
    )
    conn.close()
    return df


def compute_form_score(row: pd.Series) -> float:
    return sum(float(row.get(f, 0)) * w for f, w in WEIGHTS.items())


def get_all_players_with_data(min_matches: int = 5) -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT player FROM statsbomb_events WHERE player IS NOT NULL "
        "GROUP BY player HAVING COUNT(*) >= ?",
        conn, params=[min_matches]
    )
    conn.close()
    return df["player"].tolist()


def build_windows(players: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    X_list, y_list = [], []

    for player in players:
        series = _load_player_series(player)
        if len(series) < WINDOW + 1:
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series[FEATURES].values.astype(float))

        for i in range(len(scaled) - WINDOW):
            X_list.append(scaled[i : i + WINDOW])
            next_row = pd.Series(dict(zip(FEATURES, series[FEATURES].iloc[i + WINDOW].values)))
            y_list.append(compute_form_score(next_row))

    if not X_list:
        raise RuntimeError("학습 데이터 부족")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    y_max = float(y.max()) if y.max() > 0 else 1.0
    y = y / y_max
    return X, y, y_max


# ── 학습 ─────────────────────────────────────────────────────

def train(epochs: int = 80, lr: float = 1e-3):
    print("[Form LSTM] 학습 데이터 준비 중...")
    players = get_all_players_with_data()
    X, y, y_max = build_windows(players)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    model = FormLSTM(input_size=len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    torch.save({"state_dict": model.state_dict(), "y_max": y_max}, MODEL_PATH)
    print(f"[Form LSTM] 저장 완료 → {MODEL_PATH}")
    return model


def load_model():
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch not available")
    if not MODEL_PATH.exists():
        train()
    data = torch.load(MODEL_PATH, weights_only=True)
    model = FormLSTM(input_size=len(FEATURES))
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data["y_max"]


# ── 예측 & 트렌드 ─────────────────────────────────────────────

def get_form_trend(player_name: str) -> dict:
    """
    반환:
      has_data    — StatsBomb 데이터 존재 여부
      scores      — 경기별 폼 스코어 리스트
      match_ids   — 경기 순서 인덱스
      trend       — "상승 ↑" | "하락 ↓" | "안정 →"
      slope       — 최근 5경기 기울기
      prediction  — LSTM 다음 경기 예측값
    """
    series = _load_player_series(player_name)

    if len(series) < 3:
        return {"has_data": False, "player": player_name}

    scores = series.apply(compute_form_score, axis=1).tolist()
    match_ids = list(range(1, len(scores) + 1))

    recent = scores[-5:]
    x = np.arange(len(recent))
    slope = float(np.polyfit(x, recent, 1)[0])

    if slope > 1.5:
        trend = "Rising ↑"
    elif slope < -1.5:
        trend = "Falling ↓"
    else:
        trend = "Stable →"

    prediction = None
    if len(series) >= WINDOW:
        try:
            model, y_max = load_model()
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series[FEATURES].values.astype(float))
            window_data = torch.tensor(
                scaled[-WINDOW:][np.newaxis, :, :], dtype=torch.float32
            )
            with torch.no_grad():
                prediction = float(model(window_data).item() * y_max)
        except Exception as e:
            print(f"[Form LSTM] 예측 실패: {e}")

    return {
        "has_data": True,
        "player": player_name,
        "scores": scores,
        "match_ids": match_ids,
        "trend": trend,
        "slope": round(slope, 2),
        "prediction": round(prediction, 1) if prediction is not None else None,
    }


SEASON_LABELS = {"2223": "22-23", "2324": "23-24", "2425": "24-25", "2526": "25-26"}


def get_season_trend(player_name: str) -> dict:
    """
    players_raw + understat_xg 시즌별 스탯 → 성장 곡선 + 방향성 지표.
    반환:
      has_data / seasons / gls_p90 / ast_p90 / g_a_p90 / sh_90 / xg_p90 / xa_p90 /
      minutes / trend / trajectory_score / peak_season / momentum
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            """SELECT season, playing_time_min, per_90_minutes_gls, per_90_minutes_ast,
                      per_90_minutes_g_a, standard_sh_90
               FROM players_raw WHERE player = ? ORDER BY season ASC""",
            conn, params=[player_name],
        )
        try:
            xg_df = pd.read_sql(
                """SELECT season,
                          SUM(CAST(xg AS REAL)) / NULLIF(SUM(CAST(minutes AS REAL)), 0) * 90 AS xg_p90,
                          SUM(CAST(xa AS REAL)) / NULLIF(SUM(CAST(minutes AS REAL)), 0) * 90 AS xa_p90
                   FROM understat_xg WHERE player = ? GROUP BY season""",
                conn, params=[player_name],
            )
        except Exception:
            xg_df = pd.DataFrame()
        conn.close()
    except Exception:
        try:
            raw_path = DB_PATH.parent / "data" / "players_raw.parquet"
            all_raw = pd.read_parquet(raw_path)
            df = all_raw[all_raw["player"] == player_name].sort_values("season")
            df = df[[c for c in ["season", "playing_time_min", "per_90_minutes_gls",
                                  "per_90_minutes_ast", "per_90_minutes_g_a", "standard_sh_90"]
                     if c in df.columns]]
            xg_df = pd.DataFrame()
        except Exception:
            return {"has_data": False, "player": player_name}

    if df.empty:
        return {"has_data": False, "player": player_name}

    for col in ["playing_time_min", "per_90_minutes_gls", "per_90_minutes_ast",
                "per_90_minutes_g_a", "standard_sh_90"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[df["playing_time_min"] >= 200].copy()
    if df.empty:
        return {"has_data": False, "player": player_name}

    if not xg_df.empty:
        xg_df["season"] = xg_df["season"].astype(str)
        df = df.merge(xg_df, on="season", how="left")
    else:
        df["xg_p90"] = np.nan
        df["xa_p90"] = np.nan

    df["season_label"] = df["season"].map(SEASON_LABELS).fillna(df["season"])

    ga = df["per_90_minutes_g_a"].tolist()
    n = len(ga)

    if n >= 2:
        slope = float(np.polyfit(np.arange(n), ga, 1)[0])
        trajectory_score = round(float(np.clip(slope / 0.002, -100, 100)), 1)
        trend = "↑ Rising" if slope > 0.03 else ("↓ Declining" if slope < -0.03 else "→ Stable")
    else:
        slope = 0.0
        trajectory_score = 0.0
        trend = "→ Stable"

    peak_idx = int(np.argmax(ga)) if ga else 0
    peak_season = df["season_label"].iloc[peak_idx] if not df.empty else "—"
    momentum = round(ga[-1] - ga[-2], 3) if n >= 2 else 0.0

    return {
        "has_data":          True,
        "player":            player_name,
        "seasons":           df["season_label"].tolist(),
        "gls_p90":           df["per_90_minutes_gls"].round(3).tolist(),
        "ast_p90":           df["per_90_minutes_ast"].round(3).tolist(),
        "g_a_p90":           df["per_90_minutes_g_a"].round(3).tolist(),
        "sh_90":             df["standard_sh_90"].round(3).tolist(),
        "xg_p90":            df["xg_p90"].round(3).tolist() if "xg_p90" in df else [],
        "xa_p90":            df["xa_p90"].round(3).tolist() if "xa_p90" in df else [],
        "minutes":           df["playing_time_min"].astype(int).tolist(),
        "trend":             trend,
        "trajectory_score":  trajectory_score,
        "peak_season":       peak_season,
        "momentum":          momentum,
    }


if __name__ == "__main__":
    train()
    result = get_form_trend("Lionel Andrés Messi Cuccittini")
    print(f"\n[Messi 폼 트렌드]")
    print(f"  트렌드: {result['trend']}  (기울기: {result['slope']})")
    print(f"  최근 5경기: {[round(s, 1) for s in result['scores'][-5:]]}")
    print(f"  다음 경기 예측: {result['prediction']}")
