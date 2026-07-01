# ⚽ Football Scout AI

> **AI-powered player scouting platform for Europe's Big 5 leagues**  
> Multi-season weighted stats (22/23 – 25/26) · xG/xA from Understat · Position-specific XGBoost value model · Contract-aware valuation · Tactical role clustering · Percentile radar · PDF scout reports · Natural language search

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)](https://football-scout-ai-kujc5juarc7gdz4vhwaapb.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/XGBoost-position--specific-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)


---

## Overview

Football clubs spend hundreds of millions on transfers — often mispricing players whose statistical profile tells a different story. This project builds an end-to-end data pipeline and ML system to answer three practical scouting questions:

1. **Who is the market undervaluing right now?** — XGBoost models trained separately per position predict performance-based fair value
2. **Who plays like this player, but costs less?** — Cosine similarity over normalised stats finds cheaper like-for-like replacements
3. **How do I find players meeting specific criteria fast?** — Natural language search (Llama 3.3 70B via Groq) parses plain English or Korean into structured DB filters

**3,435 players · 5 leagues · 4 seasons · 24 ML features including xG/xA + contract data · 1,955 players with Transfermarkt valuations**

---

## Features

| | Feature | Description |
|---|---------|-------------|
| 🔍 | **Natural Language Search** | *"clinical striker under 25 in Ligue 1"* — bilingual (EN/KR), parsed by Llama 3.3 into SQL filters; supports `sort_by: form`, `undervalue`, `goals`, `tackles` |
| 💰 | **Value Scouting** | Separate XGBoost models per position (FW/MF/DF/GK); 24 features including contract years, team prestige, premium nation flag; surfaces undervalued players via `(predicted − actual) / actual` |
| 🎯 | **Similar Player Finder** | Cosine similarity over normalised stats (goals, xG, xA, tackles, interceptions + age/league tier); returns cheaper alternatives ranked by profile match |
| 🕹️ | **Tactical Role Clustering** | K-means (k=6) within each position group — 20 role labels: *Target Forward*, *Box-to-Box*, *Ball-Playing CB*, etc. |
| 📊 | **Percentile Radar** | Position-specific radar — GK uses Save%/GA90/CS%/Saves90; outfield uses xG/xA/Tackles/Interceptions; normalised vs same-position peers across Big 5 |
| 📈 | **Multi-Season Form Trend** | Season-by-season G/A trajectory (22/23–25/26) for all 3,435 players; LSTM match-by-match prediction where StatsBomb event data is available |
| 🔬 | **Model Validation** | Temporal backtest (2022-23 OOF vs 4-season full model) with RMSE bar chart + actual-vs-predicted scatter; quantifies multi-season benefit (~28% avg improvement) |
| 📉 | **Age-Value Curve** | Position-specific polynomial age curve showing market value peak by position with per-age median overlay |
| ★ | **Watchlist** | Save players across searches; side-by-side radar + stat comparison for up to 5 players |
| 📄 | **PDF Scout Report** | One-click export: player header, percentile progress bars, matplotlib polar radar chart, market value assessment with ±1σ confidence interval |

---

## Architecture

```
Data Sources
────────────
FBref  (via soccerdata)        4 seasons × 5 leagues × 4 stat types
Understat (via soccerdata)     xG, npxG, xA per player/season
Transfermarkt (scraped)        Market valuations + contract expiry dates
StatsBomb Open Data            Match event data

Pipeline
────────
FBref standard/shooting/misc ──► season-weighted merge ──► players_master (3,435 × 74 cols)
Understat xG/xA ───────────────► name-normalised join  ──┘  (81% coverage)
TM squad pages ─────────────────► market_value_eur + contract_year (96% coverage)

                    ┌── FW model (XGBoost, n=482) ──┐
players_master  ────┼── MF model (XGBoost, n=577) ──┼──► predicted_value_eur
+ TM valuations     ├── DF model (XGBoost, n=731) ──┤    undervalue_score
+ contract data     └── GK model (XGBoost, n=165) ──┘    performance_premium

players_master ──► K-means (k=6 per position) ──► 20 tactical role labels

players_raw (11,390 season rows) ──► season trend charts
StatsBomb events (1,071 rows)    ──► LSTM form prediction

App Layer
─────────
query    ──► Groq Llama 3.3 70B ──► JSON filters ──► SQLite ──► ranked results
player   ──► cosine_similarity(StandardScaler(12 stats)) ──► similar players
player   ──► scipy percentileofscore vs position peers  ──► radar chart
player   ──► matplotlib polar PNG + fpdf2 layout ──► PDF bytes ──► download
```

---

## Methodology

### Multi-Season Weighting

Stats are weighted across four seasons so recent form is prioritised without discarding career context:

| Season | Weight | Rationale |
|--------|--------|-----------|
| 2022–23 | 15% | Historical baseline |
| 2023–24 | 25% | Recent full season |
| 2024–25 | 35% | Most recent complete |
| 2025–26 | 25% | Current (partial) |

Minimum 500 minutes per season required to include a season in the weighted average.

### Value Model

| Attribute | Detail |
|-----------|--------|
| Target | `log1p(market_value_eur)` |
| Models | 4 separate XGBoost regressors — one per position group |
| Features | 24: age, seasons_count, playing_time, goals/ast per90, xG/xA/npxG per90, shots, tackles, interceptions, fouls, age_factor, league_tier, **contract_years_remaining**, **age_contract**, **team_value_m**, **premium_nation** |
| Regularization | Adaptive by sample size — GK (n=165): depth=2, λ=8; FW (n=482): depth=4, λ=2 |
| Training data | 1,955 players with TM valuations (squad-page scraping, 5 leagues × ~20 teams × ~25 players) |
| Validation | 5-fold CV RMSE (log-scale): FW **0.54**, MF **0.54**, DF **0.55**, GK **0.75** |
| Undervalue | `(predicted - actual) / actual * 100` |

Position-specific training matters because value drivers differ sharply: finishing stats (xG, npxG) dominate for forwards, while defensive actions are stronger signals for defenders.

#### Key Feature Engineering

| Feature | Description | Impact |
|---------|-------------|--------|
| `contract_years_remaining` | Years until contract expiry, scraped from TM (96% coverage) | Largest single non-performance driver |
| `age_contract` | Age × contract interaction — 24yo+4yr signals very differently from 32yo+4yr | Captures peak-years premium |
| `team_value_m` | Team's median TM market value (M€) — club prestige within league tier | Separates Man City from Brentford at same league tier |
| `premium_nation` | Flag for BRA/FRA/ENG/ESP/ARG/GER/POR/NED/ITA/BEL — international exposure proxy | ~5–10% value premium on average |

#### Model Accuracy vs Literature

| Model | RMSE (log) | Approx R² |
|-------|-----------|-----------|
| Stats only (baseline) | 0.85–0.95 | 0.55–0.65 |
| Müller et al. (2017) | 0.63–0.71 | 0.70–0.78 |
| Franceschi et al. (2023) | 0.58–0.68 | 0.73–0.80 |
| **This model (FW/MF/DF)** | **0.54–0.55** | **~0.85** |

The remaining irreducible error (~0.15–0.20) stems from market irrationality, agent influence, injury history, and transfer demand — factors not capturable from public statistical data alone.

### Temporal Backtest

To quantify the benefit of multi-season weighting, a separate model is trained on 2022-23 stats only using 5-fold OOF `cross_val_predict`. Results saved to `backtest_results` table and visualised in the Model Validation tab.

| Position | Historical (2223) | Full 4-Season | Improvement |
|----------|------------------|---------------|-------------|
| FW | 0.95 | 0.54 | −43% |
| MF | 0.95 | 0.54 | −43% |
| DF | 0.93 | 0.55 | −41% |
| GK | 1.32 | 0.75 | −43% |

### xG Integration

Understat player season stats collected via `soccerdata.Understat` — no JS rendering required. Coverage: **81.3%** of 3,435 players matched after Unicode name normalisation. Three features added: `xg_p90`, `npxg_p90`, `xa_p90`.

### League Difficulty Normalisation

Goal-scoring rates differ across leagues. Adjusted stats (`adj_gls_p90`, `adj_xg_p90`, etc.) computed as `raw × (global_mean / league_mean)`:

| League | Factor | Interpretation |
|--------|--------|----------------|
| GER-Bundesliga | 0.87 | High-scoring — stats slightly deflated |
| ENG-Premier League | 0.96 | Slightly above average |
| FRA-Ligue 1 | 1.02 | Average |
| ITA-Serie A | 1.07 | Harder to score |
| ESP-La Liga | 1.08 | Hardest — stats boosted |

### Tactical Role Clustering

K-means (k=6, n_init=10) on 10 standardised stats within each position group. Clusters are ranked by mean Goals/90 to assign consistent labels. 20 distinct role labels across all positions.

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/2eueu/football-scout-ai
cd football-scout-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment

```bash
cp .env.example .env
# Set GROQ_API_KEY  (free at console.groq.com — used for NL query parsing)
```

### 3. Run

The app auto-bootstraps `scout.db` from the parquet files in `data/` on first run — **no separate pipeline step needed**.

```bash
streamlit run app/streamlit_app.py
```

To rebuild data from scratch (requires `soccerdata` + `statsbombpy`):

```bash
python data/pipeline.py          # FBref multi-season collection (~30 min)
python models/value_scouting.py  # Transfermarkt scrape + contract data + model training (~30 min)
```

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| [FBref](https://fbref.com) | Player season stats — standard, shooting, playing_time, misc (Big 5, 4 seasons) | Public via `soccerdata` |
| [Understat](https://understat.com) | xG, npxG, xA, xG_chain per player/season | Public via `soccerdata.Understat` |
| [Transfermarkt](https://transfermarkt.com) | Market valuations + contract expiry dates (96% coverage) | Public (HTML scrape) |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Match event data | Free / open |

---

## Stack

```
Data        pandas, numpy, soccerdata, beautifulsoup4, rapidfuzz
ML          xgboost, scikit-learn, scipy
DL          PyTorch (LSTM form model — optional, graceful fallback if unavailable)
NLP         groq (Llama 3.3 70B) for NL query parsing
Storage     SQLite, parquet (pyarrow)
App         Streamlit, Plotly
Viz         matplotlib (PDF radar chart)
Export      fpdf2
```

---

## Limitations

- **Market value coverage**: TM squad-page scraping covers ~1,955 of 3,435 players (56.9%); remaining players lack market valuations and are excluded from value model training
- **xG coverage**: 81.3% matched; remaining players have xG imputed as 0
- **StatsBomb events**: Limited to open dataset competitions; LSTM form available for ~24 players only
- **GK model accuracy**: RMSE 0.75 vs 0.54–0.55 for outfield — limited by small training sample (n=165)
- **No injury history**: Injury-prone players are systematically overvalued by the model
- **No transfer demand signal**: Players linked to big clubs command premiums the model cannot capture

---

## Author

Built as a sports analytics portfolio targeting football club data analyst roles.  
Full methodology writeup: [`REPORT.md`](./REPORT.md)
