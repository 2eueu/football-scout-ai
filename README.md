# ⚽ Football Scout AI

> **AI-powered player scouting platform for Europe's Big 5 leagues**  
> Multi-season weighted stats (22/23 – 25/26) · xG/xA from Understat · Position-specific XGBoost value model · Tactical role clustering · Percentile radar · PDF scout reports · Natural language search

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)](https://football-scout-ai-kujc5juarc7gdz4vhwaapb.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/XGBoost-position--specific-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[→ Live Demo](https://football-scout-ai-kujc5juarc7gdz4vhwaapb.streamlit.app/)**

---

## Overview

Football clubs spend hundreds of millions on transfers — often mispricing players whose statistical profile tells a different story. This project builds an end-to-end data pipeline and ML system to answer three practical scouting questions:

1. **Who is the market undervaluing right now?** — XGBoost models trained separately per position predict performance-based fair value
2. **Who plays like this player, but costs less?** — Cosine similarity over 12 normalised stats finds cheaper like-for-like replacements
3. **How do I find players meeting specific criteria fast?** — Natural language search (Llama 3.3 70B via Groq) parses plain English or Korean into structured DB filters

**3,435 players · 5 leagues · 4 seasons · 21 ML features including xG/xA · 1,955 players with Transfermarkt valuations**

---

## Features

| | Feature | Description |
|---|---------|-------------|
| 🔍 | **Natural Language Search** | *"clinical striker under 25 in Ligue 1"* — bilingual (EN/KR), parsed by Llama 3.3 into SQL filters; supports `sort_by: form`, `undervalue`, `goals`, `tackles` |
| 💰 | **Value Scouting** | Separate XGBoost models per position (FW/MF/DF/GK); predicts log-market-value from 21 features; surfaces undervalued players via `(predicted − actual) / actual` |
| 🎯 | **Similar Player Finder** | Cosine similarity over normalised stats (goals, xG, xA, tackles, interceptions + age/league tier); returns cheaper alternatives ranked by profile match |
| 🕹️ | **Tactical Role Clustering** | K-means (k=6) within each position group — 20 role labels: *Target Forward*, *Box-to-Box*, *Ball-Playing CB*, etc. |
| 📊 | **Percentile Radar** | 8-stat radar (Goals/90, xG/90, Assists/90, xA/90, Shots/90, G+A/90, Tackles, Interceptions) normalised vs same-position peers across Big 5 |
| 📈 | **Multi-Season Form Trend** | Season-by-season G/A trajectory (22/23–25/26) for all 3,435 players; LSTM match-by-match prediction where StatsBomb event data is available |
| 📄 | **PDF Scout Report** | One-click export: player header, percentile progress bars, market value assessment |

---

## Architecture

```
Data Sources
────────────
FBref  (via soccerdata)        4 seasons x 5 leagues x 4 stat types
Understat (via soccerdata)     xG, npxG, xA per player/season
Transfermarkt (scraped)        Market valuations
StatsBomb Open Data            Match event data

Pipeline
────────
FBref standard/shooting/misc ──► season-weighted merge ──► players_master (3,435 x 74 cols)
Understat xG/xA ───────────────► name-normalised join  ──┘  (81% coverage)

                    ┌── FW model (XGBoost, n=135) ──┐
players_master  ────┼── MF model (XGBoost, n=161) ──┼──► predicted_value_eur
+ market values     ├── DF model (XGBoost, n=147) ──┤    undervalue_score
                    └── GK model (XGBoost, n=20)  ──┘

players_master ──► K-means (k=6 per position) ──► 20 tactical role labels

players_raw (11,390 season rows) ──► season trend charts
StatsBomb events (1,071 rows)    ──► LSTM form prediction

App Layer
─────────
query    ──► Groq Llama 3.3 70B ──► JSON filters ──► SQLite ──► ranked results
player   ──► cosine_similarity(StandardScaler(12 stats)) ──► similar players
player   ──► scipy percentileofscore vs position peers  ──► radar chart
player   ──► fpdf2 layout ──► PDF bytes ──► download button
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
| Features | 21: age, seasons_count, playing_time, goals/ast per90, xG/xA/npxG per90, shots, tackles, interceptions, fouls, age_factor, league_tier |
| Training data | 1,955 players with TM valuations (squad-page scraping, 5 leagues × ~20 teams × ~25 players) |
| Validation | 5-fold CV RMSE (log): FW 0.69, MF 0.70, DF 0.77, GK 0.99 |
| Undervalue | `(predicted - actual) / actual * 100` |

Position-specific training matters because value drivers differ sharply: finishing stats (xG, npxG) dominate for forwards, while defensive actions are stronger signals for defenders.

### xG Integration

Understat player season stats collected via `soccerdata.Understat` — no JS rendering required. Coverage: **81.3%** of 3,435 players matched after Unicode name normalisation. Three features added: `xg_p90`, `npxg_p90`, `xa_p90`.

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
python models/value_scouting.py  # Transfermarkt scrape + model training
```

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| [FBref](https://fbref.com) | Player season stats — standard, shooting, playing_time, misc (Big 5, 4 seasons) | Public via `soccerdata` |
| [Understat](https://understat.com) | xG, npxG, xA, xG_chain per player/season | Public via `soccerdata.Understat` |
| [Transfermarkt](https://transfermarkt.com) | Market valuations | Public (HTML scrape) |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Match event data | Free / open |

---

## Stack

```
Data        pandas, numpy, soccerdata, beautifulsoup4
ML          xgboost, scikit-learn, scipy
DL          PyTorch (LSTM form model — optional, graceful fallback if unavailable)
NLP         groq (Llama 3.3 70B) for NL query parsing
Storage     SQLite, parquet (pyarrow)
App         Streamlit, Plotly
Export      fpdf2
```

---

## Limitations

- **Market value coverage**: Transfermarkt squad-page scraping covers ~1,955 of 3,435 players (56.9%); remaining players lack TM market valuations
- **xG coverage**: 81.3% matched; remaining players have xG imputed as 0
- **StatsBomb events**: Limited to open dataset competitions; LSTM form available for ~24 players only
- **No contract data**: Expiring contracts depress market values independently of performance

---

## Author

Built as a sports analytics portfolio targeting football club data analyst roles.  
Full methodology writeup: [`REPORT.md`](./REPORT.md)
