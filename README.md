# ⚽ Football Scout AI

> **AI-powered player scouting and market value analysis across Europe's Big 5 leagues**  
> Multi-season data (2022–2026) · XGBoost value prediction · LSTM form trends · Natural language search

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![XGBoost](https://img.shields.io/badge/XGBoost-regression-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-ee4c2c)

---

## What This Does

Football clubs spend hundreds of millions on transfers — often mispricing players whose statistical output tells a different story. This project builds a data pipeline and ML model to answer:

> *"Given a player's multi-season performance profile, what should their market value be — and who is the market mispricing right now?"*

**Three core features:**

| Feature | Description |
|---------|-------------|
| 🔍 **Natural Language Search** | Ask in plain language — *"young pressing midfielder in the Bundesliga"* — and get ranked results |
| 💰 **Value Scouting** | XGBoost model predicts fair market value from performance data; surfaces undervalued players |
| 📈 **Form Trend (LSTM)** | Match-by-match form score with next-game prediction using PyTorch LSTM |

---

## Key Findings

Full analysis: [`REPORT.md`](./REPORT.md)

**Most undervalued players (sample, ≤ €30M budget):**

| Player | Club | Actual (M€) | Predicted (M€) | Undervalue |
|--------|------|-------------|----------------|------------|
| Óscar Mingueza | Celta Vigo | 18.0 | 24.1 | **+34%** |
| Ritsu Doan | Frankfurt | 20.0 | 25.0 | **+25%** |
| Yann Gboho | Toulouse | 15.0 | 18.6 | **+24%** |
| Georges Mikautadze | Villarreal | 28.0 | 32.5 | **+16%** |

**League pattern:** Ligue 1 and Bundesliga consistently offer better value than the Premier League, which carries a ~20% brand premium on average.

---

## Architecture

```
football-scout-ai/
├── data/
│   └── pipeline.py          # FBref multi-season collection + weighted merge
├── models/
│   ├── search.py             # NL query parsing (Groq/Llama 3.3) + DB search
│   ├── form.py               # LSTM form trend model (PyTorch)
│   └── value_scouting.py     # Transfermarkt scraper + XGBoost value model
├── app/
│   └── streamlit_app.py      # Streamlit dashboard (2 tabs)
├── REPORT.md                 # Full analysis writeup
└── scout.db                  # SQLite (gitignored — run pipeline to generate)
```

**Data flow:**
```
FBref (4 seasons) ──┐
                    ├──► Weighted avg ──► XGBoost ──► Undervalue score
Transfermarkt ──────┘

StatsBomb events ──► LSTM ──► Form trend + next-game prediction

User query ──► Groq (Llama 3.3) ──► Structured filters ──► SQLite query
```

---

## Methodology

### Multi-Season Weighting

Rather than using a single season snapshot, performance stats are weighted across four seasons:

| Season | Weight | Reason |
|--------|--------|--------|
| 2022–23 | 15% | Historical baseline |
| 2023–24 | 25% | Recent full season |
| 2024–25 | 35% | Most recent complete |
| 2025–26 | 25% | Current (in progress) |

Minimum 500 minutes per season required.

### Value Model

- **Target:** `log(market_value + 1)` — log-transformed for variance stability
- **Features:** 16 performance stats + age factor + league tier weights
- **Model:** XGBoost Regressor (300 estimators, 5-fold CV RMSE: 0.465)
- **Undervalue score:** `(predicted − actual) / actual × 100`

See [`REPORT.md`](./REPORT.md) for full methodology.

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/your-username/football-scout-ai
cd football-scout-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Add your GROQ_API_KEY (free at console.groq.com)
```

### 3. Build the database

```bash
# Collect 4 seasons of FBref data + Transfermarkt values (~30 min)
python data/pipeline.py
python models/value_scouting.py
```

### 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| [FBref](https://fbref.com) | Player season stats (Big 5 leagues, 4 seasons) | Public via `soccerdata` |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Match event data | Free / open |
| [Transfermarkt](https://transfermarkt.com) | Market valuations | Public (scraped) |

---

## Requirements

```
soccerdata       # FBref data collection
statsbombpy      # StatsBomb event data
pandas / numpy   # Data processing
xgboost          # Market value model
torch            # LSTM form model
scikit-learn     # Model validation
streamlit        # Dashboard
plotly           # Visualisations
groq             # NL query parsing (Llama 3.3 70B)
beautifulsoup4   # Transfermarkt scraping
```

---

## Limitations

- Name matching between FBref and Transfermarkt is string-based (~8% unmatched)
- No contract length data — expiring contracts deflate valuations independently of performance
- Five broad position groups; position-specific models would improve precision
- StatsBomb event data covers a limited set of competitions (La Liga 2020–21)

---

## Author

Built as part of a sports analytics portfolio targeting football club data roles.  
Full analysis writeup: [`REPORT.md`](./REPORT.md)

*Feedback and PRs welcome.*
