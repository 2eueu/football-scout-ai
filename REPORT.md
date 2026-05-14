# Finding the Hidden Gems: An AI-Powered Value Scouting Analysis Across Europe's Big 5 Leagues

*Multi-season analysis (2022–2026) | XGBoost market value prediction | Transfermarkt × FBref*

---

## Abstract

Transfer market inefficiency is one of the most expensive problems in modern football. Clubs routinely overpay for high-profile names while missing players whose statistical output outpaces their market valuation. This report presents a data-driven framework for identifying **systematically undervalued players** across Europe's Big 5 leagues, combining four seasons of weighted performance data with an XGBoost regression model trained on Transfermarkt market values.

Key findings:
- **Óscar Mingueza** (Celta Vigo) is undervalued by **+34%** relative to his predicted market value
- **Georges Mikautadze** (Villarreal) posts **0.63 goals/90** at a €28M valuation — model estimates €32.5M
- **Fisnik Asllani** (Hoffenheim, age 23) represents the strongest upside in the U-23 bracket at €30M
- Ligue 1 and Bundesliga consistently surface the most undervalued players relative to their output

---

## 1. Introduction

The average Big 5 league transfer fee has risen over 40% in the past decade, yet the correlation between transfer spend and on-pitch output remains weak. Clubs with elite scouting infrastructure — Brentford, RB Leipzig, Brighton — have demonstrated that performance-based valuation models can identify market inefficiencies before the broader market corrects them.

This project builds such a model from publicly available data, asking a simple question:

> *Given a player's multi-season performance profile, what should their market value be — and who is the market mispricing right now?*

---

## 2. Data & Methodology

### 2.1 Performance Data — FBref (via soccerdata)

Player statistics were collected across four seasons:

| Season | Weight | Rationale |
|--------|--------|-----------|
| 2022–23 | 15% | Historical baseline |
| 2023–24 | 25% | Recent full season |
| 2024–25 | 35% | Most recent complete season |
| 2025–26 | 25% | Current season (in progress) |

Seasons with fewer than 500 minutes played were excluded to avoid small-sample noise. The weighted average gives more recent seasons greater influence on the final feature vector.

**Coverage:** 3,435 players across Premier League, La Liga, Bundesliga, Serie A, Ligue 1.

### 2.2 Market Values — Transfermarkt

Current market valuations were scraped from Transfermarkt (3,000 players), matched to FBref records by player name. **2,492 players** were successfully matched for model training.

### 2.3 Feature Engineering

**Base features (16):**

| Feature | Description |
|---------|-------------|
| `per_90_minutes_gls` | Weighted-average goals per 90 min |
| `per_90_minutes_ast` | Weighted-average assists per 90 min |
| `standard_sh_90` | Shots on target per 90 min |
| `performance_tklw` | Tackles won |
| `performance_int` | Interceptions |
| `playing_time_min` | Total weighted minutes |
| `seasons_count` | Number of seasons with data |

**Derived features:**

- **Age factor** — captures the market premium on youth potential:
  - Age ≤ 23: ×1.5 | Age 24–27: ×1.2 | Age 28–30: ×1.0 | Age 31+: ×0.75

- **League tier** — adjusts for competition quality:
  - Premier League: ×1.3 | La Liga: ×1.2 | Bundesliga: ×1.1 | Serie A: ×1.05 | Ligue 1: ×1.0

### 2.4 Model — XGBoost Regression

The target variable is `log(market_value + 1)`, log-transformed to stabilise variance across the wide valuation range (€100k to €200M+).

```python
XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8
)
```

**5-fold cross-validation RMSE: 0.465 (log scale)**

### 2.5 Undervalue Score

```
undervalue_score = (predicted_value − actual_value) / actual_value × 100
```

Positive scores indicate the model believes a player is worth more than the market currently prices them.

---

## 3. Key Findings

### 3.1 Most Undervalued Forwards (≤ €30M)

| Player | Club | League | Age | Actual (M€) | Predicted (M€) | Score |
|--------|------|--------|-----|-------------|----------------|-------|
| Yann Gboho | Toulouse | Ligue 1 | 25 | 15.0 | 18.6 | **+24.1%** |
| Georges Mikautadze | Villarreal | La Liga | 25 | 28.0 | 32.5 | **+15.9%** |
| Jonathan Rowe | Bologna | Serie A | 23 | 20.0 | 23.1 | **+15.5%** |
| Mika Biereth | Monaco | Ligue 1 | 23 | 18.0 | 20.4 | **+13.3%** |
| Fisnik Asllani | Hoffenheim | Bundesliga | 23 | 30.0 | 33.8 | **+12.6%** |

**Spotlight — Georges Mikautadze:** 0.63 goals/90 across multiple seasons is elite output for a €28M valuation. His move to Villarreal from Lyon already represents a market correction in progress; our model suggests there is still upside.

### 3.2 Most Undervalued Midfielders (≤ €30M)

| Player | Club | League | Age | Actual (M€) | Predicted (M€) | Score |
|--------|------|--------|-----|-------------|----------------|-------|
| Óscar Mingueza | Celta Vigo | La Liga | 27 | 18.0 | 24.1 | **+34.0%** |
| Ritsu Doan | Frankfurt | Bundesliga | 27 | 20.0 | 25.0 | **+25.1%** |
| Youssouf Fofana | Milan | Serie A | 27 | 25.0 | 30.7 | **+22.7%** |
| Samir El Mourabet | Strasbourg | Ligue 1 | 20 | 18.0 | 21.3 | **+18.6%** |
| Aleix García | Leverkusen | Bundesliga | 28 | 20.0 | 23.6 | **+18.2%** |

**Spotlight — Ritsu Doan:** 0.26 goals/90 and 0.17 assists/90 as a winger at Frankfurt places him in the top quartile of Bundesliga wide midfielders. At €20M against a predicted €25M, he represents one of the clearest buy signals in this analysis.

### 3.3 Most Undervalued Defenders (≤ €25M)

| Player | Club | League | Age | Actual (M€) | Predicted (M€) | Score |
|--------|------|--------|-----|-------------|----------------|-------|
| Matteo Ruggeri | Atlético Madrid | La Liga | 23 | 20.0 | 23.8 | **+19.2%** |
| Malang Sarr | Lens | Ligue 1 | 27 | 15.0 | 17.8 | **+18.8%** |
| Arthur Theate | Frankfurt | Bundesliga | 25 | 20.0 | 23.0 | **+15.0%** |
| Oumar Solet | Udinese | Serie A | 26 | 20.0 | 22.9 | **+14.4%** |

### 3.4 The Youth Premium: U-23 Value Picks

When filtering for players aged 23 or under with an undervalue score above 10%:

- **Jonathan Rowe** (23, Bologna) — Serie A breakout, 0.27 goals/90
- **Mika Biereth** (23, Monaco) — 0.63 goals/90, highest raw output in this bracket
- **Fisnik Asllani** (23, Hoffenheim) — 0.40 goals/90 with 0.28 assists/90
- **Matteo Ruggeri** (23, Atlético) — left back with strong defensive metrics

### 3.5 League-Level Patterns

```
Average undervalue score by league (all positions, ≤ €30M):
  Ligue 1:        +14.2%
  Bundesliga:     +12.8%
  Serie A:        +11.4%
  La Liga:        +10.1%
  Premier League:  +6.3%
```

**Interpretation:** Premier League players carry a significant brand premium. Ligue 1 and Bundesliga offer the best hunting ground for undervalued talent — consistent with the acquisition strategy of data-driven clubs like RB Leipzig and Brighton.

---

## 4. Model Validation

| Metric | Value |
|--------|-------|
| 5-fold CV RMSE (log scale) | 0.465 ± 0.041 |
| Players in training set | 2,492 |

The model is not designed to give precise point estimates — football valuations depend on contract length, injury history, and media profile. The undervalue score is best interpreted as a **signal of systematic mispricing**: players where observable performance data consistently outpaces what the market is paying.

**Top 5 features by importance (XGBoost):**
1. `playing_time_min` — reliability proxy
2. `age_factor` — youth premium
3. `per_90_minutes_gls` — goal output
4. `league_tier` — competition adjustment
5. `per_90_minutes_g_a` — combined goal involvement

---

## 5. Limitations

1. **Name-matching accuracy** — ~8% of players unmatched due to name variants
2. **No contract length data** — expiring contracts deflate market values independently of performance
3. **Positional nuance** — five broad position groups; a #10 and a box-to-box midfielder share the same label
4. **Injury history** — chronic injury-prone players may be legitimately undervalued by the market

---

## 6. Conclusion

This analysis demonstrates that performance-based market valuation is feasible with publicly available data, and that systematic undervaluation persists across Europe's Big 5 leagues.

The most actionable finding: **a €20–30M budget targeting players aged 23–27 in Ligue 1 or Bundesliga, filtered by undervalue score > 15%, returns a shortlist of players whose output per 90 is comparable to assets priced 30–50% higher in the Premier League.**

---

## 7. Technical Stack

| Component | Technology |
|-----------|-----------|
| Data collection | FBref via `soccerdata`, Transfermarkt (custom scraper) |
| Multi-season weighting | pandas, NumPy (4 seasons, weighted average) |
| Market value prediction | XGBoost, scikit-learn |
| Form trend modelling | PyTorch LSTM |
| Natural language search | Groq API (Llama 3.3 70B) |
| Interactive dashboard | Streamlit, Plotly |
| Database | SQLite |

**Seasons covered:** 2022–23 · 2023–24 · 2024–25 · 2025–26 — Big 5 leagues — ~3,400 players

---

## Appendix: Methodology Notes

**Why log-transform the target?**
Market values span four orders of magnitude (€100k to €200M+). Without log transformation, the model over-optimises for high-valued players and produces poor estimates at the lower end where most transfer activity occurs.

**Why weighted multi-season averages?**
A player's 2022–23 statistics are less predictive of current value than their 2024–25 output. Weighting recent seasons more heavily reduces noise from career anomalies (injury seasons, loan spells) while still capturing the consistency signal that multi-season data provides.

**Why XGBoost over linear regression?**
Market value has non-linear relationships with age (the youth premium is convex) and performance (the difference between 0.3 and 0.6 goals/90 matters far more at the top end of the market). Gradient boosting handles these interactions naturally without manual feature crosses.

---

*Data: FBref (fbref.com) · Transfermarkt (transfermarkt.com)*
*Period: 2022–23 to 2025–26*
*Code: github.com/your-username/football-scout-ai*
