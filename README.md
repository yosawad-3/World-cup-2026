# ⚽ World Cup 2026 Match Predictor

A machine learning project that predicts football match outcomes and simulates a full knockout tournament (Round of 32 → Final), built around the FIFA World Cup 2026 bracket. Includes an interactive **Streamlit** web app.

🔗 **Live demo:** _add your Streamlit app URL here_

---

## 📖 Overview

This project predicts the winner of a football match between any two national teams, and can simulate an entire knockout tournament bracket to estimate which team is most likely to become champion.

The pipeline covers the full data science workflow:

1. Data cleaning (missing values, team name standardization, duplicates)
2. Feature engineering (Elo ratings, form, strength differentials)
3. Binary classification model (XGBoost) trained on historical match data
4. Knockout-tournament bracket simulation (deterministic + Monte Carlo)
5. An interactive Streamlit UI for single-match predictions and full tournament simulation

---

## 🗂️ Project Structure

```
├── app.py                    # Streamlit web app
├── requirements.txt          # Python dependencies
├── model.pkl                 # Trained XGBoost model
├── teams_data_final.csv      # Cleaned team ratings (Elo, attack, defense, form...)
├── x_columns.json            # Exact feature column order used at training time
└── README.md
```

---

## 🧩 Data

Two datasets were used:

- **`teams_data`** — one row per national team, with rating features:
  `elo, overall, max_overall, attack, defense, pace, shooting, passing, form_scored, form_conceded, form_win_rate`
- **`teams_match`** — 40,000+ historical international matches with home/away versions of the above features plus match context (`is_neutral`, `is_world_cup`, `is_continental`) and the actual result.

### Cleaning steps
- KNN imputation for missing `pace` / `shooting` / `passing` values
- Team name standardization (`strip()` + `title()`) and manual mapping for alternate names (e.g., `North Korea` → `Korea Dpr`, `Ivory Coast` → `Côte D'Ivoire`)
- Duplicate match removal
- Filtered to recent seasons (2018–2025) to avoid outdated data patterns

---

## 🤖 Model

- **Algorithm:** XGBoost Classifier (binary: home win vs. away win — draws excluded, since knockout matches always produce a winner)
- **Tuning:** `RandomizedSearchCV` over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `min_child_weight`
- **Split:** 85% train / 15% test, stratified
- **Key features:** `elo_diff` is by far the strongest predictor, followed by `overall_diff` and `defense_diff`

### Handling home-field advantage
Since World Cup matches are played on neutral ground, each match is scored twice (once with each team as "home"), and the two probability sets are averaged — this cancels out the home-field bias learned from historical data.

---

## 🏆 Tournament Simulation

Given a Round of 32 bracket, the app walks the knockout tree round by round:

```
Round of 32 → Round of 16 → Quarterfinal → Semifinal → Final
```

Two simulation modes:
- **Deterministic** — always picks the team with the higher predicted probability
- **Monte Carlo** — runs the bracket thousands of times with probability-weighted random outcomes, producing a championship-probability distribution across all teams

---

## ⚠️ Limitations

- Predictions are probability estimates based on historical patterns, not guarantees
- The model leans heavily on Elo rating differences, so it favors higher-ranked teams and may be less reliable for closely matched teams or genuine upsets
- Team rating data (`teams_data`) reflects a snapshot in time and does not capture very recent form changes (injuries, new managers, etc.)

---

## 🛠️ Tech Stack

- Python, pandas, NumPy, scikit-learn, XGBoost
- Streamlit (UI)
- joblib (model serialization)
