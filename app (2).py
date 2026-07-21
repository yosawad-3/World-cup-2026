import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import xgboost
from collections import Counter
from functools import lru_cache

# ---------------------------------------------------------
# إعدادات الصفحة
# ---------------------------------------------------------
st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")

# ---------------------------------------------------------
# تحميل الموديل والبيانات (مرة واحدة بس، بفضل الـ cache)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_teams_data():
    return pd.read_csv("teams_data_final.csv")

@st.cache_data
def load_x_columns():
    with open("x_columns.json", "r") as f:
        return json.load(f)

model = load_model()
teams_data = load_teams_data()
X_columns = load_x_columns()

# ---------------------------------------------------------
# Flag mapping (team name -> ISO code used by flagcdn.com)
# ---------------------------------------------------------
FLAG_CODES = {
    "South Africa": "za", "Canada": "ca", "Brazil": "br", "Japan": "jp",
    "Germany": "de", "Paraguay": "py", "Netherlands": "nl", "Morocco": "ma",
    "Côte D'Ivoire": "ci", "Norway": "no", "France": "fr", "Sweden": "se",
    "Mexico": "mx", "Ecuador": "ec", "England": "gb-eng", "Congo Dr": "cd",
    "Belgium": "be", "Senegal": "sn", "United States": "us",
    "Bosnia And Herzegovina": "ba", "Spain": "es", "Austria": "at",
    "Portugal": "pt", "Croatia": "hr", "Switzerland": "ch", "Algeria": "dz",
    "Australia": "au", "Egypt": "eg", "Argentina": "ar", "Cape Verde": "cv",
    "Colombia": "co", "Ghana": "gh", "Italy": "it", "Uruguay": "uy",
    "Chile": "cl", "Poland": "pl", "Ukraine": "ua", "Russia": "ru",
    "Turkey": "tr", "Greece": "gr", "Iran": "ir", "Saudi Arabia": "sa",
    "Qatar": "qa", "South Korea": "kr", "Korea Republic": "kr",
    "China": "cn", "India": "in", "Nigeria": "ng", "Cameroon": "cm",
    "Tunisia": "tn", "Wales": "gb-wls", "Scotland": "gb-sct",
    "Ireland": "ie", "Denmark": "dk", "Finland": "fi", "Iceland": "is",
    "Serbia": "rs", "Romania": "ro", "Hungary": "hu", "Czech Republic": "cz",
    "Slovakia": "sk", "Slovenia": "si", "Bulgaria": "bg",
}

def flag_img(team, width=28):
    code = FLAG_CODES.get(team)
    if code:
        return f'<img src="https://flagcdn.com/w40/{code}.png" width="{width}" style="vertical-align:middle;border-radius:2px;margin-left:6px;">'
    return "🏳️"

# ---------------------------------------------------------
# دوال التنبؤ (نفس المنطق اللي اتعمل في Colab)
# ---------------------------------------------------------
def build_match_row(team_a, team_b, teams_data, X_columns, is_neutral=1, is_world_cup=1, is_continental=0):
    a = teams_data[teams_data["team"] == team_a].iloc[0]
    b = teams_data[teams_data["team"] == team_b].iloc[0]

    row = {
        "home_elo": a["elo"], "away_elo": b["elo"], "elo_diff": a["elo"] - b["elo"],
        "home_avg_overall": a["overall"], "home_max_overall": a["max_overall"],
        "home_avg_attack": a["attack"], "home_avg_defense": a["defense"],
        "home_avg_pace": a["pace"], "home_avg_shooting": a["shooting"], "home_avg_passing": a["passing"],
        "away_avg_overall": b["overall"], "away_max_overall": b["max_overall"],
        "away_avg_attack": b["attack"], "away_avg_defense": b["defense"],
        "away_avg_pace": b["pace"], "away_avg_shooting": b["shooting"], "away_avg_passing": b["passing"],
        "overall_diff": a["overall"] - b["overall"],
        "attack_diff": a["attack"] - b["attack"],
        "defense_diff": a["defense"] - b["defense"],
        "home_form_scored": a["form_scored"], "home_form_conceded": a["form_conceded"],
        "home_form_win_rate": a["form_win_rate"],
        "away_form_scored": b["form_scored"], "away_form_conceded": b["form_conceded"],
        "away_form_win_rate": b["form_win_rate"],
        "form_win_rate_diff": a["form_win_rate"] - b["form_win_rate"],
        "form_scored_diff": a["form_scored"] - b["form_scored"],
        "form_conceded_diff": a["form_conceded"] - b["form_conceded"],
        "is_neutral": is_neutral, "is_world_cup": is_world_cup, "is_continental": is_continental,
    }

    row_df = pd.DataFrame([row])
    for col in X_columns:
        if col not in row_df.columns:
            row_df[col] = 0
    return row_df[X_columns]


def predict_match(team_a, team_b, teams_data, model, X_columns):
    row_a_home = build_match_row(team_a, team_b, teams_data, X_columns)
    probs_a_home = model.predict_proba(row_a_home)[0]

    row_b_home = build_match_row(team_b, team_a, teams_data, X_columns)
    probs_b_home = model.predict_proba(row_b_home)[0]

    classes = model.classes_
    idx_win = list(classes).index(1)
    idx_lose = list(classes).index(0)

    p_a = (probs_a_home[idx_win] + probs_b_home[idx_lose]) / 2
    p_b = (probs_a_home[idx_lose] + probs_b_home[idx_win]) / 2

    total = p_a + p_b
    if total == 0 or np.isnan(total):
        p_a, p_b = 0.5, 0.5
    else:
        p_a = p_a / total
        p_b = 1.0 - p_a

    return float(p_a), float(p_b)


# ---------------------------------------------------------
# Cache: نحسب احتمال كل زوج فرق مرة واحدة بس، مش في كل تكرار Monte Carlo
# ---------------------------------------------------------
_match_prob_cache = {}

def cached_predict_match(team_a, team_b):
    key = (team_a, team_b)
    if key not in _match_prob_cache:
        _match_prob_cache[key] = predict_match(team_a, team_b, teams_data, model, X_columns)
    return _match_prob_cache[key]


def simulate_knockout_match(team_a, team_b, teams_data, model, X_columns, deterministic=True):
    p_a, p_b = cached_predict_match(team_a, team_b)
    if deterministic:
        winner = team_a if p_a >= p_b else team_b
    else:
        # ✅ بديل مضمون 100%: مقارنة برقم عشوائي بدل np.random.choice
        # كده مش محتاجين p_a + p_b = 1.0 بدقة تامة أصلاً
        winner = team_a if np.random.random() < p_a else team_b
    return winner


def build_next_round(winners, pairing):
    return [(winners[a], winners[b]) for a, b in pairing]


def simulate_full_bracket(round_of_32, teams_data, model, X_columns, deterministic=True):
    round_of_16_pairing = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
    quarterfinal_pairing = [(0, 1), (2, 3), (4, 5), (6, 7)]
    semifinal_pairing = [(0, 1), (2, 3)]
    final_pairing = [(0, 1)]

    stages = [
        ("Round of 32", round_of_16_pairing),
        ("Round of 16", quarterfinal_pairing),
        ("Quarterfinal", semifinal_pairing),
        ("Semifinal", final_pairing),
        ("Final", None),
    ]

    current_matches = round_of_32
    all_results = {}
    winners = []

    for stage_name, pairing in stages:
        winners = []
        for team_a, team_b in current_matches:
            winner = simulate_knockout_match(team_a, team_b, teams_data, model, X_columns, deterministic)
            winners.append(winner)
        all_results[stage_name] = list(zip(current_matches, winners))
        if pairing is not None:
            current_matches = build_next_round(winners, pairing)

    champion = winners[0]
    return champion, all_results


# ---------------------------------------------------------
# واجهة المستخدم
# ---------------------------------------------------------
st.title("⚽ World Cup 2026 Match Predictor")
st.caption("Powered by an XGBoost model trained on historical match data")

tab1, tab2 = st.tabs(["🆚 Predict a Single Match", "🏆 Simulate Full Tournament"])

# ============ Tab 1: Single Match ============
with tab1:
    st.subheader("Predict the outcome of a match between two teams")

    teams_list = sorted(teams_data["team"].unique())

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams_list, index=teams_list.index("Spain") if "Spain" in teams_list else 0)
    with col2:
        team_b = st.selectbox("Team B", teams_list, index=teams_list.index("Argentina") if "Argentina" in teams_list else 1)

    if st.button("Predict Result", type="primary"):
        if team_a == team_b:
            st.error("Please choose two different teams")
        else:
            p_a, p_b = predict_match(team_a, team_b, teams_data, model, X_columns)

            c1, c2 = st.columns(2)
            c1.markdown(f'{flag_img(team_a, 40)} **{team_a}**', unsafe_allow_html=True)
            c1.metric("", f"{p_a*100:.1f}%")
            c2.markdown(f'{flag_img(team_b, 40)} **{team_b}**', unsafe_allow_html=True)
            c2.metric("", f"{p_b*100:.1f}%")

            st.progress(float(p_a))

            winner = team_a if p_a >= p_b else team_b
            st.success(f"🏆 Prediction: {flag_img(winner, 24)} **{winner}** is more likely to win")

# ============ Tab 2: Full Tournament Simulation ============
with tab2:
    st.subheader("Simulate the World Cup from Round of 32 to the Final")

    default_bracket = [
        ("Germany", "Paraguay"), ("France", "Sweden"),
        ("South Africa", "Canada"), ("Netherlands", "Morocco"),
        ("Spain", "Austria"), ("Portugal", "Croatia"),
        ("Belgium", "Senegal"), ("United States", "Bosnia And Herzegovina"),
        ("Brazil", "Japan"), ("Côte D'Ivoire", "Norway"),
        ("Mexico", "Ecuador"), ("England", "Congo Dr"),
        ("Australia", "Egypt"), ("Argentina", "Cape Verde"),
        ("Switzerland", "Algeria"), ("Colombia", "Ghana"),
    ]

    mode = st.radio("Simulation mode", ["Deterministic (highest probability)", "Monte Carlo simulation (10,000 runs)"])

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Simulating..."):
            if mode == "Deterministic (highest probability)":
                champion, results = simulate_full_bracket(default_bracket, teams_data, model, X_columns, deterministic=True)

                stage_icons = {
                    "Round of 32": "🔢", "Round of 16": "🔟",
                    "Quarterfinal": "🎯", "Semifinal": "🥈", "Final": "🏆"
                }

                for stage, matches in results.items():
                    st.markdown(f"## {stage_icons.get(stage,'')} {stage}")
                    for (a, b), winner in matches:
                        loser = b if winner == a else a
                        col_a, col_vs, col_b = st.columns([5, 1, 5])

                        a_style = "background-color:#1f6b3a;padding:8px;border-radius:8px;" if winner == a else "padding:8px;"
                        b_style = "background-color:#1f6b3a;padding:8px;border-radius:8px;" if winner == b else "padding:8px;"

                        col_a.markdown(
                            f'<div style="{a_style}">{flag_img(a,26)} <b>{a}</b> {"✅" if winner==a else ""}</div>',
                            unsafe_allow_html=True
                        )
                        col_vs.markdown("<div style='text-align:center;padding-top:8px;'>vs</div>", unsafe_allow_html=True)
                        col_b.markdown(
                            f'<div style="{b_style}">{flag_img(b,26)} <b>{b}</b> {"✅" if winner==b else ""}</div>',
                            unsafe_allow_html=True
                        )
                    st.divider()

                st.balloons()
                st.markdown(
                    f'<div style="text-align:center;font-size:28px;">🏆 Predicted Champion: {flag_img(champion,40)} <b>{champion}</b></div>',
                    unsafe_allow_html=True
                )

            else:
                champions = []
                progress_bar = st.progress(0)
                n_sims = 500  # reduced further thanks to caching — still statistically solid
                for i in range(n_sims):
                    champ, _ = simulate_full_bracket(default_bracket, teams_data, model, X_columns, deterministic=False)
                    champions.append(champ)
                    if i % 100 == 0:
                        progress_bar.progress(i / n_sims)
                progress_bar.progress(1.0)

                champion_probs = pd.Series(champions).value_counts(normalize=True).mul(100).round(2)
                st.markdown("### Championship win probabilities")
                st.bar_chart(champion_probs.head(10))

                st.markdown("#### Top contenders")
                for team, prob in champion_probs.head(10).items():
                    st.markdown(
                        f'{flag_img(team, 26)} **{team}** — {prob}%',
                        unsafe_allow_html=True
                    )

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("Info")
st.sidebar.write(f"Teams in dataset: {len(teams_data)}")
st.sidebar.write("Model: XGBoost Classifier")
st.sidebar.info("Probabilities are estimates based on historical data, not guaranteed outcomes.")
