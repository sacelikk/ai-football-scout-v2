# 🧠 AI-Driven Global Football Scout Platform (v2)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-KNN-orange.svg)

## 🌐 Live Demo
Check out the live application here: **[AI Football Scout v2](https://ai-football-scout-v2-qv7wvkbozz97iqv5sjcee2.streamlit.app/)**

---

## 📖 About the Project
This project is an advanced, AI-powered football scouting platform designed to assist clubs' analysis departments. It goes beyond simple statistical similarity by introducing a **Club Philosophy KPI Engine**. 

Instead of just asking *"Who plays like Player X?"*, this system answers: *"Among the players who play like Player X, who is the best fit for our specific tactical philosophy?"*

### 🚀 Key Features

*   **Tactical KPI Engine:** Re-ranks scouted players based on a chosen playing style:
    *   *High Press:* Prioritizes Tackles Won, Interceptions, and Aggression.
    *   *Possession:* Prioritizes Assists, Crosses, and drawing fouls.
    *   *Counter Attack:* Prioritizes Goals, Shots, and Offsides (breaking lines).
*   **Machine Learning (K-Nearest Neighbors):** Uses `StandardScaler` and `NearestNeighbors` to find players with the closest statistical footprint to a target player.
*   **Development Trends:** Merges 2024-2025 and 2025-2026 data to calculate "per 90" trends, showing whether a player is improving or regressing.
*   **Advanced Visualizations:** Uses `Plotly` to generate comparative Radar Charts based on percentile ranks, allowing scouts to visually compare the target player and the best alternative.
*   **Proof-of-Concept Physical Module:** A dedicated environment demonstrating how the algorithm processes physical tracking data (Distance covered, Max Speed). Kept purely anonymous to maintain strict data ethics and avoid faking proprietary physical data for real players.
*   **Bilingual UI:** Full support for both English and Turkish via a seamless UI toggle.

---

## 🛠️ Tech Stack
*   **Data Manipulation:** `pandas`, `numpy`
*   **Machine Learning:** `scikit-learn` (KNN, MinMaxScaler, StandardScaler)
*   **Data Visualization:** `plotly` (Radar Charts)
*   **Web Framework:** `streamlit`

---

## 💻 How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/sacelikk/ai-football-scout-v2.git
   cd ai-football-scout-v2
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run scout_dashboard.py
   ```

---

*Disclaimer: The technical event data used in this project is for educational and portfolio purposes. No proprietary or licensed tracking data belonging to real players is exposed.*
