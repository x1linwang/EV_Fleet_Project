# Smart EV Fleet Charging — IEOR E4010
## AI for Operations Research and Financial Engineering
### Columbia University, Spring 2026

---

## Project Overview

This project explores AI-driven optimization for a 50-vehicle electric taxi fleet operating in New York City. Students apply machine learning, deep learning, and reinforcement learning to minimize the total cost of charging the fleet while meeting service-level constraints.

The project is grounded in real data:
- **NYISO real-time electricity prices** for NYC (2025, 5-min resolution)
- **Weather data** co-located with price data (temperature, humidity, solar, wind)
- **Fleet arrival/departure profiles** calibrated from 243 million NYC TLC (Uber/Lyft) trips

The physical system modeled:
- 50 Zeekr Ojai LFP vehicles (76 kWh battery each)
- 10 DC fast chargers (up to 150 kW each), powered from a 1 MWh BESS
- 10 Level 2 AC slow chargers (11 kW each), powered directly from the grid
- 500 kW grid connection cap, with monthly peak demand charges

---

## Project Structure

```
Smart_EV_Charging/
├── 01_ml_forecasting.ipynb     # Notebook 1: XGBoost price forecasting
├── 02_dl_forecasting.ipynb     # Notebook 2: LSTM/GRU deep learning forecasting
├── 03_rl_charging.ipynb        # Notebook 3: PPO reinforcement learning for charging
├── 04_submission.ipynb         # Notebook 4: Validate and package submission
├── autograder.py               # Instructor autograder script
├── data/                       # Data files (see data/README.md)
│   ├── nyiso_prices_weather_nyc_2025.csv
│   ├── nyiso_rt_lbmp_nyc_2025_5min.csv
│   └── nyc_fleet_profiles.json
├── requirements.txt
└── README.md
```

---

## How to Run

### Option A: Google Colab (Recommended for students)

1. Open each notebook in Colab via the GitHub link or by uploading directly.
2. The setup cell in each notebook will install all dependencies automatically.
3. Data files will be loaded from the `data/` directory (relative to the notebook).
   - If running on Colab, the setup cell copies data from the GitHub repo or parent directory.
4. Run all cells in order (Runtime → Run all).

### Option B: Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_REPO/EV_Fleet_Project.git
cd EV_Fleet_Project/Smart_EV_Charging

# Install dependencies
pip install -r requirements.txt

# Verify data files are present
ls data/

# Launch Jupyter
jupyter notebook
```

---

## Data Files

| File | Description | Size |
|------|-------------|------|
| `nyiso_prices_weather_nyc_2025.csv` | Hourly DA/RT LBMP prices + 25 weather features, 8760 rows | ~3 MB |
| `nyiso_rt_lbmp_nyc_2025_5min.csv` | 5-minute real-time LBMP prices, ~105k rows | ~5 MB |
| `nyc_fleet_profiles.json` | Fleet arrival rates, SoC distributions, opportunity costs | <1 MB |

See `data/README.md` for column-level documentation.

---

## Notebooks

### Notebook 1: ML Price Forecasting (`01_ml_forecasting.ipynb`)
- Exploratory data analysis of NYISO prices and weather
- Lag and rolling feature engineering
- XGBoost baseline and custom feature engineering (TODO section)
- Evaluation on price-spike hours

### Notebook 2: DL Price Forecasting (`02_dl_forecasting.ipynb`)
- Sequence preparation for LSTM (168-hour lookback window)
- Baseline LSTM model
- Attention-augmented GRU (TODO section)
- Side-by-side ML vs DL comparison

### Notebook 3: RL Charging Optimization (`03_rl_charging.ipynb`)
- Full `EVChargingEnv` Gymnasium environment
- Three heuristic baselines (FIFO-max, price-aware, urgency-priority)
- LP perfect-foresight lower bound
- PPO training with Stable-Baselines3
- Reward shaping (TODO section)
- Price forecast integration (TODO section)

### Notebook 4: Submission Validation (`04_submission.ipynb`)
- Load and validate all three saved models
- Run evaluation and print metrics
- Generate `submission/student_info.json`

---

## Grading

Scores are computed by `autograder.py`:

| Component | Weight | Metric |
|-----------|--------|--------|
| ML Forecasting | 25% | MAE on RT price (vs XGBoost baseline) |
| DL Forecasting | 25% | MAE on RT price (vs LSTM baseline) |
| RL Charging | 50% | Mean episode cost (vs heuristic baselines) |

Run the autograder:
```bash
python autograder.py --submissions_dir ./submissions --holdout_seed 7777 --verbose
python autograder.py --example --holdout_seed 7777   # evaluate instructor solution
```

---

## Physical Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Battery capacity | 76 kWh | Zeekr Ojai LFP |
| Fleet size | 50 vehicles | |
| Fast chargers | 10 × 150 kW DC | Powered from BESS |
| Slow chargers | 10 × 11 kW AC | Powered from grid |
| BESS capacity | 1,000 kWh | |
| BESS max discharge | 1,500 kW | Powers all fast chargers |
| BESS max charge | 500 kW | Grid-limited |
| BESS efficiency | 92% round-trip | |
| Grid connection | 500 kW max | |
| Target SoC | 70% | Per-vehicle charging goal |
| Timestep | 5 minutes | 288 steps per episode |

---

## License

For educational use only. Data sourced from NYISO (public) and NYC TLC (public).
