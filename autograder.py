#!/usr/bin/env python3
"""
autograder.py — IEOR E4010 Smart EV Charging Project
Instructor autograder for evaluating student submissions.

Usage:
    python autograder.py --submissions_dir ./submissions --holdout_seed 7777 --verbose
    python autograder.py --example --holdout_seed 7777   # evaluate instructor solution
    python autograder.py --help
"""

import argparse
import json
import os
import pickle
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Physical constants (must match notebooks exactly) ─────────────────────────
VEHICLE_BATTERY_KWH   = 76.0
FLEET_SIZE            = 50
N_FAST                = 10
N_SLOW                = 10
MAX_FAST_KW           = 150.0
MAX_SLOW_KW           = 11.0
BESS_CAPACITY_KWH     = 1000.0
BESS_MAX_DISCHARGE_KW = 1500.0
BESS_MAX_CHARGE_KW    = 500.0
BESS_EFFICIENCY       = 0.92
DC_CHARGER_EFFICIENCY = 0.90
SLOW_CHARGE_EFF       = 0.93
GRID_MAX_KW           = 500.0
TARGET_SOC            = 0.70
DT_HOURS              = 5 / 60
STEPS_PER_EPISODE     = 288


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def find_data_dir():
    """Find the data directory by searching common locations."""
    candidates = [
        Path(__file__).parent / "data",
        Path(__file__).parent.parent,
        Path("data"),
        Path("."),
    ]
    for c in candidates:
        if (c / "nyiso_prices_weather_nyc_2025.csv").exists():
            return c
    return None


def load_data(data_dir=None):
    if data_dir is None:
        data_dir = find_data_dir()
    if data_dir is None:
        raise FileNotFoundError("Cannot find data files. Place them in data/ directory.")

    df_hourly = pd.read_csv(
        data_dir / "nyiso_prices_weather_nyc_2025.csv",
        parse_dates=["timestamp"]
    ).sort_values("timestamp").reset_index(drop=True)

    price_df_5min = pd.read_csv(
        data_dir / "nyiso_rt_lbmp_nyc_2025_5min.csv",
        parse_dates=["timestamp"]
    ).sort_values("timestamp").reset_index(drop=True)

    with open(data_dir / "nyc_fleet_profiles.json") as f:
        fleet_profiles = json.load(f)

    df_hourly["is_weekend"] = (df_hourly["timestamp"].dt.dayofweek >= 5).astype(int)
    df_hourly["hour"] = df_hourly["timestamp"].dt.hour

    return df_hourly, price_df_5min, fleet_profiles


# ─────────────────────────────────────────────────────────────────────────────
# Baseline models (trained fresh so comparison is consistent)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features_baseline(df):
    """Baseline feature engineering used in Notebook 1."""
    df = df.copy()
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"rt_lag_{lag}h"] = df["rt_lbmp_mwh"].shift(lag)
    for window in [6, 24, 168]:
        df[f"rt_roll_mean_{window}h"] = df["rt_lbmp_mwh"].shift(1).rolling(window, min_periods=window // 2).mean()
        df[f"rt_roll_std_{window}h"]  = df["rt_lbmp_mwh"].shift(1).rolling(window, min_periods=window // 2).std()
    df["spread_lag_24h"] = df["da_rt_spread_mwh"].shift(24)
    df["rt_std_lag_1h"]  = df["rt_lbmp_std_mwh"].shift(1)
    return df


BASELINE_FEATURES = (
    [f"rt_lag_{lag}h" for lag in [1, 2, 3, 6, 12, 24, 48, 168]] +
    [f"rt_roll_mean_{w}h" for w in [6, 24, 168]] +
    [f"rt_roll_std_{w}h"  for w in [6, 24, 168]] +
    ["dam_lbmp_mwh", "spread_lag_24h", "rt_std_lag_1h"] +
    ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
)

TARGET = "rt_lbmp_mwh"
SPLIT_DATE = pd.Timestamp("2025-09-01")


def train_baseline_xgb(df_hourly):
    """Train baseline XGBoost on Jan–Aug, evaluate on Sep–Dec."""
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error
    except ImportError:
        print("  xgboost not installed — skipping baseline XGB training")
        return None, np.nan

    df = engineer_features_baseline(df_hourly)
    df_model = df[BASELINE_FEATURES + [TARGET, "timestamp", "da_rt_spread_mwh"]].dropna()

    train_mask = df_model["timestamp"] < SPLIT_DATE
    val_mask   = df_model["timestamp"] >= SPLIT_DATE

    X_train = df_model.loc[train_mask, BASELINE_FEATURES].values
    y_train = df_model.loc[train_mask, TARGET].values
    X_val   = df_model.loc[val_mask, BASELINE_FEATURES].values
    y_val   = df_model.loc[val_mask, TARGET].values

    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        tree_method="hist", early_stopping_rounds=15, eval_metric="mae",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium Environment (identical to Notebook 3)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ev_env_class():
    """Dynamically import or rebuild EVChargingEnv."""
    try:
        import gymnasium as gym
        from scipy.stats import truncnorm

        class EVChargingEnv(gym.Env):
            metadata = {"render_modes": []}
            VEHICLE_BATTERY_KWH = 76.0; N_FAST = 10; N_SLOW = 10
            MAX_FAST_KW = 150.0; MAX_SLOW_KW = 11.0; BESS_CAPACITY = 1000.0
            BESS_MAX_DISCHARGE_KW = 1500.0; BESS_MAX_CHARGE_KW = 500.0
            BESS_EFF = 0.92; DC_EFF = 0.90; SLOW_EFF = 0.93; GRID_MAX_KW = 500.0
            TARGET_SOC = 0.70; DT = 5/60; N_STEPS = 288; DEADLINE_PENALTY = 50.0
            DEMAND_CHARGE_SUMMER = 53.60; DEMAND_CHARGE_ALL = 41.24

            def __init__(self, price_df, fleet_profiles, seed=None,
                         reward_weights=None, n_forecast_steps=0, forecast_fn=None):
                super().__init__()
                self.price_df = price_df.reset_index(drop=True)
                self.fleet_profiles = fleet_profiles
                self.n_forecast_steps = n_forecast_steps
                self.forecast_fn = forecast_fn
                self.rw = reward_weights or {"electricity":1.,"opportunity":1.,"deadline":1.,"peak_demand":1.}
                n_obs = 1 + self.N_FAST*3 + self.N_SLOW*3 + 1 + 1 + 1 + 1 + n_forecast_steps
                self.observation_space = gym.spaces.Box(-1., 1., shape=(n_obs,), dtype=np.float32)
                self.action_space = gym.spaces.Box(0., 1., shape=(21,), dtype=np.float32)
                self.np_random = np.random.default_rng(seed)
                self.price_df["_date"] = pd.to_datetime(self.price_df["timestamp"]).dt.date
                self._episode_days = self.price_df["_date"].unique().tolist()

            def reset(self, seed=None, options=None):
                if seed is not None: self.np_random = np.random.default_rng(seed)
                idx = int(self.np_random.integers(0, len(self._episode_days)))
                self._episode_day = self._episode_days[idx]
                dm = self.price_df["_date"] == self._episode_day
                dd = self.price_df[dm].reset_index(drop=True)
                if len(dd) < self.N_STEPS:
                    pad = self.N_STEPS - len(dd)
                    dd = pd.concat([dd]+[dd.iloc[[-1]]]*pad, ignore_index=True)
                self._prices_kwh = dd["rt_lbmp_kwh"].values[:self.N_STEPS].astype(np.float32)
                ts0 = pd.Timestamp(self._episode_day)
                self._is_weekend = ts0.dayofweek >= 5; self._month = ts0.month
                self._bess_soc = float(self.np_random.uniform(0.30, 0.90))
                self._fast_slots = [None]*self.N_FAST; self._slow_slots = [None]*self.N_SLOW
                self._vehicle_queue = []
                for _ in range(int(self.np_random.integers(3, 12))):
                    v = self._new_vehicle(0)
                    placed = False
                    for i in range(self.N_SLOW):
                        if self._slow_slots[i] is None: self._slow_slots[i]=v; placed=True; break
                    if not placed:
                        for i in range(self.N_FAST):
                            if self._fast_slots[i] is None: self._fast_slots[i]=v; placed=True; break
                    if not placed: self._vehicle_queue.append(v)
                self._step_idx = 0; self._max_grid_draw = 0.; self._total_cost = 0.
                self._total_opp_cost = 0.; self._missed_deadlines = 0; self._prev_grid_draw = 0.
                return self._get_obs(), {}

            def _new_vehicle(self, step):
                sp = self.fleet_profiles["arrival_soc"]
                a=(sp["min"]-sp["mean"])/sp["std"]; b=(sp["max"]-sp["mean"])/sp["std"]
                soc=float(truncnorm.rvs(a,b,loc=sp["mean"],scale=sp["std"],
                                        random_state=int(self.np_random.integers(0,2**30))))
                soc=float(np.clip(soc,sp["min"],sp["max"]))
                hour=(step//12)%24; dk="weekend" if self._is_weekend else "weekday"
                demand=self.fleet_profiles["ride_demand"][dk][hour]
                mxd=max(self.fleet_profiles["ride_demand"]["weekday"]); df=demand/mxd
                dep=int(self.np_random.integers(4,18)) if df>0.6 else int(self.np_random.integers(24,96))
                return {"soc":soc,"depart_step":min(step+dep,self.N_STEPS-1),"arrival_step":step}

            def _process_arrivals(self):
                hour=(self._step_idx//12)%24; dk="weekend" if self._is_weekend else "weekday"
                lam=self.fleet_profiles["depot_arrival_rate"][dk][hour]/12
                for _ in range(int(self.np_random.poisson(lam))):
                    self._vehicle_queue.append(self._new_vehicle(self._step_idx))

            def _assign_from_queue(self):
                for i in range(self.N_SLOW):
                    if self._slow_slots[i] is None and self._vehicle_queue:
                        self._slow_slots[i] = self._vehicle_queue.pop(0)
                for i in range(self.N_FAST):
                    if self._fast_slots[i] is None and self._vehicle_queue:
                        self._fast_slots[i] = self._vehicle_queue.pop(0)

            def _get_obs(self):
                obs = [float(self._bess_soc)*2-1]
                for slot in self._fast_slots:
                    if slot is None: obs.extend([0.,0.,0.])
                    else:
                        sl=max(0,slot["depart_step"]-self._step_idx)
                        obs.extend([1.,float(slot["soc"])*2-1,float(min(sl/96,1.))*2-1])
                for slot in self._slow_slots:
                    if slot is None: obs.extend([0.,0.,0.])
                    else:
                        sl=max(0,slot["depart_step"]-self._step_idx)
                        obs.extend([1.,float(slot["soc"])*2-1,float(min(sl/96,1.))*2-1])
                price=self._prices_kwh[self._step_idx] if self._step_idx<self.N_STEPS else 0.
                obs.extend([float(min(len(self._vehicle_queue)/10,1.))*2-1,
                             float(np.clip(price/0.20,-1.,1.)),
                             float(self._step_idx/self.N_STEPS)*2-1,
                             float(np.clip(self._prev_grid_draw/self.GRID_MAX_KW,0.,1.))*2-1])
                if self.n_forecast_steps > 0:
                    end = min(self._step_idx+self.n_forecast_steps+1, self.N_STEPS)
                    fc  = self._prices_kwh[self._step_idx+1:end]
                    pad = self.n_forecast_steps - len(fc)
                    if len(fc) == 0: fc = np.array([price], dtype=np.float32)
                    fv  = np.clip(np.pad(fc,(0,pad),mode="edge")/0.20,-1.,1.)
                    obs.extend(fv.tolist())
                return np.array(obs, dtype=np.float32)

            def step(self, action):
                action=np.clip(action,0.,1.)
                fr=action[:self.N_FAST]; so=action[self.N_FAST:self.N_FAST+self.N_SLOW]; bcr=action[-1]
                pk=float(self._prices_kwh[self._step_idx])
                bck=min(float(bcr)*self.BESS_MAX_CHARGE_KW,(1.-self._bess_soc)*self.BESS_CAPACITY/self.DT)
                bei=bck*self.DT; bes=bei*self.BESS_EFF
                fck=[0.]*self.N_FAST; bdk=0.; bav=self._bess_soc*self.BESS_CAPACITY/self.DT
                for i,slot in enumerate(self._fast_slots):
                    if slot is None: continue
                    rq=float(fr[i])*self.MAX_FAST_KW
                    rm=max(0.,(self.TARGET_SOC*1.05-slot["soc"])*self.VEHICLE_BATTERY_KWH/self.DT)
                    ak=max(0.,min(rq,rm,bav-bdk)); fck[i]=ak; bdk+=ak
                if bdk>self.BESS_MAX_DISCHARGE_KW:
                    sc=self.BESS_MAX_DISCHARGE_KW/bdk; fck=[k*sc for k in fck]; bdk=self.BESS_MAX_DISCHARGE_KW
                sck=[0.]*self.N_SLOW; tsg=0.
                for i,slot in enumerate(self._slow_slots):
                    if slot is None or float(so[i])<=0.5: continue
                    rm=max(0.,(self.TARGET_SOC*1.05-slot["soc"])*self.VEHICLE_BATTERY_KWH/self.DT)
                    ak=min(self.MAX_SLOW_KW,rm); sck[i]=ak; tsg+=ak
                gdk=bck+tsg
                if gdk>self.GRID_MAX_KW:
                    ex=gdk-self.GRID_MAX_KW; bck=max(0.,bck-ex); bei=bck*self.DT; bes=bei*self.BESS_EFF; gdk=bck+tsg
                beo=bdk*self.DT
                self._bess_soc=float(np.clip(self._bess_soc+(bes-beo)/self.BESS_CAPACITY,0.,1.))
                for i,slot in enumerate(self._fast_slots):
                    if slot is not None and fck[i]>0:
                        self._fast_slots[i]["soc"]=min(1.,slot["soc"]+fck[i]*self.DC_EFF*self.DT/self.VEHICLE_BATTERY_KWH)
                for i,slot in enumerate(self._slow_slots):
                    if slot is not None and sck[i]>0:
                        self._slow_slots[i]["soc"]=min(1.,slot["soc"]+sck[i]*self.SLOW_EFF*self.DT/self.VEHICLE_BATTERY_KWH)
                self._max_grid_draw=max(self._max_grid_draw,gdk); self._prev_grid_draw=gdk
                ec=gdk*pk*self.DT; self._total_cost+=ec
                hour=(self._step_idx//12)%24; dk="weekend" if self._is_weekend else "weekday"
                ocp=self.fleet_profiles["opportunity_cost"][dk][hour]
                nic=sum(1 for s in self._fast_slots+self._slow_slots if s is not None and s["soc"]>=self.TARGET_SOC)
                oc=nic*ocp*self.DT; self._total_opp_cost+=oc
                dp=0.
                for slots in [self._fast_slots, self._slow_slots]:
                    for i in range(len(slots)):
                        if slots[i] is not None and slots[i]["depart_step"]<=self._step_idx:
                            v=slots[i]
                            if v["soc"]<self.TARGET_SOC: dp+=self.DEADLINE_PENALTY*(self.TARGET_SOC-v["soc"])*100; self._missed_deadlines+=1
                            slots[i]=None
                self._process_arrivals(); self._assign_from_queue()
                reward=(-self.rw["electricity"]*ec-self.rw["opportunity"]*oc-self.rw["deadline"]*dp)
                self._step_idx+=1; terminated=self._step_idx>=self.N_STEPS
                if terminated:
                    rate=self.DEMAND_CHARGE_SUMMER if self._month in[6,7,8,9] else self.DEMAND_CHARGE_ALL
                    dc=self._max_grid_draw*rate; reward-=self.rw["peak_demand"]*dc
                info={"elec_cost":ec,"opp_cost":oc,"deadline_penalty":dp,"bess_soc":self._bess_soc,
                      "grid_draw_kw":gdk,"n_idle_charged":nic}
                if terminated:
                    rate=self.DEMAND_CHARGE_SUMMER if self._month in[6,7,8,9] else self.DEMAND_CHARGE_ALL
                    info["total_cost"]=self._total_cost; info["missed_deadlines"]=self._missed_deadlines
                    info["max_grid_draw_kw"]=self._max_grid_draw; info["demand_charge"]=self._max_grid_draw*rate
                obs=self._get_obs() if not terminated else np.zeros(self.observation_space.shape,dtype=np.float32)
                return obs,reward,terminated,False,info

        return EVChargingEnv

    except ImportError as e:
        raise RuntimeError(f"Cannot build EVChargingEnv: {e}. Install gymnasium.")


# ─────────────────────────────────────────────────────────────────────────────
# RL Heuristic baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_heuristic_baseline(EVClass, price_df, fleet_profiles, n_episodes=50, seed=42):
    """
    Run the urgency-priority heuristic (best of the three heuristics from Notebook 3).
    Returns mean episode cost.
    """
    env = EVClass(price_df, fleet_profiles, seed=seed)
    costs = []
    rng = np.random.default_rng(seed)

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 100000)))
        for _ in range(env.N_STEPS):
            # Urgency-priority heuristic
            action = np.zeros(21, dtype=np.float32)
            for i, slot in enumerate(env._fast_slots):
                if slot is not None and env._bess_soc > 0.15:
                    steps_left = max(1, slot["depart_step"] - env._step_idx)
                    soc_gap    = max(0.0, env.TARGET_SOC - slot["soc"])
                    action[i]  = min(1.0, (soc_gap / steps_left) * 500)
            for i, slot in enumerate(env._slow_slots):
                if slot is not None and slot["soc"] < env.TARGET_SOC:
                    action[env.N_FAST + i] = 1.0
            hour = (env._step_idx // 12) % 24
            if (hour < 8 or hour >= 22) and env._bess_soc < 0.85:
                action[-1] = 0.8
            elif env._bess_soc < 0.20:
                action[-1] = 1.0

            _, _, done, _, info = env.step(action)
            if done:
                costs.append(info.get("total_cost", np.nan))
                break

    return float(np.nanmean(costs))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ml_submission(submission_dir, df_hourly, baseline_mae, verbose=False):
    """
    Evaluate a student's ML model.
    Returns: (score 0-100, mae, details_dict)
    """
    result = {"component": "ML", "score": 0, "mae": np.nan, "status": "FAIL", "error": None}

    ml_path = Path(submission_dir) / "ml_model.pkl"
    if not ml_path.exists():
        result["error"] = "ml_model.pkl not found"
        return result

    try:
        with open(ml_path, "rb") as f:
            ml_obj = pickle.load(f)

        feature_fn   = ml_obj.get("feature_fn", None)
        feature_cols = ml_obj.get("feature_cols", [])

        if feature_fn is not None:
            df_feats = feature_fn(df_hourly)
        else:
            df_feats = engineer_features_baseline(df_hourly)
            feature_cols = BASELINE_FEATURES

        val_mask = df_feats["timestamp"] >= SPLIT_DATE
        df_val   = df_feats[val_mask].dropna(subset=feature_cols)

        if len(df_val) == 0:
            result["error"] = "No valid validation samples after feature engineering"
            return result

        from sklearn.metrics import mean_absolute_error
        X_val = df_val[feature_cols].values
        y_val = df_val[TARGET].values
        preds = ml_obj["model"].predict(X_val)
        mae   = float(mean_absolute_error(y_val, preds))

        result["mae"] = mae

        # Score: 100 if at or below baseline, scales linearly down to 0 at 2× baseline
        if mae <= baseline_mae:
            score = 100.0
        elif mae >= 2 * baseline_mae:
            score = 0.0
        else:
            score = 100.0 * (2 * baseline_mae - mae) / baseline_mae

        result["score"]  = round(score, 2)
        result["status"] = "PASS" if mae < baseline_mae else "PARTIAL"

        if verbose:
            print(f"    ML MAE: {mae:.3f} $/MWh  (baseline: {baseline_mae:.3f})  score={score:.1f}")

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            traceback.print_exc()

    return result


def evaluate_dl_submission(submission_dir, df_hourly, verbose=False):
    """
    Evaluate a student's DL model.
    Returns: (score 0-100, mae, details_dict)
    """
    result = {"component": "DL", "score": 0, "mae": np.nan, "status": "FAIL", "error": None}

    dl_path = Path(submission_dir) / "lstm_model.pth"
    if not dl_path.exists():
        result["error"] = "lstm_model.pth not found"
        return result

    try:
        import torch
        import torch.nn as nn
        from sklearn.metrics import mean_absolute_error

        dl_obj = torch.load(dl_path, map_location="cpu", weights_only=False)

        seq_features  = dl_obj["seq_features"]
        seq_len       = dl_obj["seq_len"]
        scaler_mean   = pd.Series(dl_obj["scaler_mean"])
        scaler_std    = pd.Series(dl_obj["scaler_std"])
        target_mean   = float(dl_obj.get("target_mean", 0.0))
        target_std    = float(dl_obj.get("target_std",  1.0))
        model_config  = dl_obj.get("model_config", {})
        model_class   = dl_obj.get("model_class", "ImprovedForecaster")

        in_size  = model_config.get("input_size", len(seq_features))
        hid_size = model_config.get("hidden_size", 128)
        n_layers = model_config.get("num_layers", 2)
        dropout  = model_config.get("dropout", 0.15)

        class _GRUAttention(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.hidden_size = hidden_size
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
                self.attention_fc = nn.Linear(hidden_size, 1, bias=False)
                self.fc1 = nn.Linear(hidden_size, 64); self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout); self.fc2 = nn.Linear(64, 1)
            def forward(self, x):
                out, _ = self.gru(x)
                sc = self.attention_fc(out) / (self.hidden_size ** 0.5)
                w  = torch.softmax(sc, dim=1); ctx = (w * out).sum(dim=1)
                return self.fc2(self.relu(self.fc1(self.dropout(ctx)))).squeeze(-1)

        class _LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.hidden_size = hidden_size
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                     dropout=dropout if num_layers > 1 else 0.0)
                self.dropout = nn.Dropout(dropout); self.fc = nn.Linear(hidden_size, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)

        if "LSTM" in model_class:
            model = _LSTMModel(in_size, hid_size, n_layers, dropout)
        else:
            model = _GRUAttention(in_size, hid_size, n_layers, dropout)

        model.load_state_dict(dl_obj["model_state_dict"])
        model.eval()

        # Prepare sequences
        df = df_hourly.copy()
        df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(float)
        val_normed = (df[seq_features] - scaler_mean) / scaler_std
        t_normed   = (df["rt_lbmp_mwh"] - target_mean) / target_std

        val_start = df[df["timestamp"] >= SPLIT_DATE].index[0]
        start = max(0, val_start - seq_len)
        feat_arr   = val_normed.values[start:].astype(np.float32)
        target_arr = t_normed.values[start:].astype(np.float32)

        X_seq, y_seq = [], []
        for i in range(len(feat_arr) - seq_len):
            X_seq.append(feat_arr[i : i + seq_len])
            y_seq.append(target_arr[i + seq_len])

        n_pre   = val_start - start
        X_val   = np.stack(X_seq[n_pre:])
        y_seq   = np.array(y_seq)
        y_true  = df["rt_lbmp_mwh"].values[val_start : val_start + len(y_seq) - n_pre]

        with torch.no_grad():
            pred_norm = model(torch.from_numpy(X_val)).numpy()
        pred_raw  = pred_norm * target_std + target_mean
        n_compare = min(len(pred_raw), len(y_true))

        mae = float(mean_absolute_error(y_true[:n_compare], pred_raw[:n_compare]))
        result["mae"] = mae

        # Reference: a naive persistence baseline MAE (predict yesterday's same hour)
        reference_mae = 8.0  # approximate for NYC 2025
        if mae <= reference_mae:
            score = 100.0
        elif mae >= 2 * reference_mae:
            score = 0.0
        else:
            score = 100.0 * (2 * reference_mae - mae) / reference_mae

        result["score"]  = round(score, 2)
        result["status"] = "PASS" if mae < reference_mae * 1.5 else "PARTIAL"

        if verbose:
            print(f"    DL MAE: {mae:.3f} $/MWh  score={score:.1f}")

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            traceback.print_exc()

    return result


def evaluate_rl_submission(submission_dir, price_df, fleet_profiles,
                            holdout_seed, n_episodes, heuristic_cost,
                            EVClass, verbose=False):
    """
    Evaluate a student's RL agent.
    Returns: (score 0-100, mean_cost, details_dict)
    """
    result = {"component": "RL", "score": 0, "mean_cost": np.nan,
              "missed_deadlines": np.nan, "status": "FAIL", "error": None}

    agent_path = Path(submission_dir) / "rl_agent.zip"
    if not agent_path.exists():
        # Also try without .zip
        agent_path_no_zip = Path(submission_dir) / "rl_agent"
        if not agent_path_no_zip.exists():
            result["error"] = "rl_agent.zip not found"
            return result
        agent_path = agent_path_no_zip

    try:
        from stable_baselines3 import PPO

        config_path = Path(submission_dir) / "rl_config.json"
        n_fc = 0
        if config_path.exists():
            with open(config_path) as f:
                rl_config = json.load(f)
            n_fc = rl_config.get("n_forecast_steps", 0)

        eval_env = EVClass(price_df, fleet_profiles, seed=holdout_seed,
                           n_forecast_steps=n_fc)
        model = PPO.load(str(agent_path).replace(".zip", ""), env=eval_env)

        rng   = np.random.default_rng(holdout_seed)
        costs = []
        missed_list = []

        for ep in range(n_episodes):
            ep_seed = int(rng.integers(0, 1_000_000))
            obs, _ = eval_env.reset(seed=ep_seed)
            for _ in range(288):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, info = eval_env.step(action)
                if done:
                    costs.append(info.get("total_cost", np.nan))
                    missed_list.append(info.get("missed_deadlines", np.nan))
                    break

        mean_cost   = float(np.nanmean(costs))
        mean_missed = float(np.nanmean(missed_list))
        result["mean_cost"]         = mean_cost
        result["missed_deadlines"]  = mean_missed

        # Score: 100 if beats heuristic by >10%, scales down to 0 at heuristic + 50%
        if mean_cost <= heuristic_cost * 0.90:
            score = 100.0
        elif mean_cost >= heuristic_cost * 1.50:
            score = 0.0
        else:
            # Linear between [heuristic*0.9, heuristic*1.5]
            score = 100.0 * (heuristic_cost * 1.5 - mean_cost) / (heuristic_cost * 0.6)

        result["score"]  = round(max(0, score), 2)
        result["status"] = "PASS" if mean_cost < heuristic_cost else "PARTIAL"

        if verbose:
            print(f"    RL mean cost: ${mean_cost:.2f}  (heuristic: ${heuristic_cost:.2f})  score={score:.1f}")
            print(f"    RL missed departures: {mean_missed:.1f}")

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            traceback.print_exc()

    return result


def evaluate_student(student_dir, df_hourly, price_df_5min, fleet_profiles,
                     baseline_xgb_mae, heuristic_cost, EVClass,
                     holdout_seed, n_rl_episodes, verbose):
    """Evaluate all three components for one student submission."""
    student_name = Path(student_dir).name

    # Try to load student_info.json for name/UNI
    info_path = Path(student_dir) / "student_info.json"
    uni = student_name
    if info_path.exists():
        with open(info_path) as f:
            si = json.load(f)
        uni = si.get("uni", student_name)

    if verbose:
        print(f"\n  Evaluating: {student_name} ({uni})")

    ml_result = evaluate_ml_submission(student_dir, df_hourly, baseline_xgb_mae, verbose)
    dl_result = evaluate_dl_submission(student_dir, df_hourly, verbose)
    rl_result = evaluate_rl_submission(
        student_dir, price_df_5min, fleet_profiles,
        holdout_seed, n_rl_episodes, heuristic_cost, EVClass, verbose
    )

    # Combined score: 0.25 ML + 0.25 DL + 0.50 RL
    combined = (
        0.25 * ml_result["score"] +
        0.25 * dl_result["score"] +
        0.50 * rl_result["score"]
    )

    return {
        "student":          student_name,
        "uni":              uni,
        "ml_score":         ml_result["score"],
        "ml_mae":           ml_result["mae"],
        "ml_status":        ml_result["status"],
        "dl_score":         dl_result["score"],
        "dl_mae":           dl_result["mae"],
        "dl_status":        dl_result["status"],
        "rl_score":         rl_result["score"],
        "rl_mean_cost":     rl_result["mean_cost"],
        "rl_missed_dep":    rl_result["missed_deadlines"],
        "rl_status":        rl_result["status"],
        "combined_score":   round(combined, 2),
        "ml_error":         ml_result.get("error"),
        "dl_error":         dl_result.get("error"),
        "rl_error":         rl_result.get("error"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IEOR E4010 Autograder — Smart EV Charging Project"
    )
    parser.add_argument(
        "--submissions_dir", type=str, default="./submissions",
        help="Directory containing one subdirectory per student submission"
    )
    parser.add_argument(
        "--holdout_seed", type=int, default=7777,
        help="Random seed for RL evaluation (keep fixed across all submissions)"
    )
    parser.add_argument(
        "--n_rl_episodes", type=int, default=50,
        help="Number of RL evaluation episodes per student"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to data directory (default: auto-detect)"
    )
    parser.add_argument(
        "--output", type=str, default="leaderboard.csv",
        help="Output CSV file for leaderboard"
    )
    parser.add_argument(
        "--example", action="store_true",
        help="Evaluate the instructor's suggested-solution models (in ./submission/)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed per-student results"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("IEOR E4010 — Smart EV Charging Project Autograder")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    data_dir = Path(args.data_dir) if args.data_dir else None
    df_hourly, price_df_5min, fleet_profiles = load_data(data_dir)
    print(f"  Hourly: {len(df_hourly)} rows, 5-min: {len(price_df_5min)} rows")

    # ── Train baseline models ─────────────────────────────────────────────────
    print("\nTraining baseline XGBoost model...")
    try:
        _, baseline_xgb_mae = train_baseline_xgb(df_hourly)
        print(f"  Baseline XGBoost MAE: {baseline_xgb_mae:.3f} $/MWh")
    except Exception as e:
        print(f"  WARNING: Could not train baseline XGB: {e}")
        baseline_xgb_mae = 6.0  # fallback

    # ── Build RL environment class ────────────────────────────────────────────
    print("\nBuilding RL environment...")
    try:
        EVClass = _build_ev_env_class()
        print("  EVChargingEnv built successfully.")
    except Exception as e:
        print(f"  ERROR: Cannot build RL environment: {e}")
        EVClass = None

    # ── Compute heuristic baseline ────────────────────────────────────────────
    heuristic_cost = np.nan
    if EVClass is not None:
        print(f"\nRunning heuristic baseline ({args.n_rl_episodes} episodes, seed={args.holdout_seed})...")
        try:
            heuristic_cost = run_heuristic_baseline(
                EVClass, price_df_5min, fleet_profiles,
                n_episodes=args.n_rl_episodes, seed=args.holdout_seed
            )
            print(f"  Urgency-Priority heuristic mean cost: ${heuristic_cost:.2f}")
        except Exception as e:
            print(f"  WARNING: Heuristic baseline failed: {e}")
            heuristic_cost = 2000.0  # fallback

    # ── Determine submission directories ─────────────────────────────────────
    if args.example:
        print("\nEvaluating instructor example solution...")
        submission_dirs = [Path("./submission")]
        if not submission_dirs[0].exists():
            print("  ERROR: ./submission/ not found. Run notebooks 1-3 first.")
            sys.exit(1)
    else:
        submissions_root = Path(args.submissions_dir)
        if not submissions_root.exists():
            print(f"\nERROR: Submissions directory not found: {args.submissions_dir}")
            print("Create a directory with one subdirectory per student.")
            sys.exit(1)

        submission_dirs = [
            d for d in submissions_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        if len(submission_dirs) == 0:
            print(f"\nNo student submissions found in {args.submissions_dir}")
            sys.exit(0)

        print(f"\nFound {len(submission_dirs)} student submission(s).")

    # ── Evaluate each submission ──────────────────────────────────────────────
    leaderboard = []

    for student_dir in sorted(submission_dirs):
        try:
            row = evaluate_student(
                student_dir=student_dir,
                df_hourly=df_hourly,
                price_df_5min=price_df_5min,
                fleet_profiles=fleet_profiles,
                baseline_xgb_mae=baseline_xgb_mae,
                heuristic_cost=heuristic_cost,
                EVClass=EVClass,
                holdout_seed=args.holdout_seed,
                n_rl_episodes=args.n_rl_episodes,
                verbose=args.verbose,
            )
            leaderboard.append(row)
        except Exception as e:
            print(f"  ERROR evaluating {student_dir.name}: {e}")
            if args.verbose:
                traceback.print_exc()

    # ── Output leaderboard ────────────────────────────────────────────────────
    if leaderboard:
        df_lb = pd.DataFrame(leaderboard).sort_values(
            "combined_score", ascending=False
        ).reset_index(drop=True)
        df_lb.index += 1  # 1-based ranking

        print("\n" + "=" * 70)
        print("LEADERBOARD")
        print("=" * 70)
        display_cols = [
            "student", "uni",
            "ml_score", "ml_mae",
            "dl_score", "dl_mae",
            "rl_score", "rl_mean_cost",
            "combined_score",
        ]
        print(df_lb[display_cols].to_string(index=True))

        df_lb.to_csv(args.output, index_label="rank")
        print(f"\nLeaderboard saved to {args.output}")

        print(f"\nBaseline XGBoost MAE: {baseline_xgb_mae:.3f} $/MWh")
        print(f"Heuristic mean cost:  ${heuristic_cost:.2f}")
        print(f"holdout_seed:         {args.holdout_seed}")

    else:
        print("\nNo results to report.")


if __name__ == "__main__":
    main()
