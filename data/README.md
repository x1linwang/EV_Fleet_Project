# Data Directory

Place the three data files here for local runs. On Google Colab, the notebook setup cell will resolve paths automatically.

## Files Required

### `nyiso_prices_weather_nyc_2025.csv`
Hourly DA/RT LBMP electricity prices for NYC (NYISO Zone J) plus weather observations, 2025.
~8760 rows (one per hour).

**Key columns:**
| Column | Description | Units |
|--------|-------------|-------|
| `timestamp` | Hourly timestamp | UTC |
| `dam_lbmp_mwh` | Day-ahead marginal price | $/MWh |
| `dam_lbmp_kwh` | Day-ahead marginal price | $/kWh |
| `rt_lbmp_mwh` | Real-time marginal price (hourly avg of 5-min) | $/MWh |
| `rt_lbmp_kwh` | Real-time marginal price | $/kWh |
| `rt_lbmp_std_mwh` | Std dev of 5-min RT prices within hour | $/MWh |
| `da_rt_spread_mwh` | DA minus RT spread | $/MWh |
| `hour_sin`, `hour_cos` | Cyclical hour encoding | — |
| `dow_sin`, `dow_cos` | Cyclical day-of-week encoding | — |
| `is_weekend` | 1 if Saturday or Sunday | — |
| `hour` | Hour of day (0–23) | — |
| `day_of_week` | Day of week (0=Mon) | — |
| `month` | Month (1–12) | — |
| `temperature_2m` | Air temperature at 2m | °C |
| `apparent_temperature` | Feels-like temperature | °C |
| `relative_humidity_2m` | Relative humidity | % |
| `dew_point_2m` | Dew point | °C |
| `wind_speed_10m` | Wind speed at 10m | km/h |
| `wind_gusts_10m` | Wind gusts at 10m | km/h |
| `cloud_cover` | Cloud cover fraction | % |
| `shortwave_radiation` | Solar radiation | W/m² |
| `precipitation` | Precipitation | mm |
| `pressure_msl` | Sea-level pressure | hPa |
| `is_day` | 1 if daytime | — |
| `cooling_degree_hours` | CDH above 18°C | °C·h |
| `heating_degree_hours` | HDH below 18°C | °C·h |
| `is_hot`, `is_very_hot`, `is_cold`, `is_very_cold` | Temperature regime flags | — |
| `humidity_discomfort` | Heat index discomfort | — |
| `solar_generation_proxy` | Solar supply proxy | — |
| `wind_power_proxy` | Wind supply proxy | — |
| `temp_change_1h`, `temp_change_3h` | Temperature change | °C |
| `temp_rolling_6h_mean`, `temp_rolling_24h_mean` | Rolling temperature | °C |
| `temp_rolling_24h_max` | Rolling max temperature | °C |
| `temp_anomaly` | Deviation from seasonal norm | °C |

### `nyiso_rt_lbmp_nyc_2025_5min.csv`
5-minute real-time LBMP prices for NYC (NYISO Zone J), 2025.
~105,120 rows (one per 5-minute interval).

**Columns:**
| Column | Description | Units |
|--------|-------------|-------|
| `timestamp` | 5-minute timestamp | UTC |
| `rt_lbmp_mwh` | Real-time marginal price | $/MWh |
| `rt_lbmp_kwh` | Real-time marginal price | $/kWh |
| `rt_losses_mwh` | Losses component | $/MWh |
| `rt_congestion_mwh` | Congestion component | $/MWh |

### `nyc_fleet_profiles.json`
Fleet statistical profiles calibrated from 243M NYC TLC (Uber/Lyft) trips.

**Top-level keys:**
- `vehicle_specs`: Battery size, fleet size, efficiency
- `depot_arrival_rate`: Poisson lambda (vehicles/hour) by hour-of-day, weekday/weekend
- `arrival_soc`: Distribution of vehicle SoC on depot arrival (mean, std, min, max)
- `opportunity_cost`: Revenue loss per vehicle-hour idle, by hour-of-day, weekday/weekend
- `trip_distance`: Trip distance distribution statistics
- `ride_demand`: Ride demand index by hour-of-day, weekday/weekend

## Data Sources
- NYISO prices: [NYISO Public Data](https://www.nyiso.com/public-data-repository) (Zone J / NYC)
- Weather: Open-Meteo API, ERA5 reanalysis, NYC coordinates (40.71°N, 74.01°W)
- Fleet profiles: NYC Taxi & Limousine Commission HVFHV Trip Data (public)
