Here’s a walkthrough of what our script does, step by step:
1. Setup & Configuration
- We set three key constants at the top:
  TARGET_YEAR    # e.g. 2025
  TARGET_ROUND   # e.g. 13 for Belgian GP
  CIRCUIT_NAME   # e.g. "Spa‑Francorchamps"
- FastF1’s cache is enabled so once data’s downloaded it won’t be fetched again.
2. Data Collection (collect_training_data)
- 2024 Reference Race: pull the 2024 session for our chosen round for baseline data.
- 2025 Pre‑Race Sessions: look up the 2025 calendar, find all rounds completed before our target, and download each race + quali.
- Each record includes grid & finish positions, points scored, best qualifying lap, pit stops, status, plus average air/track temps for both quali and race.
- We accumulate every driver‑race into a DataFrame, then sum each driver’s and team’s total points to date.
3. Feature Engineering (_feature_engineering)
- Compute races_so_far, team_changed, rookie flag.
- Compute quali_vs_grid and delta_to_pole (gap to that weekend’s pole lap).
- Compute team_mean_points_so_far (expanding mean of team’s points).
- Compute driver_dnf_rate and team_dnf_rate (fraction of non‑finishes).
- Map DriverSeasonPoints and TeamSeasonPoints, then normalize them to [0,1].
- Fill any missing weather values (RaceAirTemp, TrackTemp, QualiAirTemp, TrackTemp) with the median.
4. Model Training (train)
- Labels: is_winner for classification; 1/finish_position for regression.
- Sample weights: 2025 rows ×5, further scaled by (1 + 2·driver_points_norm + 1·team_points_norm).
- RandomForestClassifier: 5‑fold CV to report accuracy, then CalibratedClassifierCV(method="sigmoid") on full set for win probabilities.
- RandomForestRegressor: fit on the same features to predict 1/position; invert to raw position to compute MAE.
5. Making the Prediction (predict_british_2025)
- Automatically load 2025 Quali for TARGET_ROUND; extract each driver’s best Q3/Q2/Q1 lap.
- Build feature vector exactly as in training.
- win_probability from the calibrated classifier.
- predicted_finish = 1 / regressor_output.
- score = win_probability * (1 + 5 * driver_points_norm) / predicted_finish.
- Sort by score, assign predicted_position, format values, and print the finishing order plus a “pole‑based” fastest‑lap estimate.
6. How to Run & Verify
- Change TARGET_YEAR, TARGET_ROUND, CIRCUIT_NAME to target any race.
- Run python belgian_gp_predict.py.
- If Quali data is live you’ll see the order; if not, you’ll get a “Quali not available” notice—wait a few minutes and retry.
