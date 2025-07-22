import fastf1
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, accuracy_score
import warnings

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("f1_cache")

TARGET_YEAR = 2025
TARGET_ROUND = 12          # British GP
CIRCUIT_NAME = "Silverstone"
WEIGHT_FACTOR_2025 = 5.0
N_FOLDS = 5

ALLOWED_2025_DRIVERS = [
    "George Russell", "Andrea Kimi Antonelli", "Charles Leclerc", "Lewis Hamilton",
    "Max Verstappen", "Yuki Tsunoda", "Lando Norris", "Oscar Piastri",
    "Fernando Alonso", "Lance Stroll", "Pierre Gasly", "Jack Doohan",
    "Esteban Ocon", "Oliver Bearman", "Liam Lawson", "Alexander Albon",
    "Carlos Sainz", "Nico Hulkenberg", "Gabriel Bortoleto"
]


class BritishPredictor:
    def __init__(self):
        self.driver_enc = LabelEncoder()
        self.team_enc = LabelEncoder()
        self.circuit_enc = LabelEncoder()
        self.clf = None
        self.reg = RandomForestRegressor(n_estimators=400, random_state=42)
        self.data = pd.DataFrame()
        self.full_driver_points = {}
        self.full_team_points = {}
        self.is_trained = False

    # ------------------- Yardımcılar -------------------
    def _latest_completed_round(self, schedule: pd.DataFrame) -> int:
        latest = 0
        now = pd.Timestamp.now(tz=schedule.iloc[0]["Session1Date"].tz)
        for _, ev in schedule.iterrows():
            if now > ev["Session3Date"]:
                latest = max(latest, ev["RoundNumber"])
        return latest

    def _load_session_pair(self, year: int, rnd: int):
        race = fastf1.get_session(year, rnd, "R"); race.load()
        quali = fastf1.get_session(year, rnd, "Q"); quali.load()
        return race, quali

    def _extract_quali_time_position(self, dq):
        if dq is None or dq.empty:
            return 0.0, 20
        q_times = [dq.iloc[0].get("Q3"), dq.iloc[0].get("Q2"), dq.iloc[0].get("Q1")]
        best_q = 0.0
        for qt in q_times:
            if pd.notna(qt):
                best_q = qt.total_seconds() if hasattr(qt, "total_seconds") else 0.0
                break
        quali_pos = dq.iloc[0].get("Position", 20)
        return best_q, quali_pos

    def _session_weather_means(self, session):
        try:
            w = session.weather_data
            if w is None or w.empty:
                return np.nan, np.nan
            return float(w['AirTemp'].mean()), float(w['TrackTemp'].mean())
        except Exception:
            return np.nan, np.nan

    def _row_dict(self, year, rnd, circuit, r_row, quali_pos, quali_time,
                  race_air, race_track, quali_air, quali_track):
        return {
            "year": year,
            "round": rnd,
            "circuit": circuit,
            "driver": r_row["Abbreviation"],
            "driver_name": r_row["FullName"],
            "team": r_row["TeamName"],
            "grid_position": r_row["GridPosition"],
            "finish_position": r_row["Position"],
            "points_in_race": r_row["Points"],
            "quali_position": quali_pos,
            "quali_time": quali_time,
            "is_winner": 1 if r_row["Position"] == 1 else 0,
            "pit_stops": r_row.get("NumberOfPitStops", 0),
            "status": r_row.get("Status", ""),
            "RaceAirTemp": race_air,
            "RaceTrackTemp": race_track,
            "QualiAirTemp": quali_air,
            "QualiTrackTemp": quali_track
        }

    # ------------------- Veri Toplama -------------------
    def collect_training_data(self):
        records = []

        # 1) 2024 British GP
        try:
            race_2024, quali_2024 = self._load_session_pair(2024, TARGET_ROUND)
            race_air, race_track = self._session_weather_means(race_2024)
            quali_air, quali_track = self._session_weather_means(quali_2024)
            res = race_2024.results; qres = quali_2024.results
            for _, r in res.iterrows():
                dq = qres[qres["Abbreviation"] == r["Abbreviation"]]
                best_q, quali_pos = self._extract_quali_time_position(dq)
                records.append(self._row_dict(
                    2024, TARGET_ROUND, race_2024.event["Location"], r,
                    quali_pos, best_q, race_air, race_track, quali_air, quali_track
                ))
            print("✔️ 2024 British GP eklendi.")
        except Exception as e:
            print("2024 British GP yüklenemedi:", e)

        # 2) 2025 British öncesi tüm yarışlar
        try:
            schedule_2025 = fastf1.get_event_schedule(TARGET_YEAR)
            latest_round = self._latest_completed_round(schedule_2025)
        except Exception as e:
            print("2025 takvim alınamadı:", e); latest_round = 0

        cutoff = TARGET_ROUND - 1
        usable_max = min(latest_round, cutoff)
        all_2025_rows = []

        if usable_max > 0:
            for rnd in range(1, usable_max + 1):
                try:
                    race, quali = self._load_session_pair(TARGET_YEAR, rnd)
                except Exception as e:
                    print(f"2025 Round {rnd} yüklenemedi:", e); continue
                race_air, race_track = self._session_weather_means(race)
                quali_air, quali_track = self._session_weather_means(quali)
                res = race.results; qres = quali.results
                if res.empty:
                    continue
                for _, r in res.iterrows():
                    if r["FullName"] not in ALLOWED_2025_DRIVERS:
                        continue
                    dq = qres[qres["Abbreviation"] == r["Abbreviation"]]
                    best_q, quali_pos = self._extract_quali_time_position(dq)
                    all_2025_rows.append(self._row_dict(
                        TARGET_YEAR, rnd, race.event["Location"], r,
                        quali_pos, best_q, race_air, race_track, quali_air, quali_track
                    ))
                print(f"✔️ 2025 Round {rnd} eklendi.")

            all_df = pd.DataFrame(all_2025_rows)
            self.full_driver_points = all_df.groupby("driver")["points_in_race"].sum().to_dict()
            self.full_team_points = all_df.groupby("team")["points_in_race"].sum().to_dict()

            records.extend(all_2025_rows)
        else:
            print("British öncesi 2025 tamamlanmış yarış yok.")

        self.data = pd.DataFrame(records)
        print("Toplam eğitim satırı:", len(self.data))
        return self.data

    # ------------------- Feature Engineering -------------------
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["driver", "year", "round"]).reset_index(drop=True)

        df["races_so_far"] = df.groupby("driver").cumcount()
        df["prev_team"] = df.groupby("driver")["team"].shift(1)
        df["team_changed"] = (df["team"] != df["prev_team"]).astype(int).fillna(0)
        first_year = df.groupby("driver")["year"].transform("min")
        df["rookie"] = ((df["year"] == TARGET_YEAR) & (first_year == TARGET_YEAR)).astype(int)

        df["quali_vs_grid"] = df["quali_position"] - df["grid_position"]

        df = df.sort_values(["year", "round"])
        df["team_mean_points_so_far"] = (
            df.groupby(["year", "team"])["points_in_race"]
              .transform(lambda s: s.shift(1).expanding().mean())
        ).fillna(0)

        df["finished"] = (df["status"] == "Finished").astype(int)
        df["driver_dnf_rate"] = (
            df.groupby("driver")["finished"].transform(lambda s: 1 - s.shift(1).expanding().mean())
        ).fillna(0)
        df["team_dnf_rate"] = (
            df.groupby("team")["finished"].transform(lambda s: 1 - s.shift(1).expanding().mean())
        ).fillna(0)

        df["DriverSeasonPoints"] = df["driver"].map(self.full_driver_points).fillna(0)
        df["TeamSeasonPoints"] = df["team"].map(self.full_team_points).fillna(0)
        max_d = df["DriverSeasonPoints"].max() or 1
        max_t = df["TeamSeasonPoints"].max() or 1
        df["DriverSeasonPointsNorm"] = df["DriverSeasonPoints"] / max_d
        df["TeamSeasonPointsNorm"] = df["TeamSeasonPoints"] / max_t

        df["delta_to_pole"] = 0.0
        for (y, rnd), grp in df.groupby(["year", "round"]):
            valid = grp["quali_time"]
            pole = valid[valid > 0].min()
            if pd.isna(pole) or pole == 0:
                continue
            df.loc[grp.index, "delta_to_pole"] = grp["quali_time"] - pole

        for col in ["RaceAirTemp","RaceTrackTemp","QualiAirTemp","QualiTrackTemp"]:
            df[col] = df[col].fillna(df[col].median())

        return df

    def prepare_features(self):
        if self.data.empty:
            raise ValueError("Veri yok.")
        df = self._feature_engineering(self.data.copy())

        df["driver_enc"] = self.driver_enc.fit_transform(df["driver"])
        df["team_enc"] = self.team_enc.fit_transform(df["team"])
        df["circuit_enc"] = self.circuit_enc.fit_transform(df["circuit"])

        feature_cols = [
            "driver_enc","team_enc","circuit_enc",
            "grid_position","quali_position","quali_time","delta_to_pole",
            "quali_vs_grid","team_changed","team_mean_points_so_far",
            "rookie","races_so_far",
            "DriverSeasonPoints","TeamSeasonPoints",
            "DriverSeasonPointsNorm","TeamSeasonPointsNorm",
            "driver_dnf_rate","team_dnf_rate",
            "RaceAirTemp","RaceTrackTemp","QualiAirTemp","QualiTrackTemp"
        ]
        return df, feature_cols

    # ------------------- Eğitim -------------------
    def train(self):
        df, feats = self.prepare_features()
        X = df[feats]
        y_win = df["is_winner"]
        finish_pos = df["finish_position"].astype(float).clip(lower=1)
        y_reg = 1.0 / finish_pos

        strength = 1 + 2 * df["DriverSeasonPointsNorm"] + 1 * df["TeamSeasonPointsNorm"]
        sample_weights = np.where(df["year"] == TARGET_YEAR, WEIGHT_FACTOR_2025, 1.0) * strength.values

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        win_acc_scores = []
        for train_idx, test_idx in skf.split(X, y_win):
            base = RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced_subsample"
            )
            base.fit(X.iloc[train_idx], y_win.iloc[train_idx], sample_weight=sample_weights[train_idx])
            preds = base.predict(X.iloc[test_idx])
            win_acc_scores.append(accuracy_score(y_win.iloc[test_idx], preds))
        print(f"Winner CV Accuracy (ortalama): {np.mean(win_acc_scores):.3f}")

        base_full = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced_subsample"
        )
        base_full.fit(X, y_win, sample_weight=sample_weights)
        self.clf = CalibratedClassifierCV(base_full, cv=5, method="sigmoid")
        self.clf.fit(X, y_win, sample_weight=sample_weights)

        Xtr, Xte, ytr, yte, w_tr, w_te = train_test_split(
            X, y_reg, sample_weights, test_size=0.2, random_state=42
        )
        self.reg.fit(Xtr, ytr, sample_weight=w_tr)
        y_hat = self.reg.predict(Xte)
        y_hat_pos = 1.0 / np.maximum(y_hat, 1e-6)
        y_true_pos = 1.0 / np.maximum(yte, 1e-6)
        mae = mean_absolute_error(y_true_pos, y_hat_pos)
        print(f"Position MAE (test set): {mae:.2f}")

        self.is_trained = True

    # ------------------- Tahmin -------------------
    def predict_british_2025(self):
        if not self.is_trained:
            raise ValueError("Önce train() çağır.")
        quali = fastf1.get_session(TARGET_YEAR, TARGET_ROUND, "Q"); quali.load()
        qres = quali.results
        if qres.empty:
            raise ValueError("2025 British GP Quali verisi yok.")

        pole_time = None
        tmp = []
        for _, row in qres.iterrows():
            q_times = [row.get("Q3"), row.get("Q2"), row.get("Q1")]
            for qt in q_times:
                if pd.notna(qt):
                    tmp.append(qt.total_seconds())
                    break
        if tmp:
            pole_time = min(tmp)

        preds = []
        max_d = max(self.full_driver_points.values()) if self.full_driver_points else 1
        max_t = max(self.full_team_points.values()) if self.full_team_points else 1

        for _, row in qres.iterrows():
            if row["FullName"] not in ALLOWED_2025_DRIVERS:
                continue
            try:
                d_enc = self.driver_enc.transform([row["Abbreviation"]])[0]
                t_enc = self.team_enc.transform([row["TeamName"]])[0]
                c_enc = self.circuit_enc.transform([CIRCUIT_NAME])[0]
            except Exception:
                continue

            q_times = [row.get("Q3"), row.get("Q2"), row.get("Q1")]
            best_q = 0.0
            for qt in q_times:
                if pd.notna(qt):
                    best_q = qt.total_seconds() if hasattr(qt,"total_seconds") else 0.0
                    break
            quali_pos = row.get("Position",20); grid_pos = quali_pos
            delta_to_pole = (best_q - pole_time) if (pole_time and pole_time>0 and best_q>0) else 0.0

            driver_code = row["Abbreviation"]; team_name = row["TeamName"]
            d_season = self.full_driver_points.get(driver_code,0.0)
            t_season = self.full_team_points.get(team_name,0.0)
            d_norm = d_season/(max_d or 1); t_norm = t_season/(max_t or 1)

            drv_hist = self.data[(self.data["driver"] == driver_code) & (self.data["year"] == TARGET_YEAR)]
            team_hist = self.data[(self.data["team"] == team_name) & (self.data["year"] == TARGET_YEAR)]

            drv_dnf_rate = 0.0
            if not drv_hist.empty:
                prev = drv_hist.sort_values("round").iloc[:-1]
                if not prev.empty:
                    drv_dnf_rate = 1 - (prev["status"] == "Finished").mean()
            team_dnf_rate = 0.0
            if not team_hist.empty:
                prevt = team_hist.sort_values("round").iloc[:-1]
                if not prevt.empty:
                    team_dnf_rate = 1 - (prevt["status"] == "Finished").mean()

            team_form = team_hist["points_in_race"].mean() if not team_hist.empty else 0.0
            prev_team = drv_hist.sort_values("round")["team"].iloc[-1] if not drv_hist.empty else None
            team_changed = 1 if (prev_team is not None and prev_team != team_name) else 0
            races_so_far = drv_hist.shape[0]
            rookie = 1 if (drv_hist["year"].min()==TARGET_YEAR if not drv_hist.empty else True) else 0

            last_row = drv_hist.sort_values("round").iloc[-1] if not drv_hist.empty else None
            race_air = last_row["RaceAirTemp"] if last_row is not None else 0
            race_track = last_row["RaceTrackTemp"] if last_row is not None else 0
            quali_air = last_row["QualiAirTemp"] if last_row is not None else 0
            quali_track = last_row["QualiTrackTemp"] if last_row is not None else 0

            feats = [
                d_enc,t_enc,c_enc,
                grid_pos,quali_pos,best_q,delta_to_pole,
                (quali_pos-grid_pos),team_changed,team_form,
                rookie,races_so_far,
                d_season,t_season,d_norm,t_norm,
                drv_dnf_rate,team_dnf_rate,
                race_air,race_track,quali_air,quali_track
            ]

            win_prob = self.clf.predict_proba([feats])[0][1]
            reg_pred = self.reg.predict([feats])[0]
            predicted_finish = 1.0 / max(reg_pred, 1e-6)

            preds.append({
                "driver": row["FullName"],
                "team": team_name,
                "predicted_finish": predicted_finish,
                "win_probability": win_prob,
                "driver_points_norm": d_norm
            })

        dfp = pd.DataFrame(preds)
        # Nihai skor: win_prob * (1 + 5*driver_points_norm) / predicted_finish
        dfp["score"] = dfp["win_probability"] * (1 + 5*dfp["driver_points_norm"]) / dfp["predicted_finish"].replace(0, np.nan)
        dfp = dfp.sort_values("score", ascending=False).reset_index(drop=True)
        dfp["predicted_position"] = dfp.index + 1

        if pole_time:
            minutes = int(pole_time // 60)
            seconds = pole_time % 60
            fastest_lap_str = f"{minutes}:{seconds:06.3f}"
        else:
            fastest_lap_str = "Bilinmiyor"

        result = dfp[["predicted_position","driver","team","predicted_finish","win_probability","score"]]
        print("\n=== Tahmini Yarış Sonucu ===")
        print(result.to_string(index=False))
        print(f"\n🌟 Tahmini En Hızlı Tur (pole bazlı): {fastest_lap_str}")

        result.to_csv("british2025_race_prediction_final.csv", index=False)
        print("Kaydedildi: british2025_race_prediction_final.csv")
        return result


def main():
    print(">> SCRIPT BAŞLADI")
    predictor = BritishPredictor()
    predictor.collect_training_data()
    if predictor.data.empty:
        print("Veri yok."); return
    predictor.train()

    print("\n--- FASTF1'DEN HESAPLANAN 2025 SEZON PUANLARI (British öncesi) ---")
    print("Sürücü puanları:", predictor.full_driver_points)
    print("Takım puanları:", predictor.full_team_points)

    predictor.predict_british_2025()


if __name__ == "__main__":
    main()
