"""
k-NN spatial residual calibration for the gaze-estimation models
Fits a distance-weighted k-NN (k=100) mapping prediction -> residual,
applied one-step and autoregressively.

Usage:
    python knn_spatial_calibration.py --variant kinematic --data <path> --model kinematic_gaze_model.keras
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import load_model

SCREEN_WIDTH, SCREEN_HEIGHT = 800.0, 600.0
PIXEL_TO_MM = 0.5
K_NEIGHBORS = 100


# ============================================================
# Scalers (fit on train only)
# ============================================================
class EEGScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=(0, 1))
        self.std_ = X.std(axis=(0, 1))
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LabelScalerMinMax:
    def fit(self, y):
        self.min_ = y.min(axis=0)
        self.max_ = y.max(axis=0)
        diff = self.max_ - self.min_
        diff[diff == 0] = 1.0
        self.scale_ = diff
        return self

    def transform(self, y):
        return (y - self.min_) / self.scale_

    def inverse_transform(self, y_norm):
        return y_norm * self.scale_ + self.min_

    def fit_transform(self, y):
        return self.fit(y).transform(y)


def load_kinematic_split(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        X_raw, y_raw = data["EEG"], data["labels"]
    if y_raw.shape[1] > 2:
        y_raw = y_raw[:, -2:]
    X_raw, y_raw = X_raw.astype("float32"), y_raw.astype("float32")

    valid_idx = np.where(
        (y_raw[:, 0] >= 0) & (y_raw[:, 0] <= SCREEN_WIDTH) & (y_raw[:, 1] >= 0) & (y_raw[:, 1] <= SCREEN_HEIGHT)
    )[0]
    X_clean, y_clean = X_raw[valid_idx], y_raw[valid_idx]

    X_eeg_input = X_clean[1:]
    X_prev_gaze_input = y_clean[:-1]
    y_target_output = y_clean[1:]
    N = len(X_eeg_input)
    train_size, val_size = int(N * 0.7), int(N * 0.15)

    return {
        "train": (X_eeg_input[:train_size], X_prev_gaze_input[:train_size], y_target_output[:train_size]),
        "test": (
            X_eeg_input[train_size + val_size :],
            X_prev_gaze_input[train_size + val_size :],
            y_target_output[train_size + val_size :],
        ),
    }


def load_topo_split(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        X_raw = data["EEG"].astype("float32")
        y_raw = data["labels"].astype("float32")
    if y_raw.shape[1] > 2:
        y_raw = y_raw[:, -2:]
    valid_idx = np.where(
        (y_raw[:, 0] >= 0) & (y_raw[:, 0] <= SCREEN_WIDTH) & (y_raw[:, 1] >= 0) & (y_raw[:, 1] <= SCREEN_HEIGHT)
    )[0]
    X_clean, y_clean = X_raw[valid_idx], y_raw[valid_idx]

    topo_mean = np.mean(X_clean, axis=1)
    topo_std = np.std(X_clean, axis=1)
    X_topo = np.concatenate([topo_mean, topo_std], axis=1)

    N = len(X_clean)
    train_end, val_end = int(0.70 * N), int(0.85 * N)
    return {
        "train": (X_clean[:train_end], X_topo[:train_end], y_clean[:train_end]),
        "test": (X_clean[val_end:], X_topo[val_end:], y_clean[val_end:]),
    }


def print_metrics(y_true, y_pred, title):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[{title}] MAE: {mae*PIXEL_TO_MM:.2f} mm | RMSE: {rmse*PIXEL_TO_MM:.2f} mm")
    return mae, rmse


def run_kinematic(args):
    split = load_kinematic_split(args.data)
    X_eeg_train, X_prev_train, y_train = split["train"]
    X_eeg_test, X_prev_test, y_test = split["test"]

    eeg_scaler = EEGScaler().fit(X_eeg_train)
    X_eeg_train_n, X_eeg_test_n = eeg_scaler.transform(X_eeg_train), eeg_scaler.transform(X_eeg_test)

    label_scaler = LabelScalerMinMax().fit(y_train)
    y_train_n = label_scaler.transform(y_train)
    X_prev_train_n = label_scaler.transform(X_prev_train)
    X_prev_test_n = label_scaler.transform(X_prev_test)

    print(f"Loading Kinematic model from {args.model}...")
    model = load_model(args.model)

    print("Fitting k-NN spatial corrector on training residuals...")
    y_pred_train_n = model.predict([X_eeg_train_n, X_prev_train_n], verbose=1)
    residuals_train = y_train_n - y_pred_train_n
    corrector = KNeighborsRegressor(n_neighbors=K_NEIGHBORS, weights="distance", n_jobs=-1)
    corrector.fit(y_pred_train_n, residuals_train)

    # One-step (teacher-forced).
    y_pred_1s_n = model.predict([X_eeg_test_n, X_prev_test_n], verbose=1)
    y_pred_1s_calib_n = y_pred_1s_n + corrector.predict(y_pred_1s_n)
    y_pred_1s_calib_px = label_scaler.inverse_transform(y_pred_1s_calib_n)
    print_metrics(y_test, y_pred_1s_calib_px, "One-step (corrected)")

    # Autoregressive, with the corrected estimate fed forward as prev_gaze.
    print("Running autoregressive evaluation with correction feedback...")
    preds = []
    prev = X_prev_test_n[0].reshape(1, -1)
    for t in range(len(X_eeg_test_n)):
        eeg_t = X_eeg_test_n[t : t + 1]
        raw_n = model.predict([eeg_t, prev], verbose=0)
        corrected_n = raw_n + corrector.predict(raw_n)
        preds.append(corrected_n[0])
        prev = corrected_n
    y_pred_ar_px = label_scaler.inverse_transform(np.array(preds))
    mae_cl, rmse_cl = print_metrics(y_test, y_pred_ar_px, "Autoregressive (corrected)")

    return y_test, y_pred_ar_px, mae_cl * PIXEL_TO_MM, rmse_cl * PIXEL_TO_MM, "Kinematic Hybrid (k-NN corrected)"


def run_topo_aware(args):
    split = load_topo_split(args.data)
    X_eeg_train, X_topo_train, y_train = split["train"]
    X_eeg_test, X_topo_test, y_test = split["test"]

    eeg_scaler = EEGScaler().fit(X_eeg_train)
    X_eeg_train_n, X_eeg_test_n = eeg_scaler.transform(X_eeg_train), eeg_scaler.transform(X_eeg_test)

    label_scaler = LabelScalerMinMax().fit(y_train)
    y_train_n = label_scaler.transform(y_train)

    topo_mean_val, topo_std_val = X_topo_train.mean(axis=0), X_topo_train.std(axis=0)
    topo_std_val[topo_std_val == 0] = 1.0

    def scale_topo(X):
        return (X - topo_mean_val) / topo_std_val

    X_topo_train_n, X_topo_test_n = scale_topo(X_topo_train), scale_topo(X_topo_test)

    print(f"Loading Topography-Aware model from {args.model}...")
    model = load_model(args.model)

    print("Fitting k-NN spatial corrector on training residuals...")
    y_pred_train_n = model.predict([X_eeg_train_n, X_topo_train_n], verbose=1)
    residuals_train = y_train_n - y_pred_train_n
    corrector = KNeighborsRegressor(n_neighbors=K_NEIGHBORS, weights="distance", n_jobs=-1)
    corrector.fit(y_pred_train_n, residuals_train)

    y_pred_test_n = model.predict([X_eeg_test_n, X_topo_test_n], verbose=1)
    y_pred_calib_n = y_pred_test_n + corrector.predict(y_pred_test_n)
    y_pred_calib_px = label_scaler.inverse_transform(y_pred_calib_n)
    mae, rmse = print_metrics(y_test, y_pred_calib_px, "Corrected")

    return y_test, y_pred_calib_px, mae * PIXEL_TO_MM, rmse * PIXEL_TO_MM, "Topography-Aware (k-NN corrected)"


def plot_results(y_true, y_pred, mae_mm, rmse_mm, title):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(y_true[:, 0], y_true[:, 1], c="blue", s=15, alpha=0.4, label="Actual Gaze")
    ax.scatter(y_pred[:, 0], y_pred[:, 1], c="red", s=15, marker="x", alpha=0.5, label="Predicted Gaze")
    ax.set_xlim(0, SCREEN_WIDTH)
    ax.set_ylim(SCREEN_HEIGHT, 0)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"{title}\nMAE: {mae_mm:.2f} mm | RMSE: {rmse_mm:.2f} mm")
    plt.tight_layout()
    plt.savefig("knn_calibration_result.png", dpi=200)
    plt.show()

    errors = np.linalg.norm(y_true - y_pred, axis=1)
    plt.figure(figsize=(7, 6))
    sns.histplot(errors, bins=50, kde=True)
    plt.axvline(np.mean(errors), color="red", ls="--", label=f"Mean: {np.mean(errors):.1f} px")
    plt.title("Corrected error distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("knn_calibration_error_distribution.png", dpi=200)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["kinematic", "topo_aware"], required=True)
    parser.add_argument("--data", required=True, help="Path to Position_task_with_dots_synchronised_min.npz")
    parser.add_argument("--model", required=True, help="Path to the trained .keras model")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.variant == "kinematic":
        y_true, y_pred, mae_mm, rmse_mm, title = run_kinematic(args)
    else:
        y_true, y_pred, mae_mm, rmse_mm, title = run_topo_aware(args)

    if not args.no_plots:
        plot_results(y_true, y_pred, mae_mm, rmse_mm, title)


if __name__ == "__main__":
    main()
