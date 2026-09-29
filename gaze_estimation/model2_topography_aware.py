"""
Gaze estimation - Model II: Topography-Aware Model
Fuses a dynamic EEG branch (Conv1D+attention+BiLSTM) with a static
per-trial topography branch; predicts gaze position directly.

Usage:
    python model2_topography_aware.py --data <path>
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import (
    Add,
    Bidirectional,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    LSTM,
    MaxPooling1D,
    MultiHeadAttention,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

SCREEN_WIDTH, SCREEN_HEIGHT = 800.0, 600.0
PIXEL_TO_MM = 0.5

EPOCHS = 120
BATCH_SIZE = 32
PATIENCE = 10
LEARNING_RATE = 5e-4


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


class LabelScalerMinMax:
    def fit(self, y):
        self.min_ = y.min(axis=0)
        self.max_ = y.max(axis=0)
        self.diff_ = self.max_ - self.min_
        self.diff_[self.diff_ == 0] = 1.0
        return self

    def transform(self, y):
        return (y - self.min_) / self.diff_

    def inverse_transform(self, y_norm):
        return y_norm * self.diff_ + self.min_


def load_and_extract_topography(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        X_raw = data["EEG"].astype("float32")
        y_raw = data["labels"].astype("float32")
    if y_raw.shape[1] > 2:
        y_raw = y_raw[:, -2:]

    valid_idx = np.where(
        (y_raw[:, 0] >= 0) & (y_raw[:, 0] <= SCREEN_WIDTH) & (y_raw[:, 1] >= 0) & (y_raw[:, 1] <= SCREEN_HEIGHT)
    )[0]
    X_clean, y_clean = X_raw[valid_idx], y_raw[valid_idx]

    print("Extracting topography features (per-trial mean/std maps)...")
    topo_mean = np.mean(X_clean, axis=1)  # (N, 129)
    topo_std = np.std(X_clean, axis=1)  # (N, 129)
    X_topo = np.concatenate([topo_mean, topo_std], axis=1)  # (N, 258)
    print(f"Topography features shape: {X_topo.shape}")

    return X_clean, X_topo, y_clean


def build_topo_aware_model(eeg_shape, topo_shape):
    input_eeg = Input(shape=eeg_shape, name="eeg_input")
    x = Conv1D(64, kernel_size=15, padding="same", activation="relu")(input_eeg)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, kernel_size=10, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.4)(x)
    dynamic_features = Dense(128, activation="relu")(x)

    input_topo = Input(shape=topo_shape, name="topo_input")
    y = Dense(128, activation="relu")(input_topo)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    y = Dense(128, activation="relu")(y)
    static_features = Dense(64, activation="relu")(y)

    combined = concatenate([dynamic_features, static_features])
    z = Dense(256, activation="relu")(combined)
    z = Dropout(0.4)(z)
    z = Dense(128, activation="relu")(z)
    out = Dense(2, name="gaze_output", activation="sigmoid")(z)

    return Model(inputs=[input_eeg, input_topo], outputs=out)


# ============================================================
# Plots
# ============================================================
def plot_results(history, y_true, y_pred, mae_mm, rmse_mm):
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="Val Loss", ls="--")
    axes[0].set_title("Model Loss (Huber)")
    axes[0].legend()
    axes[1].plot(history.history["mean_absolute_error"], label="Train MAE")
    if "val_mean_absolute_error" in history.history:
        axes[1].plot(history.history["val_mean_absolute_error"], label="Val MAE", ls="--")
    axes[1].set_title("MAE (normalized space)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("model2_learning_curves.png", dpi=200)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    ax1.scatter(y_true[:, 0], y_true[:, 1], c="blue", s=10, alpha=0.4, label="Actual Gaze")
    ax1.set_title("Actual Gaze Locations")
    ax1.set_xlim(-50, SCREEN_WIDTH + 50)
    ax1.set_ylim(-50, SCREEN_HEIGHT + 50)
    ax1.invert_yaxis()
    ax1.set_aspect("equal")
    ax1.legend()
    ax2.scatter(y_pred[:, 0], y_pred[:, 1], c="red", s=10, alpha=0.4, label="Predicted Gaze")
    ax2.set_title("Predicted Gaze Locations")
    ax2.set_aspect("equal")
    ax2.legend()
    fig.suptitle(f"Topography-Aware Model\nMAE: {mae_mm:.2f} mm | RMSE: {rmse_mm:.2f} mm")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("model2_gaze_comparison.png", dpi=200)
    plt.show()

    errors = np.linalg.norm(y_true - y_pred, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(errors, bins=50, kde=True, ax=axes[0])
    axes[0].axvline(np.mean(errors), color="red", ls="--", label=f"Mean: {np.mean(errors):.1f} px")
    axes[0].set_title("Error distribution")
    axes[0].legend()
    sc = axes[1].scatter(y_true[:, 0], y_true[:, 1], c=errors, cmap="jet", s=10, alpha=0.6)
    plt.colorbar(sc, ax=axes[1], label="Error (px)")
    axes[1].invert_yaxis()
    axes[1].set_title("Spatial error heatmap")
    plt.tight_layout()
    plt.savefig("model2_error_analysis.png", dpi=200)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Position_task_with_dots_synchronised_min.npz")
    parser.add_argument("--out", default="topo_aware_gaze_model.keras")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    X_clean, X_topo, y_clean = load_and_extract_topography(args.data)

    N = len(X_clean)
    train_end, val_end = int(0.70 * N), int(0.85 * N)

    X_eeg_train, X_topo_train, y_train = X_clean[:train_end], X_topo[:train_end], y_clean[:train_end]
    X_eeg_val, X_topo_val, y_val = X_clean[train_end:val_end], X_topo[train_end:val_end], y_clean[train_end:val_end]
    X_eeg_test, X_topo_test, y_test = X_clean[val_end:], X_topo[val_end:], y_clean[val_end:]

    eeg_scaler = EEGScaler().fit(X_eeg_train)
    X_eeg_train_n, X_eeg_val_n, X_eeg_test_n = (eeg_scaler.transform(x) for x in (X_eeg_train, X_eeg_val, X_eeg_test))

    label_scaler = LabelScalerMinMax().fit(y_train)
    y_train_n, y_val_n = label_scaler.transform(y_train), label_scaler.transform(y_val)

    topo_mean_val = X_topo_train.mean(axis=0)
    topo_std_val = X_topo_train.std(axis=0)
    topo_std_val[topo_std_val == 0] = 1.0

    def scale_topo(X):
        return (X - topo_mean_val) / topo_std_val

    X_topo_train_n, X_topo_val_n, X_topo_test_n = (scale_topo(x) for x in (X_topo_train, X_topo_val, X_topo_test))

    model = build_topo_aware_model((X_eeg_train_n.shape[1], X_eeg_train_n.shape[2]), (X_topo_train_n.shape[1],))
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=tf.keras.losses.Huber(delta=0.1), metrics=["mean_absolute_error"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1, monitor="val_loss"),
    ]

    history = model.fit(
        x=[X_eeg_train_n, X_topo_train_n],
        y=y_train_n,
        validation_data=([X_eeg_val_n, X_topo_val_n], y_val_n),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    y_pred_n = model.predict([X_eeg_test_n, X_topo_test_n], batch_size=BATCH_SIZE)
    y_pred_px = label_scaler.inverse_transform(y_pred_n)

    mae = mean_absolute_error(y_test, y_pred_px)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_px))
    print(f"\nTest MAE: {mae:.2f} px ({mae*PIXEL_TO_MM:.2f} mm)")
    print(f"Test RMSE: {rmse:.2f} px ({rmse*PIXEL_TO_MM:.2f} mm)")

    model.save(args.out)
    print(f"Model saved to {args.out}")

    if not args.no_plots:
        plot_results(history, y_test, y_pred_px, mae * PIXEL_TO_MM, rmse * PIXEL_TO_MM)


if __name__ == "__main__":
    main()
