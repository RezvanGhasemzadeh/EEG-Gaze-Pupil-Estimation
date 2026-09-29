"""
Gaze estimation - Model I: Kinematic Hybrid Model
Conv1D+attention+BiLSTM branch fused with the previous gaze position;
predicts a delta added kinematically to it.

Usage:
    python model1_kinematic_hybrid.py --data <path>
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

EPOCHS = 100
BATCH_BASE = 32
PATIENCE = 10
LEARNING_RATE = 1e-3


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


def get_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        return tf.distribute.TPUStrategy(tpu)
    except Exception as e:  # noqa: BLE001
        print(f"No TPU or couldn't initialize TPU: {e}")
        return tf.distribute.get_strategy()


def load_and_split(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        X_raw = data["EEG"]
        y_raw = data["labels"]
    if y_raw.shape[1] > 2:
        y_raw = y_raw[:, -2:]
    X_raw, y_raw = X_raw.astype("float32"), y_raw.astype("float32")

    valid_idx = np.where(
        (y_raw[:, 0] >= 0) & (y_raw[:, 0] <= SCREEN_WIDTH) & (y_raw[:, 1] >= 0) & (y_raw[:, 1] <= SCREEN_HEIGHT)
    )[0]
    X_clean, y_clean = X_raw[valid_idx], y_raw[valid_idx]

    # Supervised pairs: predict gaze at t from EEG at t and gaze at t-1.
    X_eeg_input = X_clean[1:]
    X_prev_gaze_input = y_clean[:-1]
    y_target_output = y_clean[1:]
    N = len(X_eeg_input)

    train_end, val_end = int(0.70 * N), int(0.85 * N)
    splits = {
        "train": (X_eeg_input[:train_end], X_prev_gaze_input[:train_end], y_target_output[:train_end]),
        "val": (X_eeg_input[train_end:val_end], X_prev_gaze_input[train_end:val_end], y_target_output[train_end:val_end]),
        "test": (X_eeg_input[val_end:], X_prev_gaze_input[val_end:], y_target_output[val_end:]),
    }
    print("Split sizes:", {k: len(v[0]) for k, v in splits.items()})
    return splits


def build_kinematic_model(eeg_shape, gaze_shape):
    input_eeg = Input(shape=eeg_shape, name="eeg_input")
    input_prev_gaze = Input(shape=gaze_shape, name="prev_gaze_input")

    x = Conv1D(64, kernel_size=10, padding="same", activation="relu")(input_eeg)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, kernel_size=10, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    attn = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)

    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    eeg_features = x

    gaze_features = Dense(32, activation="relu")(input_prev_gaze)
    combined = concatenate([eeg_features, gaze_features])

    h = Dense(64, activation="relu")(combined)
    delta = Dense(2, name="delta_output")(h)
    out = Add(name="kinematic_sum")([input_prev_gaze, delta])

    return Model(inputs=[input_eeg, input_prev_gaze], outputs=out)


def autoregressive_evaluate(model, X_eeg_norm_seq, first_prev_gaze_norm, y_true_pixels, label_scaler):
    preds_norm = []
    prev_gaze = first_prev_gaze_norm.reshape(1, -1).astype("float32")
    for t in range(len(X_eeg_norm_seq)):
        eeg_t = X_eeg_norm_seq[t : t + 1].astype("float32")
        pred_t = model.predict([eeg_t, prev_gaze], verbose=0)
        preds_norm.append(pred_t.reshape(-1))
        prev_gaze = pred_t
    preds_pixels = label_scaler.inverse_transform(np.array(preds_norm))
    mae = mean_absolute_error(y_true_pixels, preds_pixels)
    rmse = np.sqrt(mean_squared_error(y_true_pixels, preds_pixels))
    return preds_pixels, mae, rmse


# ============================================================
# Plots
# ============================================================
def plot_results(history, y_true, y_pred, mae_mm, rmse_mm):
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss", ls="--")
    axes[0].set_title("Model Loss")
    axes[0].legend()
    axes[1].plot(history.history["mean_absolute_error"], label="Train MAE")
    axes[1].plot(history.history["val_mean_absolute_error"], label="Val MAE", ls="--")
    axes[1].set_title("MAE (normalized space)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("model1_learning_curves.png", dpi=200)
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
    fig.suptitle(f"Kinematic Hybrid Model (Autoregressive)\nMAE: {mae_mm:.2f} mm | RMSE: {rmse_mm:.2f} mm")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("model1_gaze_comparison.png", dpi=200)
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
    plt.savefig("model1_error_analysis.png", dpi=200)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Position_task_with_dots_synchronised_min.npz")
    parser.add_argument("--out", default="kinematic_gaze_model.keras")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    strategy = get_strategy()
    print("Replicas:", strategy.num_replicas_in_sync)

    splits = load_and_split(args.data)
    X_eeg_train, X_prev_gaze_train, y_train = splits["train"]
    X_eeg_val, X_prev_gaze_val, y_val = splits["val"]
    X_eeg_test, X_prev_gaze_test, y_test = splits["test"]

    eeg_scaler = EEGScaler().fit(X_eeg_train)
    label_scaler = LabelScalerMinMax().fit(y_train)

    X_eeg_train_n, X_eeg_val_n, X_eeg_test_n = (eeg_scaler.transform(x) for x in (X_eeg_train, X_eeg_val, X_eeg_test))
    y_train_n, y_val_n = label_scaler.transform(y_train), label_scaler.transform(y_val)
    X_prev_train_n, X_prev_val_n, X_prev_test_n = (
        label_scaler.transform(x) for x in (X_prev_gaze_train, X_prev_gaze_val, X_prev_gaze_test)
    )

    with strategy.scope():
        model = build_kinematic_model((X_eeg_train_n.shape[1], X_eeg_train_n.shape[2]), (X_prev_train_n.shape[1],))
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mean_squared_error", metrics=["mean_absolute_error"])
    model.summary()

    batch_size = BATCH_BASE * strategy.num_replicas_in_sync
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)]

    history = model.fit(
        x=[X_eeg_train_n, X_prev_train_n],
        y=y_train_n,
        validation_data=([X_eeg_val_n, X_prev_val_n], y_val_n),
        epochs=EPOCHS,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # One-step (teacher-forced) evaluation.
    y_pred_test_n = model.predict([X_eeg_test_n, X_prev_test_n], batch_size=batch_size)
    y_pred_test_px = label_scaler.inverse_transform(y_pred_test_n)
    mae_1s = mean_absolute_error(y_test, y_pred_test_px)
    rmse_1s = np.sqrt(mean_squared_error(y_test, y_pred_test_px))
    print(f"\nOne-step: MAE {mae_1s*PIXEL_TO_MM:.2f} mm, RMSE {rmse_1s*PIXEL_TO_MM:.2f} mm")

    # Autoregressive (closed-loop) evaluation.
    y_pred_cl_px, mae_cl, rmse_cl = autoregressive_evaluate(model, X_eeg_test_n, X_prev_test_n[0], y_test, label_scaler)
    print(f"Autoregressive: MAE {mae_cl*PIXEL_TO_MM:.2f} mm, RMSE {rmse_cl*PIXEL_TO_MM:.2f} mm")

    model.save(args.out)
    print(f"Model saved to {args.out}")

    if not args.no_plots:
        plot_results(history, y_test, y_pred_cl_px, mae_cl * PIXEL_TO_MM, rmse_cl * PIXEL_TO_MM)


if __name__ == "__main__":
    main()
