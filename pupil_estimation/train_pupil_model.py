"""
Pupil-size estimation - main model
Residual Conv1D + GlobalAveragePooling1D + LSTM + Dense on 129-channel
EEG windows -> mean pupil size. Same architecture and preprocessing for
both datasets. Normalization for the subject-dependent split is fit on
each subject's training portion only (no val/test leakage into scaling).

Usage:
    python train_pupil_model.py --dataset vss --data-folder <path>
"""

import argparse
import glob
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Add,
    AveragePooling1D,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    ReLU,
    Reshape,
)
from tensorflow.keras.models import Model

WINDOW_SIZE = 500
OVERLAP_STEP = 250  # training windows (50% overlap)
NORMAL_STEP = 500  # val/test windows (no overlap)

PUPIL_BAND = (0.1, 10.0)  # pupil trace band-pass, identical for both datasets

TRAIN_RATIO = 0.80  # 80% of each subject -> train+val, 20% -> test
VAL_RATIO_WITHIN_TRAIN = 0.10  # validation carved out of the 80% train portion only


# ============================================================
# Signal processing
# ============================================================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, data)


def clean_and_interpolate(data, method="cubic"):
    data = data.copy()
    data[data == 0] = np.nan
    valid = ~np.isnan(data)
    if np.sum(valid) > 1:
        interp_func = interp1d(np.where(valid)[0], data[valid], kind=method, fill_value="extrapolate")
        data = interp_func(np.arange(len(data)))
    if np.any(np.isnan(data)):
        data[np.isnan(data)] = np.nanmean(data)
    return data


def normalize_pupil(pupil, fit_slice=None):
    """Min-max normalize using stats fit on fit_slice (default: whole array)."""
    ref = pupil if fit_slice is None else pupil[fit_slice]
    p_min, p_max = np.min(ref), np.max(ref)
    denom = p_max - p_min
    if denom == 0:
        return np.zeros_like(pupil)
    return (pupil - p_min) / denom


def normalize_eeg(eeg, fit_slice=None):
    """Per-channel min-max normalize; eeg shape (samples, channels).
    Stats are fit on fit_slice rows (default: the whole array)."""
    ref = eeg if fit_slice is None else eeg[fit_slice]
    ch_min = np.min(ref, axis=0, keepdims=True)
    ch_max = np.max(ref, axis=0, keepdims=True)
    denom = ch_max - ch_min
    denom[denom == 0] = 1.0
    return (eeg - ch_min) / denom


# ============================================================
# Local .mat loading
# ============================================================
def list_mat_files(data_folder):
    return sorted(glob.glob(os.path.join(data_folder, "*.mat")))


def load_mat_file(file_path):
    data = loadmat(file_path)
    sEEG = data["sEEG"]
    return sEEG["data"][0, 0], sEEG["srate"][0, 0][0, 0], sEEG, sEEG["event"][0, 0]


def load_all_subjects(data_folder, pupil_band):
    """Load + interpolate + band-pass filter every subject. Normalization
    is NOT applied here - it is fit per split (train-only for
    subject_dependent) inside split_subject_dependent/independent."""
    mat_files = list_mat_files(data_folder)
    print(f"{len(mat_files)} files")

    subjects_data = []
    for subj_id, file_path in enumerate(mat_files):
        print(f"Processing subject {subj_id}: {file_path}")
        eeg_data, srate, sEEG, events = load_mat_file(file_path)

        latencies = np.array([int(ev["latency"][0][0]) for ev in events[0]])
        first_latency, last_latency = int(latencies[0]), int(latencies[-1])

        pupil = eeg_data[132, first_latency - 1 : last_latency]
        pupil = clean_and_interpolate(pupil, method="cubic")
        pupil = bandpass_filter(pupil, *pupil_band, srate)

        eeg = eeg_data[0:129, first_latency - 1 : last_latency].T  # (samples, channels), raw

        subjects_data.append({"subject": subj_id, "eeg_raw": eeg, "pupil_raw": pupil, "mat_path": file_path})

    print("Preprocessing completed.")
    return subjects_data


# ============================================================
# Subject-dependent split (80/20 per subject, val carved from the 80%)
# Normalization is fit on each subject's TRAIN portion only, to avoid
# val/test statistics leaking into the scale applied to training data.
# ============================================================
def split_subject_dependent(subjects_data):
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

    for subj in subjects_data:
        eeg_raw, pupil_raw = subj["eeg_raw"], subj["pupil_raw"]
        n_samples = len(pupil_raw)

        split_idx = int(TRAIN_RATIO * n_samples)  # 80% / 20% boundary
        val_size = int(VAL_RATIO_WITHIN_TRAIN * split_idx)  # val = 10% of the 80% train+val region
        train_end = split_idx - val_size  # e.g. 72% train / 8% val / 20% test

        # Fit normalization on the training portion only, then apply to
        # the whole subject (train, val and test all get the same scale).
        fit_slice = slice(0, train_end)
        eeg = normalize_eeg(eeg_raw, fit_slice=fit_slice)
        pupil = normalize_pupil(pupil_raw, fit_slice=fit_slice)

        # TRAIN: overlapping windows, up to train_end
        for i in range(0, train_end - WINDOW_SIZE + 1, OVERLAP_STEP):
            X_train.append(eeg[i : i + WINDOW_SIZE, :])
            y_train.append(np.mean(pupil[i : i + WINDOW_SIZE]))

        # VAL: non-overlapping windows, train_end -> split_idx (never touches test)
        for i in range(train_end, split_idx - WINDOW_SIZE + 1, NORMAL_STEP):
            X_val.append(eeg[i : i + WINDOW_SIZE, :])
            y_val.append(np.mean(pupil[i : i + WINDOW_SIZE]))

        # TEST: non-overlapping windows, final 20% of the subject
        for i in range(split_idx, n_samples - WINDOW_SIZE + 1, NORMAL_STEP):
            X_test.append(eeg[i : i + WINDOW_SIZE, :])
            y_test.append(np.mean(pupil[i : i + WINDOW_SIZE]))

    X_train, y_train = shuffle(np.array(X_train), np.array(y_train), random_state=42)
    return (X_train, y_train), (np.array(X_val), np.array(y_val)), (np.array(X_test), np.array(y_test))


# ============================================================
# Subject-independent split (val subjects carved from the train subjects)
# Each subject belongs entirely to one split, so normalizing a subject on
# its own full signal never leaks across subjects.
# ============================================================
def split_subject_independent(subjects_data):
    np.random.seed(42)
    subject_ids = np.arange(len(subjects_data))
    np.random.shuffle(subject_ids)

    n_train = int(0.80 * len(subject_ids))
    train_ids, test_ids = subject_ids[:n_train], subject_ids[n_train:]

    if len(train_ids) < 2:
        raise ValueError("Not enough training subjects to create separate training and validation subjects.")

    n_val_subjects = max(1, int(VAL_RATIO_WITHIN_TRAIN * len(train_ids)))
    val_ids = train_ids[-n_val_subjects:]
    actual_train_ids = train_ids[:-n_val_subjects]

    def collect(ids, step):
        X, y = [], []
        for idx in ids:
            eeg = normalize_eeg(subjects_data[idx]["eeg_raw"])
            pupil = normalize_pupil(subjects_data[idx]["pupil_raw"])
            for i in range(0, len(pupil) - WINDOW_SIZE + 1, step):
                X.append(eeg[i : i + WINDOW_SIZE, :])
                y.append(np.mean(pupil[i : i + WINDOW_SIZE]))
        return np.array(X), np.array(y)

    X_train, y_train = collect(actual_train_ids, OVERLAP_STEP)
    X_val, y_val = collect(val_ids, NORMAL_STEP)
    X_test, y_test = collect(test_ids, NORMAL_STEP)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================================
# Model
# ============================================================
def build_pupil_model():
    input_layer = Input(shape=(WINDOW_SIZE, 129))
    x = Conv1D(16, 1, padding="same", activation="relu")(input_layer)
    x = BatchNormalization()(x)

    res = Conv1D(32, 1, padding="same")(x)
    x = Conv1D(32, 9, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, 1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, res])
    x = ReLU()(x)
    x = AveragePooling1D(2)(x)

    res = Conv1D(64, 1, padding="same")(x)
    x = Conv1D(64, 9, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, 1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, res])
    x = ReLU()(x)
    x = AveragePooling1D(2)(x)

    x = GlobalAveragePooling1D()(x)
    x = Reshape((1, x.shape[-1]))(x)
    x = LSTM(64)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.1)(x)
    output = Dense(1)(x)

    model = Model(input_layer, output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vss", "dot"], required=True)
    parser.add_argument("--data-folder", required=True, help="Local folder containing the .mat recordings")
    parser.add_argument("--split", choices=["subject_dependent", "subject_independent"], default="subject_dependent")
    parser.add_argument("--out", default=None, help="Path to save the trained model (default: pupil_{dataset}_model.keras)")
    args = parser.parse_args()

    data_folder = args.data_folder
    out_path = args.out or f"pupil_{args.dataset}_model.keras"

    subjects_data = load_all_subjects(data_folder, PUPIL_BAND)

    if args.split == "subject_dependent":
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_subject_dependent(subjects_data)
    else:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_subject_independent(subjects_data)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = build_pupil_model()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50, batch_size=64,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    )

    test_loss, test_mae = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test).squeeze()
    test_r2 = r2_score(y_test, y_pred)
    print(f"TEST MAE: {test_mae:.4f} | TEST R2: {test_r2:.4f}")

    model.save(out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
