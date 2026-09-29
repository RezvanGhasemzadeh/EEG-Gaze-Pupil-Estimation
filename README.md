# EEG-Based Eye Tracking: Reproducibility Code

Code accompanying *"EEG-Based Eye Tracking: A Deep Learning Framework for
Gaze Position and Pupil Size Estimation"* (Heliyon, HELIYON-D-26-07304).

## Data

The raw EEGEyeNet recordings are not redistributed here - see the
manuscript's Data and Code Availability statement to obtain them. Scripts
take a local data path via `--data`/`--data-folder`.

## Installation

```bash
pip install -r requirements.txt
```

## Repository structure

```
gaze_estimation/
    model1_kinematic_hybrid.py     Model I - Kinematic Hybrid
    model2_topography_aware.py     Model II - Topography-Aware
    knn_spatial_calibration.py     k-NN spatial residual calibration

pupil_estimation/
    train_pupil_model.py           Pupil-size model
```

Each script is self-contained (no shared modules); gaze- and
pupil-estimation code are kept separate.

## Usage

Gaze estimation:

```bash
python gaze_estimation/model1_kinematic_hybrid.py --data <path>
python gaze_estimation/model2_topography_aware.py --data <path>
python gaze_estimation/knn_spatial_calibration.py --variant kinematic --data <path> --model kinematic_gaze_model.keras
```

Pupil-size estimation (`--data-folder` points at a local directory of
`.mat` files):

```bash
python pupil_estimation/train_pupil_model.py --dataset vss --data-folder <path>
```
