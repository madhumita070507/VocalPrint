# VocalPrint 🎤
### A Speech Motor Dynamics Index as a Non-Invasive Biomarker for Neurological Speech Disorders

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![librosa](https://img.shields.io/badge/librosa-0.11.0-green)

## Overview
VocalPrint extracts 14 frame-level acoustic features from raw speech audio
and produces a Neural Motor Index (NMI) score (0–100) indicating speech
motor health. Validated across 5 datasets and 3 neurological disorder
categories using rigorous speaker-level Leave-One-Out Cross-Validation.

## Results
| Dataset | Speakers | Best Model | AUC | Accuracy |
|---------|----------|-----------|-----|----------|
| TORGO | 55 | Logistic Regression | 0.865 | 78.18% |
| UASpeech | 28 | XGBoost | 0.933 | 96.43% |
| Parkinson's | 32 | Logistic Regression | 0.854 | 81.25% |
| Stuttering | — | — | 0.571 | — |
| Vocal Pathology | — | — | 0.670 | — |

> UASpeech result (96.43%) outperforms state-of-the-art SALR
> (wav2vec 2.0 + contrastive learning) at 70.48% under equivalent
> speaker-level evaluation.

## Features — SSI 3.0 Pipeline

### Core Features (compute_motor_features)
- Pitch Instability: σ(ΔF0) / μ(F0)
- Energy Instability: σ(ΔRMS) / μ(RMS)
- Pitch Range: max(F0) - min(F0)
- Energy Range: max(RMS) - min(RMS)
- Pitch Variability: σ(F0)
- MFCC Variability: mean(σ(MFCC_k))

### Extra Features (extract_extra_features)
- Frame-level Jitter (mean + std)
- Frame-level Shimmer (mean + std)
- Zero Crossing Rate (mean + std)
- Spectral Centroid variability
- Voiced Ratio

### Aggregation
Each speaker: mean + std across all utterances → 28-dimensional fingerprint

## Project Structure
## Installation
```bash
pip install librosa scikit-learn xgboost numpy tqdm pandas
```

## Usage

### Feature Extraction
```python
import librosa
import numpy as np

# Core features
def compute_motor_features(wav_path, sr=16000):
    y, sr = librosa.load(wav_path, sr=sr, duration=4.0)
    f0 = librosa.yin(y, fmin=80, fmax=300, sr=sr)
    f0 = f0[~(f0 < 1)]
    pitch_instability = np.std(np.diff(f0)) / (np.mean(f0) + 1e-6)
    # ... returns 6-dim vector

# Extra features  
def extract_extra_features(wav_path, sr=16000):
    y, sr = librosa.load(wav_path, sr=sr, duration=4.0)
    # jitter, shimmer, ZCR, centroid, voiced ratio
    # ... returns 8-dim vector
```

### Speaker-Level Prediction
```python
# Aggregate across multiple files from same speaker
all_features = []
for wav_path in speaker_files:
    f1 = compute_motor_features(wav_path)
    f2 = extract_extra_features(wav_path)
    if f1 and f2:
        all_features.append(f1 + f2)

arr = np.array(all_features)
# 28-dim fingerprint
speaker_vector = np.concatenate([arr.mean(axis=0), arr.std(axis=0)])
```

## Evaluation Methodology
- **TORGO, UASpeech, Parkinson's:** Speaker-level Leave-One-Out CV
- **Stuttering, Vocal Pathology:** 5-Fold Stratified CV
- **Primary metric:** ROC-AUC
- **No speaker leakage:** Test speaker never appears in training fold

## Datasets
| Dataset | Source | Disorder |
|---------|--------|---------|
| TORGO | University of Toronto | Dysarthria |
| UASpeech | University of Illinois | Dysarthria |
| Parkinson's | UCI ML Repository | Parkinson's Disease |
| SEP-28K | ICASSP 2021 | Stuttering |
| Vocal Pathology | Kaggle | Structural pathology |

## Tech Stack
- **Audio:** librosa 0.11.0, YIN pitch extraction
- **ML:** scikit-learn 1.6.1, XGBoost
- **Compute:** Google Colab, Google Drive

## Key Finding
Papers reporting 97–99% accuracy on TORGO use file-level splits allowing
speaker leakage. Under rigorous speaker-level LOO-CV, VocalPrint achieves
competitive results with full interpretability and cross-disorder
specificity that no existing paper demonstrates.

## License
Academic use only.
