"""
Audio emotion recognition using MFCC features + Random Forest.

At startup, a Random Forest is trained on synthetic MFCC data as a placeholder.
For production accuracy, replace `_build_synthetic_training_data()` with a real
labelled dataset (e.g. RAVDESS: https://zenodo.org/record/1188976).
"""

import io
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# The 7 core emotions the RF maps to (subset that audio typically captures well)
AUDIO_EMOTIONS = ['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'disgusted']

N_MFCC = 40  # number of MFCC coefficients to extract


def extract_mfcc(audio_bytes: bytes) -> np.ndarray:
    """Load audio bytes and return a (N_MFCC,) mean-MFCC feature vector."""
    audio_io = io.BytesIO(audio_bytes)
    try:
        y, sr = librosa.load(audio_io, sr=22050, mono=True, duration=10.0)
    except Exception:
        # If librosa can't decode (e.g. webm), try soundfile as fallback
        import soundfile as sf
        audio_io.seek(0)
        y, sr = sf.read(audio_io, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfccs, axis=1)  # shape: (N_MFCC,)


def _build_synthetic_training_data(n_samples_per_class: int = 120):
    """
    Generate synthetic MFCC-shaped training data with class-specific offsets.
    Replace this with real audio features for production use.
    """
    rng = np.random.RandomState(42)
    X, y = [], []

    # Each emotion gets a slightly different centroid in MFCC space
    offsets = {
        'angry':     rng.randn(N_MFCC) * 3 + np.linspace(-10, 10, N_MFCC),
        'fearful':   rng.randn(N_MFCC) * 2 + np.linspace(-5, 5, N_MFCC),
        'happy':     rng.randn(N_MFCC) * 2 + np.linspace(5, 15, N_MFCC),
        'neutral':   np.zeros(N_MFCC),
        'sad':       rng.randn(N_MFCC) * 2 + np.linspace(-15, -5, N_MFCC),
        'surprised': rng.randn(N_MFCC) * 3 + np.linspace(8, 18, N_MFCC),
        'disgusted': rng.randn(N_MFCC) * 2 + np.linspace(-8, 2, N_MFCC),
    }

    for emotion in AUDIO_EMOTIONS:
        centroid = offsets[emotion]
        samples = centroid + rng.randn(n_samples_per_class, N_MFCC) * 4
        X.append(samples)
        y.extend([emotion] * n_samples_per_class)

    return np.vstack(X), np.array(y)


def build_rf_model() -> tuple[RandomForestClassifier, LabelEncoder]:
    """Train and return a (model, label_encoder) tuple."""
    print("Training MFCC Random Forest audio model...")
    X, y = _build_synthetic_training_data()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y_enc)
    print(f"Audio RF model ready — {len(le.classes_)} emotion classes.")
    return clf, le


def predict_audio_emotion(audio_bytes: bytes, clf: RandomForestClassifier, le: LabelEncoder) -> dict:
    """
    Extract MFCC from audio bytes and return a dict of {emotion_key: probability_pct}.
    """
    features = extract_mfcc(audio_bytes).reshape(1, -1)
    proba = clf.predict_proba(features)[0]  # shape: (n_classes,)

    return {le.classes_[i]: float(proba[i]) * 100.0 for i in range(len(le.classes_))}
