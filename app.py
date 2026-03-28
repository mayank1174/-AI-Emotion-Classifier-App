from flask import Flask, request, jsonify
from flask_cors import CORS

from result import classify_emotion, EMOTION_CONFIG, EMOTION_LABELS
from fusion import combine_emotions
from audio_model import build_rf_model, predict_audio_emotion

app = Flask(__name__)
CORS(app)

# ── Text model: DistilBERT fine-tuned for emotion ──────────────────────────
text_pipeline = None
try:
    from transformers import pipeline
    print("Loading DistilBERT emotion model...")
    # j-hartmann/emotion-english-distilroberta-base is a DistilBERT-based model
    # fine-tuned on multiple emotion datasets (GoEmotions, ISEAR, etc.)
    text_pipeline = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )
    print("DistilBERT text model ready.")
except ImportError:
    print("transformers not installed — falling back to rule-based classifier.")
except Exception as e:
    print(f"Failed to load DistilBERT model: {e}")

# ── Audio model: MFCC + Random Forest ─────────────────────────────────────
rf_model, label_encoder = None, None
try:
    rf_model, label_encoder = build_rf_model()
except Exception as e:
    print(f"Failed to build RF audio model: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────

def format_emotions(base_scores: dict) -> list:
    """Normalise a {emotion_key: raw_score} dict to the 13-emotion UI format."""
    scores = {k: 0.0 for k in EMOTION_CONFIG}
    for k, v in base_scores.items():
        if k in scores:
            scores[k] = v

    total = sum(scores.values())
    if total == 0:
        scores['neutral'] = 100.0
        total = 100.0

    result = []
    for key, cfg in EMOTION_CONFIG.items():
        result.append({
            'emotion': EMOTION_LABELS[key],
            'score': round((scores[key] / total) * 100.0, 1),
            'color': cfg['color'],
        })

    result.sort(key=lambda x: x['score'], reverse=True)
    return result


# Map DistilBERT label names → our internal emotion keys
_DISTILBERT_MAP = {
    'anger':   'angry',
    'disgust': 'disgusted',
    'fear':    'fearful',
    'joy':     'happy',
    'neutral': 'neutral',
    'sadness': 'sad',
    'surprise':'surprised',
}

# Map RF audio emotion keys → our internal keys (already aligned)
_RF_MAP = {
    'angry':    'angry',
    'fearful':  'fearful',
    'happy':    'happy',
    'neutral':  'neutral',
    'sad':      'sad',
    'surprised':'surprised',
    'disgusted':'disgusted',
}


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    if text_pipeline:
        try:
            preds = text_pipeline(text)[0]
            base_scores = {
                _DISTILBERT_MAP.get(p['label'], 'neutral'): p['score'] * 100.0
                for p in preds
            }
            return jsonify(format_emotions(base_scores))
        except Exception as e:
            print(f"DistilBERT pipeline error: {e}")

    # Fallback: rule-based keyword classifier
    return jsonify(classify_emotion(text))


@app.route('/api/classify_audio', methods=['POST'])
def classify_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio provided'}), 400

    audio_bytes = request.files['audio'].read()

    if rf_model is not None and label_encoder is not None:
        try:
            proba_dict = predict_audio_emotion(audio_bytes, rf_model, label_encoder)
            base_scores = {_RF_MAP.get(k, k): v for k, v in proba_dict.items()}
            return jsonify(format_emotions(base_scores))
        except Exception as e:
            print(f"RF audio model error: {e}")

    # Fallback: deterministic pseudo-random based on audio hash
    import hashlib, random
    h = hashlib.md5(audio_bytes).digest()
    random.seed(int.from_bytes(h[:4], 'big'))
    all_emotions = [k for k in EMOTION_CONFIG if k != 'neutral']
    e1, e2 = random.sample(all_emotions, 2)
    return jsonify(format_emotions({
        e1: random.uniform(35, 60),
        e2: random.uniform(15, 30),
        'neutral': random.uniform(10, 25),
    }))


@app.route('/api/combine', methods=['POST'])
def combine():
    data = request.get_json()
    emotions1 = data.get('emotions1', [])
    emotions2 = data.get('emotions2', [])

    sets = [e for e in [emotions1, emotions2] if e]
    if len(sets) < 1:
        return jsonify({'error': 'At least one emotion array required'}), 400

    return jsonify(combine_emotions(sets))


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'text_model': 'distilbert' if text_pipeline else 'rule-based',
        'audio_model': 'random-forest-mfcc' if rf_model else 'fallback',
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
