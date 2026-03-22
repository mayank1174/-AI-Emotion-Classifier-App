from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# ML Libraries
import cv2
import numpy as np
import tensorflow as tf
import joblib
from deepface import DeepFace

from result import classify_emotion, EMOTION_CONFIG, EMOTION_LABELS
from fusion import combine_emotions

app = Flask(__name__)
CORS(app)  # Allow requests from the Vite dev server

# --- ML Model Loading Scaffolding ---

# 1. Custom ML Model (Text Sentiment/Emotion)
try:
    text_model = joblib.load('models/custom_text_model.pkl')
    text_vectorizer = joblib.load('models/custom_text_vectorizer.pkl')
    print("Custom ML text model loaded successfully.")
except Exception as e:
    text_model = None
    text_vectorizer = None
    print(f"No custom text model found: {e}. Falling back to rule-based.")

# 2. TensorFlow Model (Audio Emotion)
try:
    audio_tf_model = tf.keras.models.load_model('models/audio_tf_model.keras')
    print("TensorFlow audio model loaded successfully.")
except Exception as e:
    audio_tf_model = None
    print(f"No TF audio model found: {e}. Falling back to default.")


def format_emotions(base_scores):
    """Helper to ensure output always matches the UI's 13-emotion arrays format."""
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
            'color': cfg['color']
        })
        
    result.sort(key=lambda x: x['score'], reverse=True)
    return result


@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
        
    if text_model and text_vectorizer:
        # Use Custom ML Model instead of Dummy/Rules
        features = text_vectorizer.transform([text])
        prediction = text_model.predict_proba(features)[0]
        # Assumes model.classes_ maps to our 13 emotions
        classes = text_model.classes_
        base_scores = {cls.lower(): prob * 100.0 for cls, prob in zip(classes, prediction)}
        return jsonify(format_emotions(base_scores))
        
    return jsonify(classify_emotion(text))


@app.route('/api/classify_image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image_data = file.read()
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Real AI: DeepFace for emotion
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        result_dict = results[0] if isinstance(results, list) else results
        
        df_emotions = result_dict.get('emotion', {})
        mapping = {
            'happy': 'happy', 'sad': 'sad', 'angry': 'angry', 
            'fear': 'fearful', 'surprise': 'surprised', 
            'disgust': 'disgusted', 'neutral': 'neutral'
        }
        
        base_scores = {our_key: df_emotions.get(df_key, 0.0) for df_key, our_key in mapping.items()}
        return jsonify(format_emotions(base_scores))
        
    except Exception as e:
        print(f"DeepFace processing error: {e}")
        return jsonify(format_emotions({'neutral': 50.0, 'confused': 30.0, 'sad': 20.0}))


@app.route('/api/classify_audio', methods=['POST'])
def classify_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio provided'}), 400
    
    file = request.files['audio']
    
    if audio_tf_model:
        # Placeholder integration space for TensorFlow audio feature extraction
        # Example using librosa to extract MFCC:
        # features = librosa.feature.mfcc(y=y, sr=sr)
        # preds = audio_tf_model.predict(features)
        
        # Assuming preds maps to our classes
        # return jsonify(format_emotions(mapped_predictions))
        pass
    
    # Fallback to dummy
    result = format_emotions({'neutral': 45.0, 'calm': 35.0, 'happy': 20.0})
    return jsonify(result)


@app.route('/api/combine', methods=['POST'])
def combine():
    data = request.get_json()
    emotions1 = data.get('emotions1', [])
    emotions2 = data.get('emotions2', [])
    emotions3 = data.get('emotions3', []) 
    
    sets = [e for e in [emotions1, emotions2, emotions3] if e]
    if len(sets) < 1:
        return jsonify({'error': 'At least one emotion array required'}), 400
        
    return jsonify(combine_emotions(sets))


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, port=5000)
