from flask import Flask, request, jsonify
from flask_cors import CORS

from result import classify_emotion
from fusion import combine_emotions

app = Flask(__name__)
CORS(app)  # Allow requests from the Vite dev server

@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    return jsonify(classify_emotion(text))


@app.route('/api/classify_image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    result = [
        {'emotion': 'Happy', 'score': 65.0, 'color': '#22c55e'},
        {'emotion': 'Neutral', 'score': 20.0, 'color': '#6b7280'},
        {'emotion': 'Surprised', 'score': 15.0, 'color': '#a855f7'},
    ]
    return jsonify(result)


@app.route('/api/classify_audio', methods=['POST'])
def classify_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio provided'}), 400
    
    file = request.files['audio']
    
    result = [
        {'emotion': 'Neutral', 'score': 45.0, 'color': '#6b7280'},
        {'emotion': 'Calm', 'score': 35.0, 'color': '#06b6d4'},
        {'emotion': 'Happy', 'score': 20.0, 'color': '#22c55e'},
    ]
    return jsonify(result)


@app.route('/api/combine', methods=['POST'])
def combine():
    data = request.get_json()
    emotions1 = data.get('emotions1', [])
    emotions2 = data.get('emotions2', [])
    emotions3 = data.get('emotions3', []) # Added support for 3rd modality
    
    sets = [e for e in [emotions1, emotions2, emotions3] if e]
    if len(sets) < 1:
        return jsonify({'error': 'At least one emotion array required'}), 400
        
    return jsonify(combine_emotions(sets))


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
