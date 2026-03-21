import re

EMOTION_CONFIG = {
    'happy':    { 'keywords': ['happy','joy','excited','love','great','wonderful','amazing','excellent','fantastic','awesome','good','glad','delighted','pleased','cheerful','thrilled','ecstatic'], 'color': '#22c55e' },
    'sad':      { 'keywords': ['sad','unhappy','depressed','miserable','disappointed','sorry','unfortunate','terrible','awful','bad','hurt','crying','tears','heartbroken','gloomy','melancholy'], 'color': '#3b82f6' },
    'angry':    { 'keywords': ['angry','mad','furious','hate','annoyed','irritated','frustrated','outraged','enraged','disgusted','stupid','ridiculous','infuriated','rage'], 'color': '#ef4444' },
    'fearful':  { 'keywords': ['scared','afraid','fear','terrified','frightened','panic','horror','dread','petrified','alarmed'], 'color': '#f97316' },
    'anxious':  { 'keywords': ['anxious','anxiety','nervous','uneasy','restless','tense','apprehensive','jittery','on edge','uncomfortable'], 'color': '#f59e0b' },
    'stressed': { 'keywords': ['stressed','stress','overwhelmed','pressure','burden','strained','exhausted','tired','burnt out','overworked','tension'], 'color': '#dc2626' },
    'worried':  { 'keywords': ['worried','worry','concern','concerned','troubled','bothered','distressed','uncertain','doubtful','hesitant'], 'color': '#fb923c' },
    'confused': { 'keywords': ['confused','bewildered','puzzled','perplexed','lost','unclear','unsure','disoriented','baffled','mystified'], 'color': '#8b5cf6' },
    'excited':  { 'keywords': ['excited','thrilled','enthusiastic','eager','pumped','energized','hyped','stoked','exhilarated','animated'], 'color': '#ec4899' },
    'surprised':{ 'keywords': ['surprised','shocked','amazed','astonished','wow','unexpected','incredible','unbelievable','omg','stunning','startled'], 'color': '#a855f7' },
    'disgusted':{ 'keywords': ['disgusted','disgust','revolted','repulsed','sickened','nauseated','gross','yuck','ew','appalled'], 'color': '#059669' },
    'calm':     { 'keywords': ['calm','peaceful','relaxed','serene','tranquil','composed','mellow','chill','content','soothed'], 'color': '#06b6d4' },
    'neutral':  { 'keywords': [], 'color': '#6b7280' },
}

EMOTION_LABELS = {
    'happy': 'Happy', 'sad': 'Sad', 'angry': 'Angry', 'fearful': 'Fearful',
    'anxious': 'Anxious', 'stressed': 'Stressed', 'worried': 'Worried',
    'confused': 'Confused', 'excited': 'Excited', 'surprised': 'Surprised',
    'disgusted': 'Disgusted', 'calm': 'Calm', 'neutral': 'Neutral',
}

def classify_emotion(text: str) -> list:
    scores = {k: 0.0 for k in EMOTION_CONFIG}
    scores['neutral'] = 30.0  # base neutral score

    lower = text.lower()

    for emotion, cfg in EMOTION_CONFIG.items():
        if emotion == 'neutral':
            continue
        for keyword in cfg['keywords']:
            matches = len(re.findall(re.escape(keyword), lower))
            scores[emotion] += matches * 25

    # Punctuation bonuses
    exclamations = len(re.findall(r'!', text))
    questions    = len(re.findall(r'\?', text))
    ellipses     = len(re.findall(r'\.\.\.', text))

    scores['surprised'] += questions * 8
    scores['excited']   += exclamations * 6
    scores['happy']     += exclamations * 4
    scores['worried']   += ellipses * 10
    scores['anxious']   += ellipses * 8

    total = sum(scores.values())

    # Normalize so all scores sum to exactly 100
    if total == 0:
        scores['neutral'] = 100.0
        normalized = scores
    else:
        normalized = {k: (v / total) * 100.0 for k, v in scores.items()}

    result = []
    for key, cfg in EMOTION_CONFIG.items():
        result.append({
            'emotion': EMOTION_LABELS[key],
            'score': round(normalized[key], 1),
            'color': cfg['color'],
        })

    result.sort(key=lambda x: x['score'], reverse=True)
    return result
