def combine_emotions(sets: list) -> list:
    combined: dict = {}
    for emotion_set in sets:
        for item in emotion_set:
            key = item['emotion']
            if key not in combined:
                combined[key] = {'score': 0.0, 'color': item['color'], 'count': 0}
            combined[key]['score'] += item['score']
            combined[key]['count'] += 1

    result = []
    for emotion, data in combined.items():
        result.append({
            'emotion': emotion,
            'score': data['score'] / data['count'],
            'color': data['color'],
        })
    result.sort(key=lambda x: x['score'], reverse=True)
    return result
