import time
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2
from .common import ROOT, read_json, write_json
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .attendance_manager import simulation_info


def session_path(session):
    root = (ROOT / 'sessions').resolve()
    path = (root / session).resolve()
    if path.parent != root or not path.is_dir(): raise ValueError('Choose an existing session ID')
    return path


def recognize(session):
    directory = session_path(session)
    image = cv2.imread(str(directory / 'classroom.png'))
    if image is None: raise ValueError('Missing or unreadable classroom image')
    recognizer = FaceRecognizer()
    start = time.perf_counter()
    detections = FaceDetector(recognizer.config['detector_backend']).detect(image)
    predictions, errors = [], []
    for face in detections:
        try:
            prediction = recognizer.predict(face['crop'])
        except Exception as exc:
            errors.append(str(exc))
            prediction = {'name': 'Unknown', 'distance': None, 'margin': None, 'candidates': [], 'reason': str(exc)}
        prediction['box'] = face['box']
        predictions.append(prediction)
        color = (0, 150, 0) if prediction['name'] not in ('Unknown', 'Needs Review') else (0, 150, 255) if prediction['name'] == 'Needs Review' else (0, 0, 220)
        x,y,w,h = face['box']
        cv2.rectangle(image, (x,y), (x+w,y+h), color, 2)
        distance = prediction['distance']
        text = prediction['name'] + (f' d={distance:.3f}' if distance is not None else '')
        cv2.putText(image, text, (x, max(20,y-7)), 0, .48, color, 1, cv2.LINE_AA)
    accepted = set(p['name'] for p in predictions if p['name'] not in ('Unknown','Needs Review'))
    # Duplicate assignment cannot count as confident unique attendance.
    accepted = {name for name in accepted if sum(p['name'] == name for p in predictions) == 1}
    info = simulation_info(session)
    caption = f'Detected: {len(detections)} | Recognized students: {len(accepted)}'
    if info: caption += f" | Absences prevues (simulation): {info['expected_absent_count']}"
    cv2.putText(image,caption,(20,image.shape[0]-15),0,.55,(30,30,30),2)
    result = {'session_id': session, 'timestamp': datetime.now(ZoneInfo('Africa/Casablanca')).isoformat(),
              'faces': predictions, 'detected_count': len(detections), 'recognized_count': len(accepted),
              'inference_ms': (time.perf_counter()-start)*1000, 'errors': errors,
              'configuration': recognizer.config}
    if info: result['simulation'] = info
    cv2.imwrite(str(directory / 'annotated.png'), image)
    write_json(directory / 'predictions.json', result)
    return result


def compare_truth(session):
    """Optional evaluation AFTER recognition; never called by prediction or attendance."""
    directory = session_path(session)
    truth, result = read_json(directory / 'ground_truth.json'), read_json(directory / 'predictions.json')
    faces = result['faces']; unmatched = set(range(len(faces))); matches = []
    for person in truth['students']:
        x,y,w,h = person['portrait_box']
        indices = [i for i in unmatched if x <= faces[i]['box'][0]+faces[i]['box'][2]/2 <= x+w
                   and y <= faces[i]['box'][1]+faces[i]['box'][3]/2 <= y+h]
        if indices:
            index = max(indices, key=lambda i: faces[i]['box'][2]*faces[i]['box'][3])
            unmatched.remove(index)
            matches.append({'truth': person['name'], 'prediction': faces[index]['name'], 'detected': True})
        else: matches.append({'truth': person['name'], 'prediction': 'Missed detection', 'detected': False})
    report = {'matches': matches, 'extra_detections': len(unmatched),
              'detection_recall': sum(r['detected'] for r in matches)/len(matches) if matches else None,
              'correct_identity_count': sum(r['truth'] == r['prediction'] for r in matches),
              'ground_truth_count': len(matches), 'detected_count': result['detected_count']}
    write_json(directory / 'comparison.json', report)
    return report
