"""Held-out evaluation, with rejection explicitly counted as a missed identity."""
import time
import cv2
import numpy as np
import pandas as pd
from .common import ROOT, IMAGE_EXTENSIONS, read_json, write_json
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer


def metrics(rows, names):
    labels = names + ['Unknown', 'Needs Review']
    confusion = np.zeros((len(labels), len(labels)), int)
    for r in rows: confusion[labels.index(r['truth']), labels.index(r['prediction'])] += 1
    per_class = []
    for i, name in enumerate(names):
        tp, fp, fn = confusion[i, i], confusion[:, i].sum()-confusion[i, i], confusion[i].sum()-confusion[i, i]
        precision, recall = float(tp / max(1, tp+fp)), float(tp / max(1, tp+fn))
        per_class.append({'name': name, 'precision': precision, 'recall': recall,
                          'f1': 2*precision*recall/max(1e-12, precision+recall), 'support': int(confusion[i].sum())})
    known = [r for r in rows if r['truth'] != 'Unknown']
    unknown = [r for r in rows if r['truth'] == 'Unknown']
    return {'count': len(rows), 'accuracy': sum(r['truth'] == r['prediction'] for r in rows)/max(1, len(rows)),
        'macro_precision': float(np.mean([r['precision'] for r in per_class])),
        'macro_recall': float(np.mean([r['recall'] for r in per_class])),
        'macro_f1': float(np.mean([r['f1'] for r in per_class])), 'per_class': per_class,
        'false_acceptance_rate_unknown': sum(r['prediction'] in names for r in unknown)/len(unknown) if unknown else None,
        'false_rejection_rate_known': sum(r['prediction'] in ('Unknown', 'Needs Review') for r in known)/len(known) if known else None,
        'wrong_identity_acceptance_rate_known': sum(r['prediction'] in names and r['prediction'] != r['truth'] for r in known)/max(1,len(known)),
        'mean_inference_ms': float(np.mean([r['inference_ms'] for r in rows])) if rows else None,
        'labels': labels, 'confusion_matrix': confusion.tolist(), 'unknown_count': len(unknown)}


def evaluate():
    manifest = read_json(ROOT / 'reports/dataset_manifest.json')
    recognizer = FaceRecognizer()
    names = sorted(set(recognizer.labels.tolist()))
    train = [r for r in manifest if r['split'] == 'train']
    lbph = cv2.face.LBPHFaceRecognizer_create()
    encode = {name: i for i, name in enumerate(names)}
    def gray(crop): return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (200, 200))
    lbph.train([gray(cv2.imread(str(ROOT / r['path']))) for r in train],
               np.array([encode[r['name']] for r in train], dtype=np.int32))
    validation = []
    for r in manifest:
        if r['split'] == 'validation':
            label, distance = lbph.predict(gray(cv2.imread(str(ROOT / r['path']))))
            validation.append((names[label] == r['name'], distance))
    # Strictest distance that admits the most correct validation samples with zero wrong accepts.
    thresholds = [0.] + [float(d)+1e-6 for _, d in validation]
    safe = [t for t in thresholds if not any(not correct and d < t for correct, d in validation)]
    threshold = max(safe, key=lambda t: (sum(correct and d < t for correct, d in validation), -t))
    lbph.save(str(ROOT / 'embeddings/lbph_baseline.yml'))
    samples = [(r['name'], r['path'], cv2.imread(str(ROOT / r['path']))) for r in manifest if r['split'] == 'test']
    unknown_errors = []
    detector = FaceDetector(recognizer.config['detector_backend'])
    for path in sorted((ROOT / 'unknown_dataset').rglob('*')):
        if path.suffix.lower() not in IMAGE_EXTENSIONS: continue
        try:
            faces = detector.detect(cv2.imread(str(path)))
            if len(faces) != 1: raise ValueError('Expected exactly one face')
            samples.append(('Unknown', path.relative_to(ROOT).as_posix(), faces[0]['crop']))
        except Exception as exc: unknown_errors.append({'path': str(path), 'error': str(exc)})
    outputs = {'FaceNet512': [], 'LBPH': []}
    from deepface import DeepFace
    DeepFace.build_model('Facenet512')  # Measure warm CPU inference; startup is excluded.
    for truth, path, crop in samples:
        for model in outputs:
            start = time.perf_counter()
            if model == 'FaceNet512':
                # Clear this crop's cache to measure actual warm-model inference, not disk reads.
                import hashlib
                key = hashlib.sha256(recognizer.embedder.signature.encode()+crop.tobytes()).hexdigest()
                cache = recognizer.embedder.directory / f'{key}.npy'
                if cache.exists(): cache.unlink()
                prediction = recognizer.predict(crop)
                name, distance = prediction['name'], prediction['distance']
            else:
                label, distance = lbph.predict(gray(crop))
                name = names[label] if distance < threshold else 'Unknown'
            outputs[model].append({'truth': truth, 'prediction': name, 'path': path,
                                  'distance': distance, 'inference_ms': (time.perf_counter()-start)*1000})
    reports = {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for model, rows in outputs.items():
        result = metrics(rows, names)
        reports[model] = result
        pd.DataFrame(rows).to_csv(ROOT / f'reports/{model}_predictions.csv', index=False)
        edge = max(8,len(result['labels'])*.45)
        fig, ax = plt.subplots(figsize=(edge, edge))
        cm = np.array(result['confusion_matrix']); ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(result['labels'])), result['labels'], rotation=45, ha='right')
        ax.set_yticks(range(len(result['labels'])), result['labels'])
        for i, j in np.ndindex(cm.shape): ax.text(j, i, str(cm[i, j]), ha='center')
        ax.set(xlabel='Prediction', ylabel='Truth', title=f'{model}: independent held-out originals')
        fig.tight_layout(); fig.savefig(ROOT / f'reports/{model}_confusion.png'); plt.close(fig)
    write_json(ROOT / 'reports/evaluation.json', {'models': reports, 'lbph_threshold': threshold,
        'unknown_errors': unknown_errors, 'limitations': [
            'Only one held-out test original per registered identity; uncertainty is high.',
            'Unknown FAR is unavailable (null) without usable independent unknown portraits.',
            'Test crops are pre-detected; inference timing excludes detection and model startup. Embedding disk-cache hits are disabled for timed test inference.',
            'Augmentation is not independent evidence or a substitute for distinct source portraits.']})
    generation = ROOT/'reports/synthetic_sources/generation_jobs.json'
    if generation.exists():
        report = read_json(ROOT/'reports/evaluation.json')
        report['limitations'].append('New fictional students have jointly generated contact-sheet views. Panel-level test results are correlated synthetic demonstration measurements, not independent photographic generalization evidence.')
        write_json(ROOT/'reports/evaluation.json',report)
    return reports
