import hashlib
import cv2
import numpy as np
from .common import ROOT, read_json, write_json, roster
from .face_embeddings import FaceEmbeddings


def manifest_digest():
    path = ROOT / 'reports/dataset_manifest.json'
    digest = hashlib.sha256(path.read_bytes())
    digest.update((ROOT/'students.json').read_bytes())
    digest.update(read_json(ROOT / 'reports/dataset_quality.json')['backend'].encode())
    # Detector/alignment changes can produce different crops with the same source manifest.
    for row in read_json(path):
        digest.update(hashlib.sha256((ROOT / row['path']).read_bytes()).digest())
    return digest.hexdigest()


def class_scores(vector, vectors, labels):
    """Average the closest TWO independent originals per identity, never one neighbor."""
    scores = {}
    for name in sorted(set(labels)):
        distances = np.sort(np.clip(1 - vectors[labels == name] @ vector, 0, 2))
        if len(distances) >= 2: scores[name] = float(np.mean(distances[:2]))
    return sorted(scores.items(), key=lambda item: item[1])


def decide(scores, threshold, margin, blocked=(), thresholds=None, review_thresholds=None):
    if len(scores) < 2:
        return {'name': 'Unknown', 'distance': None, 'margin': None, 'candidates': [], 'reason': 'Insufficient gallery'}
    (name, distance), (_, runner) = scores[:2]
    gap = runner - distance
    limit = (thresholds or {}).get(name, threshold)
    peers = list(blocked.get(name, [])) if isinstance(blocked, dict) else []
    review_limit = max((review_thresholds or {}).get(candidate, limit) for candidate in [name]+peers)
    candidates = [n for n, d in scores if d-distance < margin or n == name]
    if name in blocked and distance <= review_limit:
        status, reason = 'Needs Review', 'Overlapping identities: independent replacement portraits required'
        candidates = list(dict.fromkeys([name] + list(blocked.get(name, [])))) if isinstance(blocked, dict) else candidates
    elif distance > limit:
        status, candidates, reason = 'Unknown', [], 'Distance exceeds threshold'
    elif gap < margin or name in blocked:
        status, reason = 'Needs Review', 'Ambiguous identity or overlapping gallery'
        candidates = list(dict.fromkeys(candidates + list(blocked.get(name, [])))) if isinstance(blocked, dict) else candidates
    else:
        status, candidates, reason = name, [name], 'Accepted'
    return {'name': status, 'distance': distance, 'margin': gap, 'candidates': candidates, 'reason': reason,
            'best_candidate': str(name), 'runner_up': str(scores[1][0]),
            'threshold': float(limit), 'required_margin': float(margin)}


def build_gallery():
    manifest = read_json(ROOT / 'reports/dataset_manifest.json')
    embedder = FaceEmbeddings()
    records = [r for r in manifest if not r['augmentation']]
    vectors = {r['path']: embedder.embed(cv2.imread(str(ROOT / r['path']))) for r in records}
    train = [r for r in records if r['split'] == 'train']
    names = np.array([r['name'] for r in train])
    matrix = np.array([vectors[r['path']] for r in train])
    if not train or any(sum(names == r['Name']) < 2 for r in roster()):
        raise ValueError('At least two independent usable training portraits per registered student are required')
    # Calibrate using validation only. Test samples are never used to choose rules.
    validation = [r for r in records if r['split'] == 'validation']
    if set(r['name'] for r in validation) != set(names):
        raise ValueError('Every student needs a held-out validation original. Add portraits and prepare again.')
    # Pairwise train+validation overlap is a dataset warning and a conservative safety gate.
    development = [r for r in records if r['split'] != 'test']
    classes = sorted(set(names))
    within, similarity, overlap, blocked = {}, np.zeros((len(classes), len(classes))), [], {}
    for a in classes:
        va = np.array([vectors[r['path']] for r in development if r['name'] == a])
        own = np.clip(1-va @ va.T, 0, 2)[np.triu_indices(len(va), 1)]
        within[a] = float(np.max(own))
    for i, a in enumerate(classes):
        va = np.array([vectors[r['path']] for r in development if r['name'] == a])
        for j, b in enumerate(classes):
            vb = np.array([vectors[r['path']] for r in development if r['name'] == b])
            cross = np.clip(1-va @ vb.T, 0, 2)
            similarity[i, j] = float(cross.mean())
            if i < j and float(cross.min()) <= max(within[a], within[b]):
                overlap.append({'identities': [a, b], 'minimum_cross_distance': float(cross.min()),
                                'maximum_within_distance': max(within[a], within[b])})
                blocked.setdefault(a, []).append(b); blocked.setdefault(b, []).append(a)
    from .calibration import validation_views, distance_boundaries
    from .face_detector import FaceDetector
    backend = read_json(ROOT / 'reports/dataset_quality.json')['backend']
    detector = FaceDetector(backend)
    trials, views, failed_views = [], [], []
    for record in validation:
        for view, crop in validation_views(record, detector):
            meta = {'name': record['name'], 'source': record['source'], 'view': view}
            if crop is None:
                failed_views.append(meta)
                continue
            scores = class_scores(embedder.embed(crop), matrix, names)
            trials.append((record['name'], scores)); views.append(meta)
    thresholds, review_thresholds, statistics = distance_boundaries(trials, classes)
    best, chosen_margin, curve = (-1, 0), 2.1, []
    for margin in (.03, .05, .08, .10, .15, .20, .30, 2.1):
        predictions = [decide(s, 0., margin, blocked, thresholds, review_thresholds)['name'] for _, s in trials]
        wrong = sum(p not in (n, 'Unknown', 'Needs Review') for (n, _), p in zip(trials, predictions))
        correct = sum(p == n for (n, _), p in zip(trials, predictions))
        curve.append({'margin': margin, 'correct': correct, 'wrong_accepts': wrong})
        if wrong == 0 and (correct, margin) > best:
            best, chosen_margin = (correct, margin), margin
    config = {'threshold': float(min(thresholds.values())), 'thresholds': thresholds,
              'review_thresholds': review_thresholds, 'margin': chosen_margin, 'blocked': blocked,
              'calibration_method': 'validation_genuine_impostor_boundaries_with_classroom_views_v2',
              'manifest_digest': manifest_digest(), 'embedding_signature': embedder.signature,
              'detector_backend': backend,
              'validation_count': len(validation), 'unknown_calibration': False,
              'note': 'Small closed-set validation; unknown rejection is provisional until independent unknown portraits are supplied.'}
    np.savez_compressed(ROOT / 'embeddings/gallery.npz', vectors=matrix, labels=names)
    write_json(ROOT / 'embeddings/config.json', config)
    write_json(ROOT / 'reports/calibration.json', {'config': config, 'grid': curve, 'validation_trials': trials,
        'class_statistics': statistics, 'validation_views': views, 'failed_detection_views': failed_views,
        'validation_originals': len(validation),
        'independent_validation_originals': None if (ROOT/'reports/synthetic_sources/generation_jobs.json').exists() else len(validation),
        'note': 'Transformed views are correlated robustness checks, not additional independent validation identities or test images.'})
    write_json(ROOT / 'reports/identity_overlap.json', {'within_maximum': within, 'overlapping_pairs': overlap,
        'replacement_sources': [r['source'] for r in development if r['name'] in blocked],
        'guidance': 'Replace visually overlapping fictional portrait sets with independently created, clearly distinguishable portraits. Augmentation does not create independent identities. Re-run prepare and build after importing.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    edge = max(7,len(classes)*.5)
    fig, ax = plt.subplots(figsize=(edge, edge)); im = ax.imshow(similarity, vmin=0, vmax=max(.1, similarity.max()))
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha='right'); ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)): ax.text(j, i, f'{similarity[i,j]:.3f}', ha='center', color='white', fontsize=7)
    ax.set_title('Mean development cosine distances (train + validation)')
    fig.colorbar(im); fig.tight_layout(); fig.savefig(ROOT / 'reports/embedding_similarity.png'); plt.close(fig)
    return config


class FaceRecognizer:
    def __init__(self):
        self.config = read_json(ROOT / 'embeddings/config.json')
        if self.config['manifest_digest'] != manifest_digest():
            raise ValueError('Dataset split, detector or processed crops changed. Rebuild the gallery before recognition.')
        data = np.load(ROOT / 'embeddings/gallery.npz', allow_pickle=False)
        self.vectors, self.labels = data['vectors'], data['labels']
        self.embedder = FaceEmbeddings()

    def predict(self, crop):
        return decide(class_scores(self.embedder.embed(crop), self.vectors, self.labels),
                      self.config['threshold'], self.config['margin'], self.config['blocked'],
                      self.config.get('thresholds'), self.config.get('review_thresholds'))
