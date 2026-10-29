"""Reproducible synthetic collages; truth is stored separately from predictions."""
from datetime import datetime
from uuid import uuid4
import math
import cv2
import numpy as np
from .common import ROOT, read_json, write_json, roster


def generate(count=None, seed=None, split='test', rotation=7):
    manifest = read_json(ROOT / 'reports/dataset_manifest.json')
    available = {}
    for row in manifest:
        if row['split'] == split and not row['augmentation']:
            available.setdefault(row['name'], []).append(row)
    rng = np.random.default_rng(seed)
    if count is None: count = int(rng.integers(0, len(available)+1))
    if not 0 <= count <= len(available): raise ValueError(f'Count must be 0..{len(available)} for split {split}')
    if not 0 <= rotation <= 15: raise ValueError('Use moderate rotation between 0 and 15 degrees')
    chosen = rng.choice(sorted(available), count, replace=False).tolist()
    slots = max(6, count)
    columns = max(3, math.ceil(math.sqrt(slots)))
    rows = math.ceil(slots / columns)
    image = np.full((110+rows*270, columns*310, 3), (220, 225, 229), np.uint8)
    cv2.rectangle(image, (35, 15), (image.shape[1]-35, 85), (67, 86, 66), -1)
    cv2.putText(image, 'VIRTUAL CLASSROOM', (60, 59), 0, 1, (245, 245, 245), 2)
    truth = []
    positions = rng.permutation(columns*rows)[:count]
    for name, slot in zip(chosen, positions):
        record = available[name][int(rng.integers(len(available[name])))]
        # Use the ORIGINAL held-out portrait, so recognition must detect faces anew.
        source = cv2.imread(str(ROOT / record['source']))
        if source is None: raise ValueError(f'Unreadable source {record["source"]}')
        size = int(rng.integers(190, 231))
        scale = size / max(source.shape[:2])
        portrait = cv2.resize(source, (round(source.shape[1]*scale), round(source.shape[0]*scale)))
        h, w = portrait.shape[:2]
        angle = float(rng.uniform(-rotation, rotation))
        portrait = cv2.warpAffine(portrait, cv2.getRotationMatrix2D((w/2, h/2), angle, 1), (w,h), borderMode=cv2.BORDER_REFLECT_101)
        brightness = float(rng.uniform(.90, 1.10))
        portrait = cv2.convertScaleAbs(portrait, alpha=brightness)
        x = int(slot % columns)*310 + int(rng.integers(20, 60))
        y = 115 + int(slot // columns)*270 + int(rng.integers(0, 20))
        image[y:y+h, x:x+w] = portrait
        cv2.rectangle(image, (x-8,y+h+4), (x+w+8,y+h+15), (115, 136, 157), -1)
        truth.append({'name': name, 'source': record['source'], 'portrait_box': [x,y,w,h],
                      'rotation': angle, 'brightness': brightness})
    session = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    directory = ROOT / 'sessions' / session
    directory.mkdir(parents=True)
    cv2.imwrite(str(directory / 'classroom.png'), image)
    write_json(directory / 'ground_truth.json', {'session_id': session, 'seed': seed, 'split': split, 'students': truth})
    # Count-only simulation metadata is safe for UI/coverage checks. Identity labels
    # remain in ground_truth.json and are never passed to recognition/attendance.
    write_json(directory/'simulation.json',{'kind':'synthetic_classroom','session_id':session,
        'registered_count':len(roster()),'generated_count':count,
        'expected_absent_count':len(roster())-count})
    return session
