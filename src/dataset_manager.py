"""Audit originals, group duplicate sources, split BEFORE augmentation."""
import hashlib
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from .common import ROOT, IMAGE_EXTENSIONS, roster, write_json, read_json
from .face_detector import FaceDetector


def augment(image):
    """Small identity-preserving perturbations; training only."""
    yield 'bright', cv2.convertScaleAbs(image, alpha=1.04, beta=10)
    yield 'dim', cv2.convertScaleAbs(image, alpha=.94, beta=-5)
    matrix = cv2.getRotationMatrix2D((80, 80), 5, 1)
    yield 'rotate', cv2.warpAffine(image, matrix, (160, 160), borderMode=cv2.BORDER_REFLECT_101)
    yield 'scale', cv2.resize(cv2.resize(image, (125, 125)), (160, 160))


def phash(image):
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32))
    values = cv2.dct(np.float32(small))[:8, :8].flatten()[1:]
    return values > np.median(values)


def prepare(backend='yunet', seed=42):
    detector = FaceDetector(backend)
    manifest_path = ROOT/'reports/dataset_manifest.json'
    quality_path = ROOT/'reports/dataset_quality.json'
    previous = read_json(manifest_path) if manifest_path.exists() else []
    preserve_splits = quality_path.exists() and read_json(quality_path).get('seed') == seed
    old_splits = {}
    for item in previous:
        if not item['augmentation']:
            old_splits.setdefault(item['name'],{})[item['sha256']] = item['split']
    excluded_path = ROOT / 'dataset_selection.json'
    excluded = read_json(excluded_path) if excluded_path.exists() else {}
    rows, usable = [], []
    for student in roster():
        name = student['Name']
        for path in sorted((ROOT / 'dataset' / name).rglob('*')):
            if not path.is_file():
                continue
            row = {'name': name, 'source': path.relative_to(ROOT).as_posix(),
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'issues': []}
            image = cv2.imread(str(path)) if path.suffix.lower() in IMAGE_EXTENSIONS else None
            if image is None:
                row['issues'].append('corrupt_or_unsupported')
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                row.update(width=image.shape[1], height=image.shape[0], brightness=float(gray.mean()))
                faces = detector.detect(image)
                row['face_count'] = len(faces)
                if len(faces) != 1:
                    row['issues'].append('expected_exactly_one_face')
                else:
                    face = faces[0]
                    blur = float(cv2.Laplacian(cv2.cvtColor(face['crop'], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                    row.update(blur=blur, face_size=min(face['box'][2:]), eyes_available=face['eyes_available'])
                    if blur < 35: row['issues'].append('blurry')
                    if row['face_size'] < 60: row['issues'].append('small_face')
                    if not 35 < row['brightness'] < 225: row['issues'].append('exposure')
                    if row['source'] in excluded:
                        row['issues'].append('excluded_by_operator: ' + excluded[row['source']])
                    if not row['issues']:
                        usable.append((row, face['crop'], phash(image)))
            rows.append(row)
    # Connected components make duplicate grouping transitive across identities.
    parent = list(range(len(usable)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    duplicate_pairs = []
    for i, (a, _, ha) in enumerate(usable):
        for j in range(i):
            b, _, hb = usable[j]
            if a['sha256'] == b['sha256'] or np.count_nonzero(ha != hb) <= 3:
                parent[find(i)] = find(j)
                duplicate_pairs.append([a['source'], b['source']])
    groups = {}
    for i, item in enumerate(usable): groups.setdefault(find(i), []).append(item)
    by_name = {}
    for group in groups.values():
        if len({v[0]['name'] for v in group}) > 1:
            for row, _, _ in group: row['issues'].append('cross_label_duplicate_review')
            continue
        row, crop, _ = group[0]
        row['duplicate_sources'] = [v[0]['source'] for v in group[1:]]
        by_name.setdefault(row['name'], []).append((row, crop))
        for duplicate, _, _ in group[1:]: duplicate['issues'].append('duplicate_excluded')
    # Outputs belong to this pipeline; build in staging to avoid stale samples on reruns.
    staging = ROOT / 'reports' / 'preprocessing_staging'
    if staging.exists(): shutil.rmtree(staging)
    manifest, warnings = [], []
    if not by_name:
        raise ValueError('No usable independent portraits remain; inspect quality or dataset_selection.json')
    for name, entries in sorted(by_name.items()):
        prior = old_splits.get(name,{})
        unchanged = preserve_splits and set(prior) == {row['sha256'] for row,_ in entries}
        # Adding another student must not reshuffle existing students' test data.
        student_seed = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8],'little')
        np.random.default_rng(student_seed).shuffle(entries)
        if len(entries) < 4:
            warnings.append(f'{name}: fewer than four independent sources; add portraits before reliable calibration/evaluation.')
        n_test = 1 if len(entries) >= 3 else 0
        n_val = 1 if len(entries) >= 4 else 0
        for i, (row, crop) in enumerate(entries):
            split = prior[row['sha256']] if unchanged else 'test' if i < n_test else 'validation' if i < n_test+n_val else 'train'
            base = 'test_dataset' if split == 'test' else 'processed_dataset'
            relative = f'{base}/{name}/{row["sha256"][:16]}.png'
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), crop): raise IOError(target)
            manifest.append({'name': name, 'source': row['source'], 'sha256': row['sha256'],
                             'split': split, 'path': relative, 'augmentation': False})
            if split == 'train':
                for suffix, aug in augment(crop):
                    rel = f'{base}/{name}/{row["sha256"][:16]}_{suffix}.png'
                    cv2.imwrite(str(staging / rel), aug)
                    manifest.append({**manifest[-1], 'name': name, 'source': row['source'],
                        'sha256': row['sha256'], 'split': split, 'path': rel, 'augmentation': True})
    for directory in ('processed_dataset', 'test_dataset'):
        output = ROOT / directory
        if output.exists(): shutil.rmtree(output)
        source = staging / directory
        if source.exists(): shutil.move(str(source), str(output))
        else: output.mkdir()
    write_json(ROOT / 'reports/dataset_manifest.json', manifest)
    write_json(ROOT / 'reports/dataset_quality.json', {'backend': backend, 'seed': seed,
        'original_count': len(rows), 'usable_unique_count': sum(len(v) for v in by_name.values()),
        'duplicate_pairs': duplicate_pairs, 'warnings': warnings, 'images': rows})
    pd.DataFrame(rows).to_csv(ROOT / 'reports/dataset_quality.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(max(10,len(roster())*.6), 5))
    pd.Series([r.get('blur', 0) for r in rows]).plot.hist(ax=axes[0], bins=12, title='Face sharpness (Laplacian variance)')
    pd.DataFrame(manifest).groupby(['name', 'split']).size().unstack(fill_value=0).plot.bar(ax=axes[1], title='Split counts (includes train augmentation)')
    fig.tight_layout(); fig.savefig(ROOT / 'reports/dataset_quality.png'); plt.close(fig)
    # Every original appears in the contact sheet, including rejected images.
    tiles = []
    for row in rows:
        image = cv2.imread(str(ROOT / row['source']))
        tile = np.full((170, 140, 3), 240, np.uint8)
        if image is not None:
            from .face_detector import letterbox
            tile[:140] = letterbox(image, 140)
        cv2.putText(tile, f"{row['name']}/{row['source'].split('/')[-1]}", (3, 158), 0, .35, (0, 0, 0), 1)
        tiles.append(tile)
    if tiles:
        while len(tiles) % 5: tiles.append(np.full_like(tiles[0], 240))
        cv2.imwrite(str(ROOT / 'reports/contact_sheet.jpg'), np.vstack([np.hstack(tiles[i:i+5]) for i in range(0, len(tiles), 5)]))
    return {'originals': len(rows), 'samples': len(manifest), 'warnings': warnings}


def import_portrait(name, source):
    """Explicitly import a new fictional portrait, without overwriting an original."""
    if name not in {r['Name'] for r in roster()}: raise ValueError('Choose an existing student label')
    source = Path(source)
    image = cv2.imread(str(source))
    if image is None or len(FaceDetector().detect(image)) != 1:
        raise ValueError('Replacement must contain exactly one detectable face')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target = ROOT / 'dataset' / name / f'import_{digest}{source.suffix.lower()}'
    if target.exists(): raise FileExistsError('This portrait was already imported')
    shutil.copy2(source, target)
    return str(target)


def exclude_set(name):
    """Retire an old portrait set from processing while retaining every original file."""
    if name not in {r['Name'] for r in roster()}: raise ValueError('Unknown student label')
    path = ROOT / 'dataset_selection.json'
    selection = read_json(path) if path.exists() else {}
    for source in (ROOT / 'dataset' / name).rglob('*'):
        if source.is_file():
            selection[source.relative_to(ROOT).as_posix()] = 'Portrait set retired for independent synthetic replacements'
    write_json(path, selection)
    return {'name': name, 'excluded_sources': sum(p.startswith(f'dataset/{name}/') for p in selection),
            'next_step': 'Import at least five independent new synthetic portraits, then prepare, build and evaluate.'}
