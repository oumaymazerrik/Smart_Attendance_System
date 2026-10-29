"""Integrate generated contact-sheet panels without modifying existing portraits.

This module extracts image assets; it does not synthesize identities or augment
existing faces. Generation prompts/provenance are retained in a separate report.
"""
import shutil
from pathlib import Path
import cv2
import numpy as np
from .common import ROOT, read_json, write_json, roster
from .face_detector import FaceDetector
from .dataset_manager import phash


def integrate_generated(manifest_path, activate=False):
    manifest = read_json(manifest_path)
    detector = FaceDetector()
    staged, pending, details = [], [], []
    source_directory = ROOT/'reports/synthetic_sources'
    source_directory.mkdir(parents=True,exist_ok=True)
    for student in manifest['students']:
        name, source = student['name'], student.get('sheet')
        if not source:
            pending.append(name); continue
        source = Path(source)
        if not source.is_absolute(): source = ROOT/source
        if not source.exists():
            pending.append(name); continue
        if Path(name).name != name or name in ('.','..'):
            raise ValueError('Invalid student directory name')
        saved_sheet = source_directory/f'{name}.png'
        if not saved_sheet.exists(): shutil.copy2(source,saved_sheet)
        student['sheet'] = saved_sheet.relative_to(ROOT).as_posix()
        image = cv2.imread(str(saved_sheet))
        if image is None: raise ValueError(f'Unreadable contact sheet: {name}')
        height,width = image.shape[:2]
        if abs(width/height-1.5) > .03:
            raise ValueError(f'{name}: expected a 3-column by 2-row contact sheet')
        folder = source_directory/'panels'/name
        folder.mkdir(parents=True,exist_ok=True)
        hashes, files = [], []
        for row in range(2):
            for column in range(3):
                panel = image[round(row*height/2):round((row+1)*height/2),
                              round(column*width/3):round((column+1)*width/3)]
                faces = detector.detect(panel)
                if len(faces) != 1:
                    raise ValueError(f'{name}, panel {row*3+column+1}: expected one face, found {len(faces)}')
                face = faces[0]
                blur = cv2.Laplacian(cv2.cvtColor(face['crop'],cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
                if min(face['box'][2:]) < 60 or blur < 35:
                    raise ValueError(f'{name}: generated portrait does not pass face size/sharpness checks')
                path = folder/f'synthetic_{row*3+column+1:02}.png'
                if not path.exists(): cv2.imwrite(str(path),panel)
                files.append(path); hashes.append(phash(panel))
        # Count connected source groups using the same duplicate rule as preprocessing.
        parent = list(range(6))
        def find(i):
            while parent[i] != i: i=parent[i]
            return i
        for i in range(6):
            for j in range(i):
                if np.count_nonzero(hashes[i] != hashes[j]) <= 3: parent[find(i)] = find(j)
        groups = len({find(i) for i in range(6)})
        details.append({'name':name,'panels':6,'retained_duplicate_groups':groups})
        if groups < 4: raise ValueError(f'{name}: only {groups} distinct view groups; add independent portraits')
        staged.append((name,files))
    write_json(source_directory/'integration_audit.json',{'students':details,'pending':pending,
               'activated':activate and not pending,'provenance':manifest['correlation_note']})
    write_json(source_directory/'generation_jobs.json',manifest)
    if activate:
        if pending: raise ValueError(f'Generation is incomplete: {", ".join(pending)}')
        students = roster(); names = {s['Name'] for s in students}
        for name,files in staged:
            target = ROOT/'dataset'/name
            target.mkdir(exist_ok=True)
            for path in files:
                destination = target/path.name
                if destination.exists() and destination.read_bytes() != path.read_bytes():
                    raise FileExistsError(f'Refusing to overwrite {destination}')
                if not destination.exists(): shutil.copy2(path,destination)
            if name not in names:
                next_id = max(int(s['Student_ID'][1:]) for s in students)+1
                students.append({'Student_ID':f'E{next_id:03}','Name':name}); names.add(name)
        write_json(ROOT/'students.json',students)
    return {'ready':len(staged),'pending':pending,'activated':activate and not pending}
