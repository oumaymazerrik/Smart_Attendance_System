"""Reproducible integration check for the expanded synthetic student registry."""
import hashlib
import cv2
import pandas as pd
from .common import ROOT, read_json, write_json, roster
from .classroom_generator import generate
from .session_runner import recognize, compare_truth
from .attendance_manager import export_attendance


def validate_expansion():
    students = roster()
    assert len(students) == 25, 'Expected exactly 25 registered fictional identities'
    original_roster = read_json(ROOT/'reports/before_25_students/students.json')
    assert students[:len(original_roster)] == original_roster, 'Original student IDs changed'
    old_manifest = read_json(ROOT/'reports/before_25_students/dataset_manifest.json')
    manifest = read_json(ROOT/'reports/dataset_manifest.json')
    old_splits = {r['source']:r['split'] for r in old_manifest if not r['augmentation']}
    current_splits = {r['source']:r['split'] for r in manifest if not r['augmentation']}
    assert all(current_splits.get(source)==split for source,split in old_splits.items())
    old_audit = read_json(ROOT/'reports/before_25_students/dataset_quality.json')
    for row in old_audit['images']:
        assert hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest() == row['sha256']
    expected = {s['Name'] for s in students}
    for split in ('train','validation','test'):
        assert {r['name'] for r in manifest if r['split']==split} == expected
    session = generate(count=25,seed=2026)
    result = recognize(session)  # Predictions are produced before loading ground truth.
    table = export_attendance(session,result,completed=True,coverage_adequate=False)
    comparison = compare_truth(session)
    csv = pd.read_csv(ROOT/'attendance_reports'/f'{session}.csv')
    assert len(csv)==25 and csv.Student_ID.is_unique
    assert len(table)==25
    assert not result['errors'],result['errors']
    shape = cv2.imread(str(ROOT/'sessions'/session/'classroom.png')).shape
    report = {'registered_students':len(students),'original_student_ids_preserved':True,
              'original_portrait_bytes_preserved':True,'original_splits_preserved':True,
              'original_image_count':read_json(ROOT/'reports/dataset_quality.json')['original_count'],
              'retained_source_count':read_json(ROOT/'reports/dataset_quality.json')['usable_unique_count'],
              'session_id':session,'image_shape':list(shape),'comparison':comparison,
              'attendance_rows':len(csv),'unique_attendance_ids':True,
              'statuses':table.Status.value_counts().to_dict(),
              'inference_ms':result['inference_ms'],
              'limitations':['New synthetic views share a generation context; results are demonstration measurements.',
                             'No independent unknown identities are available for open-set evaluation.']}
    write_json(ROOT/'reports/expansion_25_students.json',report)
    assert comparison['detected_count']==25 and all(r['detected'] for r in comparison['matches']), 'Inspect full-class detection coverage in expansion report'
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(validate_expansion(),indent=2))
