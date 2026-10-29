"""Exercise absent-list exports on complete synthetic sessions, without using truth as input."""
from .common import ROOT, roster, read_json, write_json
from .classroom_generator import generate
from .session_runner import recognize
from .attendance_manager import export_attendance, attendance_groups
import pandas as pd


def validate_absences():
    registered = {student['Name'] for student in roster()}
    reports = []
    for count in (0,20,25):
        session = generate(count=count,seed=42)
        result = recognize(session)
        table = export_attendance(session,result,completed=True,coverage_adequate=True)
        groups = attendance_groups(table)
        # Ground truth is read ONLY after model predictions and attendance exports.
        truth = read_json(ROOT/'sessions'/session/'ground_truth.json')
        actual_present = {student['name'] for student in truth['students']}
        expected_absent = registered-actual_present
        predicted_absent = set(groups['absent'].Name)
        record = {'session_id':session,'requested_count':count,
            'detected_faces':result['detected_count'],
            'counts':{name:len(rows) for name,rows in groups.items()},
            'confirmed_absent_names':sorted(predicted_absent),
            'review_names':groups['review'].Name.tolist(),
            'expected_absent_names_for_evaluation_only':sorted(expected_absent),
            'incorrect_absent_names':sorted(predicted_absent-expected_absent)}
        reports.append(record)
        assert not record['incorrect_absent_names'], 'A visible student was incorrectly marked absent'
        csv = pd.read_csv(ROOT/'attendance_reports'/f'{session}_absents.csv')
        assert set(csv.Name)==predicted_absent and csv.Student_ID.is_unique
        assert len(table)==25 and table.Student_ID.is_unique
        if count == 0: assert len(csv)==25
        if count == 25: assert len(csv)==0
    write_json(ROOT/'reports/absence_validation.json',{'scenarios':reports,
        'note':'Simulation coverage was explicitly confirmed for these tests. Recognition and absence decisions never read ground-truth labels.'})
    return reports


if __name__ == '__main__':
    import json
    print(json.dumps(validate_absences(),indent=2))
