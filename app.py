"""CLI entry point. Run without arguments to open the desktop interface."""
import argparse
import json
import logging
from src.common import ROOT, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description='Synthetic FaceNet512 attendance lab')
    sub = parser.add_subparsers(dest='command')
    p = sub.add_parser('prepare', help='Audit/split/align originals and augment training only')
    p.add_argument('--backend', default='yunet', choices=['opencv', 'retinaface', 'mtcnn', 'yunet'])
    p.add_argument('--seed', type=int, default=42)
    sub.add_parser('build', help='Build gallery, calibrate and analyze identity overlap')
    sub.add_parser('evaluate', help='Compare FaceNet512 and LBPH on held-out test crops')
    p = sub.add_parser('generate', help='Generate a new classroom')
    p.add_argument('--count', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--split', choices=['train','validation','test'], default='test')
    p.add_argument('--rotation', type=float, default=7)
    p = sub.add_parser('recognize', help='Analyze a session image independently')
    p.add_argument('session')
    p.add_argument('--complete', action='store_true')
    p.add_argument('--coverage-adequate', action='store_true', help='Operator confirms all classroom areas are visible and detection is adequate')
    p.add_argument('--compare', action='store_true')
    p.add_argument('--keep-open', action='store_true',help='Keep attendance provisional instead of finalizing a generated still-image session')
    p = sub.add_parser('import-portrait', help='Add a new independent synthetic portrait under an existing label')
    p.add_argument('name'); p.add_argument('path')
    p = sub.add_parser('exclude-set', help='Retire a portrait set from processing without deleting originals')
    p.add_argument('name')
    sub.add_parser('demo', help='Execute seeded 0/1/3/all registered students and rotation scenarios')
    sub.add_parser('gui')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        if args.command == 'prepare':
            from src.dataset_manager import prepare
            result = prepare(args.backend, args.seed)
        elif args.command == 'build':
            from src.face_recognizer import build_gallery
            result = build_gallery()
        elif args.command == 'evaluate':
            from src.evaluation import evaluate
            result = evaluate()
        elif args.command == 'generate':
            from src.classroom_generator import generate
            result = generate(args.count, args.seed, args.split, args.rotation)
        elif args.command == 'recognize':
            from src.session_runner import recognize, compare_truth
            from src.attendance_manager import export_attendance, simulation_info, simulation_completion
            result = recognize(args.session)
            if args.keep_open:
                policy = {'completed':False,'coverage_adequate':False}
            elif args.complete or args.coverage_adequate:
                policy = {'completed':args.complete,'coverage_adequate':args.coverage_adequate}
            else:
                policy = simulation_completion(result,simulation_info(args.session))
            table = export_attendance(args.session,result,**policy)
            result['attendance'] = {
                'counts':table.Status.value_counts().to_dict(),
                'path':str(ROOT/'attendance_reports'/f'{args.session}.csv'),
                'absents_path':str(ROOT/'attendance_reports'/f'{args.session}_absents.csv'),
                'review_path':str(ROOT/'attendance_reports'/f'{args.session}_a_verifier.csv'),
                'not_recognized_path':str(ROOT/'attendance_reports'/f'{args.session}_non_reconnus.csv')}
            if args.compare: result['comparison'] = compare_truth(args.session)
        elif args.command == 'import-portrait':
            from src.dataset_manager import import_portrait
            result = import_portrait(args.name, args.path)
        elif args.command == 'exclude-set':
            from src.dataset_manager import exclude_set
            result = exclude_set(args.name)
        elif args.command == 'demo':
            result = run_demo()
        else:
            from src.ui import launch
            launch(); return 0
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception:
        logging.exception('Operation failed. See README for setup and recovery steps.')
        return 1


def run_demo():
    import pandas as pd
    from src.classroom_generator import generate
    from src.session_runner import recognize, compare_truth, session_path
    from src.attendance_manager import export_attendance
    from src.common import roster
    records = []
    for i, (count, rotation) in enumerate([(0,0),(1,7),(3,7),(len(roster()),7),(3,12),(3,7)]):
        session = generate(count, 100+i, rotation=rotation)
        result = recognize(session)
        table = export_attendance(session, result, completed=True, coverage_adequate=False)
        comparison = compare_truth(session)
        assert len(table) == len(roster()) and table.Student_ID.is_unique
        assert len(pd.read_csv(ROOT / 'attendance_reports' / f'{session}.csv')) == len(roster())
        assert not result['errors'], result['errors']
        if count == 0: assert result['detected_count'] == 0
        records.append({'session': session, 'requested_count': count, 'rotation': rotation,
                        'comparison': comparison, 'statuses': table.Status.value_counts().to_dict()})
    first, second = generate(3, 999), generate(3, 999)
    assert (session_path(first)/'classroom.png').read_bytes() == (session_path(second)/'classroom.png').read_bytes()
    # Recognition cannot depend on ground-truth labels: temporarily remove the file.
    path = session_path(first)/'ground_truth.json'
    backup = path.with_suffix('.test_backup')
    path.rename(backup)
    try: independent = recognize(first)
    finally: backup.rename(path)
    assert independent['detected_count'] >= 0
    write_json(ROOT / 'reports/demo_validation.json', {'scenarios': records, 'seed_reproducibility': True,
        'recognition_without_ground_truth': True,
        'ground_truth_isolation_session': first,
        'ground_truth_isolation_comparison': compare_truth(first),
        'note': 'Counts and identity outcomes are measurements, not assertions of perfect detection or recognition. Coverage was left unconfirmed.'})
    return records


if __name__ == '__main__':
    raise SystemExit(main())
