import hashlib
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import numpy as np
from src.common import ROOT, read_json, roster
from src.face_detector import FaceDetector, iou, letterbox
from src.face_recognizer import decide, class_scores, FaceRecognizer
from src.attendance_manager import (attendance_table, attendance_groups, export_attendance,
                                    simulation_info,simulation_completion)
from src.calibration import distance_boundaries


class PipelineTests(unittest.TestCase):
    def test_originals_and_split_integrity(self):
        audit = read_json(ROOT/'reports/dataset_quality.json')
        for r in audit['images']:
            self.assertEqual(hashlib.sha256((ROOT/r['source']).read_bytes()).hexdigest(),r['sha256'])
        manifest = read_json(ROOT/'reports/dataset_manifest.json')
        groups = {}
        for r in manifest:
            groups.setdefault(r['sha256'],set()).add(r['split'])
            self.assertTrue((ROOT/r['path']).exists())
            if r['augmentation']: self.assertEqual(r['split'],'train')
        self.assertTrue(all(len(s)==1 for s in groups.values()))
        calibration = read_json(ROOT/'reports/calibration.json')
        validation_sources = {r['source'] for r in manifest if r['split']=='validation'}
        for view in calibration.get('validation_views',[]):
            self.assertIn(view['source'], validation_sources)

    def test_rejection(self):
        scores = [('A',.10),('B',.11)]
        self.assertEqual(decide(scores,.3,.03)['name'],'Needs Review')
        self.assertEqual(decide(scores,.05,.03)['name'],'Unknown')
        self.assertEqual(decide([('A',.1),('B',.5)],.3,.03)['name'],'A')
        self.assertEqual(decide([('A',.1),('B',.5)],.3,.03,{'A':['B']})['name'],'Needs Review')

    def test_changed_crop_invalidates_gallery(self):
        manifest = read_json(ROOT/'reports/dataset_manifest.json')
        path = ROOT / next(r['path'] for r in manifest if r['split']=='train')
        original = path.read_bytes()
        try:
            path.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError,'Rebuild the gallery'):
                FaceRecognizer()
        finally:
            path.write_bytes(original)

    def test_validation_separation_leaves_distance_headroom(self):
        trials = [('A',[('A',.10),('B',.45)]), ('A',[('A',.20),('B',.40)]),
                  ('B',[('B',.12),('A',.50)]), ('B',[('B',.18),('A',.42)])]
        thresholds, review, _ = distance_boundaries(trials,['A','B'])
        self.assertGreater(thresholds['A'],.20)
        self.assertLess(thresholds['A'],.42)
        self.assertEqual(decide([('A',.25),('B',.5)],0,.03,(),thresholds,review)['name'],'A')
        self.assertEqual(decide([('A',.8),('B',.9)],0,.03,(),thresholds,review)['name'],'Unknown')

    def test_overlap_review_can_exceed_acceptance_threshold(self):
        scores = [('A',.25),('B',.26)]
        result = decide(scores,.15,.03,{'A':['B']},{'A':.15},{'A':.3})
        self.assertEqual(result['name'],'Needs Review')
        self.assertEqual(result['candidates'],['A','B'])
        self.assertEqual(decide([('A',.8),('B',.9)],.15,.03,{'A':['B']},
                                {'A':.15},{'A':.3})['name'],'Unknown')

    def test_two_references_required(self):
        scores = class_scores(np.array([1.,0.]),np.array([[1.,0.],[0.,1.],[1.,0.]]),np.array(['A','A','B']))
        self.assertEqual(scores,[('A',.5)])

    def test_safe_attendance_and_duplicates(self):
        names = [r['Name'] for r in roster()]
        blank = attendance_table([])
        self.assertTrue((blank.Status=='Needs Review').all())
        self.assertTrue((attendance_table([],True,True).Status=='Absent').all())
        p = [{'name':names[0],'distance':.1,'candidates':[names[0]]}]
        table = attendance_table(p+p,True,True)
        self.assertEqual(table.iloc[0].Status,'Needs Review')
        self.assertTrue(table.Student_ID.is_unique)
        ambiguous = [{'name':'Needs Review','distance':.1,'candidates':names[:2]}]
        table = attendance_table(ambiguous,True,True)
        self.assertTrue((table.iloc[:2].Status=='Needs Review').all())
        unknown = [{'name':'Unknown','distance':.8,'candidates':[]}]
        self.assertTrue((attendance_table(unknown,True,True).Status=='Needs Review').all())

    def test_nms_geometry_and_letterbox(self):
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1)
        self.assertEqual(iou([0,0,10,10],[20,20,10,10]),0)
        self.assertEqual(letterbox(np.zeros((80,160,3),np.uint8)).shape,(160,160,3))

    def test_absence_exports_exclude_present_and_ambiguous_students(self):
        names = [r['Name'] for r in roster()]
        result = {'timestamp':'2026-10-29T10:00:00+01:00','errors':[], 'faces':[
            {'name':names[0],'distance':.1,'candidates':[names[0]]},
            {'name':'Needs Review','distance':.2,'candidates':names[1:3]}]}
        with tempfile.TemporaryDirectory() as directory, patch('src.attendance_manager.ROOT',Path(directory)):
            table = export_attendance('example',result,False,False)
            self.assertEqual(len(attendance_groups(table)['absent']),0)
            base = Path(directory)/'attendance_reports'
            self.assertEqual(len(pd.read_csv(base/'example_absents.csv')),0)
            table = export_attendance('example',result,True,True)
            absent = pd.read_csv(base/'example_absents.csv')
            review = pd.read_csv(base/'example_a_verifier.csv')
            self.assertEqual(set(absent.Name),set(names[3:]))
            self.assertEqual(set(review.Name),set(names[1:3]))
            self.assertEqual(len(pd.read_csv(base/'example_non_reconnus.csv')),len(names)-1)
            export_attendance('example',result,True,True)
            self.assertTrue(pd.read_csv(base/'example.csv').Student_ID.is_unique)
            result['errors'] = ['Embedding inference failed']
            export_attendance('example',result,True,True)
            self.assertEqual(len(pd.read_csv(base/'example_absents.csv')),0)
            self.assertFalse(read_json(base/'example.json')['coverage_adequate'])

    def test_generated_session_finalization_uses_counts_without_identity_labels(self):
        size = len(roster())
        names = [r['Name'] for r in roster()]
        faces = [{'name':name,'distance':.1,'candidates':[name]} for name in names[:20]]
        result = {'detected_count':20,'faces':faces,'errors':[]}
        info = {'registered_count':size,'generated_count':20}
        policy = simulation_completion(result,info)
        self.assertTrue(policy['completed']); self.assertTrue(policy['coverage_adequate'])
        table = attendance_table(faces,**policy)
        self.assertEqual(len(attendance_groups(table)['absent']),size-20)
        missed = {**result,'detected_count':19}
        self.assertFalse(simulation_completion(missed,info)['coverage_adequate'])
        self.assertFalse(simulation_completion({**result,'errors':['failed']},info)['coverage_adequate'])
        self.assertFalse(simulation_completion(result,None)['completed'])
        # Unknown faces do not acquire identities or force absences from the count.
        unknown = {**result,'faces':[{'name':'Unknown','distance':.9,'candidates':[]}]*20}
        self.assertTrue(simulation_completion(unknown,info)['coverage_adequate'])
        table = attendance_table(unknown['faces'],**simulation_completion(unknown,info))
        self.assertTrue((table.Status=='Needs Review').all())

    def test_large_classroom_tiles_preserve_coordinates_and_merge_duplicates(self):
        class StubDetector(FaceDetector):
            def _detect_frame(self, image):
                self.last_size = image.shape[:2]
                points = np.argwhere(image[:,:,0] > 0)
                if not len(points): return []
                y,x = np.round(points.mean(axis=0)).astype(int)
                return [{'box':[int(x)-40,int(y)-40,80,80], 'confidence':.99,
                         'crop':np.zeros((160,160,3),np.uint8),'eyes_available':True}]
        image = np.zeros((1460,1550,3),np.uint8)
        image[550,550] = 255
        detector = StubDetector()
        faces = detector.detect(image)
        self.assertEqual(len(faces),1)
        self.assertEqual(faces[0]['box'],[510,510,80,80])
        self.assertLessEqual(max(detector.last_size),640)


if __name__=='__main__': unittest.main()
