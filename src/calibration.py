"""Validation-only operating-point calibration, including classroom-scale variation."""
import cv2
import numpy as np
from .common import ROOT, roster


def validation_views(record, detector):
    """All views stay in validation; none becomes a gallery or independent test sample."""
    yield 'original_crop', cv2.imread(str(ROOT / record['path']))
    source = cv2.imread(str(ROOT / record['source']))
    if source is None:
        raise ValueError(f"Unreadable validation source: {record['source']}")
    for size in (190, 230):
        ratio = size / max(source.shape[:2])
        portrait = cv2.resize(source, (round(source.shape[1]*ratio), round(source.shape[0]*ratio)))
        h, w = portrait.shape[:2]
        for angle in (-7, 0, 7):
            rotated = cv2.warpAffine(portrait, cv2.getRotationMatrix2D((w/2,h/2),angle,1),
                                    (w,h), borderMode=cv2.BORDER_REFLECT_101)
            for brightness in (.9, 1.1):
                # A classroom-sized canvas reproduces detector downscaling. It contains
                # exactly this validation source, with no ground-truth text or test images.
                contexts = [('classroom',650,930)]
                if len(roster()) >= 10:
                    contexts.append(('native_tile',640,640))
                for context,canvas_height,canvas_width in contexts:
                    canvas = np.full((canvas_height,canvas_width,3),220,np.uint8)
                    canvas[140:140+h,100:100+w] = cv2.convertScaleAbs(rotated,alpha=brightness)
                    faces = detector.detect(canvas)
                    view = f'{context}_size{size}_angle{angle}_brightness{brightness}'
                    yield view, faces[0]['crop'] if len(faces) == 1 else None


def distance_boundaries(trials, names):
    """Use genuine/impostor separation, rather than the smallest genuine distance.

    Each other registered student's validation crop is an impostor for this class.
    This is not a substitute for independent unknown identities.
    """
    thresholds, review, statistics = {}, {}, {}
    for name in names:
        genuine = [dict(scores)[name] for truth, scores in trials if truth == name]
        impostor = [dict(scores)[name] for truth, scores in trials if truth != name]
        if not genuine or not impostor:
            raise ValueError(f'Insufficient genuine/impostor validation scores for {name}')
        upper, lower = max(genuine), min(impostor)
        threshold = (upper+lower)/2 if upper < lower else float(np.nextafter(lower, 0))
        thresholds[name] = float(threshold)
        review[name] = float(upper)
        statistics[name] = {'genuine_max': float(upper), 'impostor_min': float(lower),
                            'separable': upper < lower, 'threshold': float(threshold),
                            'genuine_views': len(genuine), 'impostor_views': len(impostor)}
    return thresholds, review, statistics
