"""DeepFace detection, eye alignment, aspect-preserving crops and NMS."""
import cv2
import numpy as np


def letterbox(image, size=160):
    h, w = image.shape[:2]
    if not h or not w:
        raise ValueError('Empty face crop')
    scale = size / max(h, w)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    dh, dw = size - resized.shape[0], size - resized.shape[1]
    return cv2.copyMakeBorder(resized, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2, cv2.BORDER_REPLICATE)


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    r, d = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    area = max(0, r-x) * max(0, d-y)
    return area / max(1, a[2]*a[3] + b[2]*b[3] - area)


class FaceDetector:
    def __init__(self, backend='yunet'):
        self.backend = backend

    def detect(self, image):
        if image is None:
            raise ValueError('Cannot detect faces in an unreadable image')
        # Preserve small faces in large classrooms instead of shrinking all 25
        # students to fit YuNet's 640-pixel input. Tiles overlap; NMS merges them.
        if self.backend == 'yunet' and max(image.shape[:2]) > 1000:
            detections = []
            height, width = image.shape[:2]
            tile, step = 640, 400  # 240px overlap covers a whole generated portrait.
            xs = sorted(set(list(range(0, max(1,width-tile+1),step))+[max(0,width-tile)]))
            ys = sorted(set(list(range(0, max(1,height-tile+1),step))+[max(0,height-tile)]))
            for y in ys:
                for x in xs:
                    region = image[y:min(y+tile,height),x:min(x+tile,width)]
                    for face in self._detect_frame(region):
                        fx,fy,fw,fh = face['box']
                        # Internal tile edges can create partial duplicate faces.
                        # Their complete view is covered by a neighboring tile.
                        if ((x>0 and fx<5) or (y>0 and fy<5)
                            or (x+region.shape[1]<width and fx+fw>region.shape[1]-5)
                            or (y+region.shape[0]<height and fy+fh>region.shape[0]-5)):
                            continue
                        face['box'][0] += x; face['box'][1] += y
                        detections.append(face)
            kept = []
            for face in sorted(detections,key=lambda f:f['confidence'],reverse=True):
                if not any(iou(face['box'],old['box']) > .30 for old in kept):
                    kept.append(face)
            return sorted(kept,key=lambda f:(f['box'][1],f['box'][0]))
        return self._detect_frame(image)

    def _detect_frame(self, image):
        from deepface import DeepFace
        if image is None:
            raise ValueError('Cannot detect faces in an unreadable image')
        try:
            faces = DeepFace.extract_faces(image, detector_backend=self.backend,
                enforce_detection=True, align=False, color_face='bgr', normalize_face=False)
        except ValueError as exc:
            if 'Face could not be detected' in str(exc):
                return []
            raise
        kept = []
        for face in sorted(faces, key=lambda f: f['confidence'], reverse=True):
            area = face['facial_area']
            x, y = max(0, area['x']), max(0, area['y'])
            box = [x, y, min(area['w'], image.shape[1]-x), min(area['h'], image.shape[0]-y)]
            if min(box[2:]) < 30 or any(iou(box, f['box']) > .35 for f in kept):
                continue
            crop = np.asarray(face['face'], dtype=np.uint8)
            eyes_available = area.get('left_eye') is not None and area.get('right_eye') is not None
            if eyes_available:
                # Detect at native resolution first: DeepFace's align=True pads the whole
                # image, which makes YuNet downscale small classroom faces too aggressively.
                left, right = sorted([area['left_eye'], area['right_eye']], key=lambda eye: eye[0])
                angle = float(np.degrees(np.arctan2(right[1]-left[1], right[0]-left[0])))
                h, w = crop.shape[:2]
                crop = cv2.warpAffine(crop, cv2.getRotationMatrix2D((w/2,h/2),angle,1), (w,h), borderMode=cv2.BORDER_REFLECT_101)
            kept.append({'box': box, 'crop': letterbox(crop),
                         'confidence': float(face['confidence']),
                         'eyes_available': eyes_available})
        return sorted(kept, key=lambda f: f['box'][0])
