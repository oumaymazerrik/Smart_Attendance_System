"""FaceNet512 cache: one inference per unique normalized crop/configuration."""
import hashlib
import os
import numpy as np
from .common import ROOT

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


class FaceEmbeddings:
    signature = 'Facenet512-base-aligned-bgr-letterbox160-v1'

    def __init__(self):
        self.directory = ROOT / 'embeddings/cache'
        self.directory.mkdir(parents=True, exist_ok=True)

    def embed(self, crop):
        from deepface import DeepFace
        key = hashlib.sha256(self.signature.encode() + crop.tobytes()).hexdigest()
        path = self.directory / f'{key}.npy'
        if path.exists():
            embedding = np.load(path, allow_pickle=False)
        else:
            result = DeepFace.represent(crop, model_name='Facenet512', detector_backend='skip',
                                       normalization='base', enforce_detection=False)
            embedding = np.asarray(result[0]['embedding'], dtype=np.float32)
            if embedding.shape != (512,) or not np.isfinite(embedding).all():
                raise ValueError('FaceNet512 returned an invalid embedding')
            embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
            np.save(path, embedding, allow_pickle=False)
        if embedding.shape != (512,) or not np.isfinite(embedding).all():
            raise ValueError(f'Invalid embedding cache: {path}')
        return embedding
