"""Synthetic classroom attendance lab."""
import os
import sys

# Keep the tested ONNX CPU path even if another framework is later installed.
os.environ.setdefault('DEEPFACE_BACKEND_ENGINE', 'onnx')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

# DeepFace logs Unicode during its first model download; Windows pipes can be cp1252.
for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')
