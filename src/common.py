import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def roster():
    path = ROOT / 'students.json'
    if not path.exists():
        names = sorted(p.name for p in (ROOT / 'dataset').iterdir() if p.is_dir())
        write_json(path, [{'Student_ID': f'E{i+1:03}', 'Name': name} for i, name in enumerate(names)])
    return read_json(path)
