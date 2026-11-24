"""Read-only dependency/API checks, optionally validating local project photos."""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--reference-inputs', action='store_true')
    args = parser.parse_args()
    if args.reference_inputs and not args.data:
        parser.error('--reference-inputs requires --data')
    errors = []
    print('Python:', sys.version.split()[0])
    if sys.version_info < (3, 10):
        errors.append('Python 3.10 or later is required')
    modules = {}
    for name in ['numpy', 'cv2', 'matplotlib']:
        try:
            mod = importlib.import_module(name)
            modules[name] = mod
            print(f'{name}: {mod.__version__}')
        except ImportError as exc:
            errors.append(f'{name}: {exc}; install requirements.txt in the active environment')
    if 'cv2' in modules:
        cv = modules['cv2']
        for attr in ['SIFT_create', 'USAC_MAGSAC', 'PyRotationWarper',
                     'detail_BestOf2NearestMatcher', 'detail_HomographyBasedEstimator',
                     'detail_BundleAdjusterRay', 'detail_GainCompensator',
                     'detail_MultiBandBlender']:
            if not hasattr(cv, attr):
                errors.append('OpenCV API missing: '+attr)
        for attr in ['computeImageFeatures2', 'waveCorrect']:
            if not hasattr(getattr(cv, 'detail', None), attr):
                errors.append('OpenCV detail API missing: '+attr)
        if args.data:
            for folder, count in [('Q1', 2), ('Q2', 3), ('Q3', 8)]:
                data = ROOT/'data'/folder
                files = sorted(p for p in data.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
                if len(files) != count:
                    errors.append(f'{folder}: expected {count} photos, found {len(files)}; see data/README.md')
                    continue
                if {p.stem for p in files} != {f'{folder}_img_{i}' for i in range(1, count+1)}:
                    errors.append(f'{folder}: expected filenames {folder}_img_1 ... {folder}_img_{count}')
                for path in files:
                    image = cv.imread(str(path))
                    if image is None:
                        errors.append('Cannot decode '+str(path.relative_to(ROOT)))
                print(f'{folder}: {len(files)} photos')
    if args.reference_inputs:
        manifest = json.loads((ROOT/'data/input_manifest.json').read_text())
        for item in manifest['recorded_inputs']:
            path = ROOT/item['path']
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                errors.append('Reference input mismatch: '+item['path'])
    for error in errors:
        print('ERROR:', error, file=sys.stderr)
    if not errors:
        print('Environment checks passed'+(' (including local photos)' if args.data else ' (no dataset required)'))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
