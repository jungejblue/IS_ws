"""Verify recorded archive hashes and safely extract the local benchmark data."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def checked_members(archive, destination, expected_root):
    """Validate every entry before creating any output files."""
    entries = []
    seen = set()
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if (name.is_absolute() or '..' in name.parts or '\\' in info.filename
                or ':' in info.filename or not name.parts or name.parts[0] != expected_root
                or stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError('Unsafe or unexpected archive entry: '+info.filename)
        target = destination.joinpath(*name.parts)
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Archive path escapes destination: '+info.filename)
        if target in seen:
            raise ValueError('Duplicate archive entry: '+info.filename)
        seen.add(target)
        entries.append((info, target))
    return entries


def matches_entry(path, info):
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    crc = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            crc = zlib.crc32(chunk, crc)
    return (crc & 0xffffffff) == info.CRC


def unpack(path, destination, expected_root, verify_only=False):
    with zipfile.ZipFile(path) as archive:
        entries = checked_members(archive, destination, expected_root)
        # Detect conflicts before extracting any missing entries.
        for info, target in entries:
            if info.is_dir():
                if target.exists() and not target.is_dir():
                    raise ValueError('Directory conflicts with file: '+str(target))
            elif target.exists() and not matches_entry(target, info):
                raise ValueError('Existing file differs; refusing to overwrite: '+str(target))
        if verify_only:
            missing = sum(not target.exists() for info, target in entries if not info.is_dir())
            if missing:
                raise ValueError(f'{missing} files are not extracted; run without --verify-only')
        else:
            for info, target in entries:
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.unpack-', delete=False) as output:
                            temporary = Path(output.name)
                            with archive.open(info) as source:
                                shutil.copyfileobj(source, output, length=1024*1024)
                        temporary.replace(target)
                    finally:
                        if temporary is not None and temporary.exists():
                            temporary.unlink()
    return sum(not info.is_dir() for info, target in entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', 'hpatches', 'udis'], default='all')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    integrity = json.loads((ROOT/'data/benchmark_sources.json').read_text())
    settings = [('hpatches', 'hpatches-sequences-release.zip', 'hpatches-sequences-release'),
                ('udis', 'udis_testing.zip', 'testing')]
    for key, filename, prefix in settings:
        if args.dataset not in ('all', key):
            continue
        path = ROOT/'data/benchmarks'/filename
        if not path.is_file():
            parser.error(f'{path} missing; download instructions: data/README.md')
        if digest(path) != integrity['downloads'][filename]['sha256']:
            parser.error(f'{filename}: SHA-256 differs from the recorded dataset; do not mix this with old scores')
        count = unpack(path, path.parent, prefix, args.verify_only)
        print(f'{filename}: SHA-256 verified; {count} extracted files verified', flush=True)


if __name__ == '__main__':
    main()
