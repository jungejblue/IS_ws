"""Check the submission contract, local documentation links and Git artifact policy."""
import argparse
import ast
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def git_paths(*args):
    return [p for p in subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').split('\0') if p]


def check_contract(path):
    errors = []
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name not in ('cv2', 'numpy'):
                    errors.append(f'{path.name}: unsupported submission import {item.name}')
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ('cv2', 'numpy'):
                errors.append(f'{path.name}: local/unsupported submission import {node.module}')
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ('Stitcher', 'Stitcher_create', 'imwrite', 'imshow', 'waitKey'):
                errors.append(f'{path.name}: forbidden submission call {node.func.attr}')
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ('q1', 'q2', 'q3', 'e1', 'e2'):
        fn = functions.get(name)
        if fn is None or [a.arg for a in fn.args.args] != ['img_paths'] or fn.args.vararg or fn.args.kwarg:
            errors.append(f'{path.name}: required signature {name}(img_paths) missing')
    return errors


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k, v in attrs if k in ('href', 'src') and v)


def local_links(path):
    content = path.read_text(encoding='utf-8')
    if path.suffix == '.md':
        # Code examples are not rendered links.
        content = re.sub(r'```.*?```', '', content, flags=re.S)
        return re.findall(r'!?\[[^\]]*\]\(([^\n)]+)\)', content)
    parser = Links()
    parser.feed(content)
    return parser.links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-tracked-local', action='store_true', help='Preview artifact policy before removing files from the Git index')
    args = parser.parse_args()
    candidates = set(git_paths('ls-files', '-c', '-o', '--exclude-standard', '-z'))
    ignored = set(git_paths('ls-files', '-c', '-i', '--exclude-standard', '-z'))
    prospective = {p for p in candidates-ignored if (ROOT/p).is_file()}
    errors = check_contract(ROOT/'stitching.py')
    if ignored and not args.allow_tracked_local:
        errors.append(f'{len(ignored)} tracked local artifacts match .gitignore; untrack without deleting local data (docs/running.md)')
    checked_links = 0
    for relative in sorted(prospective):
        path = ROOT/relative
        if path.stat().st_size > 10*1024*1024:
            errors.append(f'File exceeds 10 MiB repository limit: {relative}')
        if path.suffix not in ('.md', '.html'):
            continue
        for link in local_links(path):
            target = urlsplit(link.strip('<>'))
            if target.scheme or target.netloc or not target.path:
                continue
            dest = (path.parent/unquote(target.path)).resolve()
            try:
                name = dest.relative_to(ROOT).as_posix()
            except ValueError:
                errors.append(f'{relative}: link leaves repository: {link}')
                continue
            if not dest.exists():
                errors.append(f'{relative}: missing link: {link}')
            elif dest.is_file() and name not in prospective:
                errors.append(f'{relative}: link only exists locally, not in shared repository: {link}')
            checked_links += 1
    for error in errors[:40]:
        print('ERROR:', error, file=sys.stderr)
    if len(errors)>40:
        print(f'... and {len(errors)-40} more errors', file=sys.stderr)
    print(f'Checked submission contract, {len(prospective)} shared files, {checked_links} local links; {len(ignored)} tracked local artifacts')
    if not errors:
        print('Repository checks passed'+(' (migration preview)' if args.allow_tracked_local else ''))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
