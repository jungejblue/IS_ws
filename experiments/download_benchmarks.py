"""Download public HPatches/UDIS-D data and save source metadata (stdlib only)."""
import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'benchmarks'
SOURCES = {
    'hpatches': 'https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip?download=true',
    'udis_folder': 'https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY',
    'udis_evaluator': 'https://raw.githubusercontent.com/nie-lang/UDIS2/main/Warp/Codes/test.py',
    'udis_dataset': 'https://raw.githubusercontent.com/nie-lang/UDIS2/main/Warp/Codes/dataset.py',
}


def fetch(url, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    with opener.open(request, timeout=90) as response:
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type and '/download?' in url:
            page = response.read().decode()
            action = re.search(r'<form[^>]*action="([^"]+)"', page)
            if not action:
                raise RuntimeError('Google Drive download unavailable: '+page[:300])
            fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', page))
            return fetch(html.unescape(action[1])+'?'+urllib.parse.urlencode(fields), output)
        temp = output.with_suffix(output.suffix+'.part')
        size = 0
        digest = hashlib.sha256()
        last = time.monotonic()
        with temp.open('wb') as stream:
            while True:
                data = response.read(1024*1024)
                if not data:
                    break
                stream.write(data); digest.update(data); size += len(data)
                if time.monotonic()-last > 15:
                    print(output.name,round(size/1e6,1),'MB',flush=True); last=time.monotonic()
        temp.replace(output)
        record = dict(source_url=url,bytes=size,sha256=digest.hexdigest(),content_type=content_type)
        output.with_suffix(output.suffix+'.source.json').write_text(json.dumps(record,indent=2))
        print(output.name,record,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['metadata','hpatches','drive-folder','drive-file'])
    parser.add_argument('--id')
    parser.add_argument('--name',default='udis_testing.zip')
    args=parser.parse_args()
    if args.action=='metadata':
        for key in ['udis_folder','udis_evaluator','udis_dataset']:
            fetch(SOURCES[key],ROOT/'sources'/(key+('.html' if key=='udis_folder' else '.txt')))
    elif args.action=='hpatches':
        fetch(SOURCES['hpatches'],ROOT/'hpatches-sequences-release.zip')
    else:
        if not args.id or not re.fullmatch(r'[A-Za-z0-9_-]+',args.id):
            parser.error('A public Drive ID is required')
        if Path(args.name).name!=args.name:
            parser.error('name must be a basename')
        if args.action=='drive-folder':
            fetch('https://drive.google.com/drive/folders/'+args.id,ROOT/'sources'/(args.id+'.html'))
        else:
            fetch('https://drive.usercontent.google.com/download?'+urllib.parse.urlencode(
                dict(id=args.id,export='download',confirm='t')),ROOT/args.name)


if __name__=='__main__':
    main()
