"""Download files from a Hugging Face dataset mirror.

The official HaGRID host (SberCloud) is not resolvable from this environment, so
this pulls from HF mirrors instead.

Usage:
    python fetch_hagrid.py --repo <user/dataset> --file <path/in/repo> [--out DIR]
    python fetch_hagrid.py --repo <user/dataset> --list
"""
import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0"}


def list_repo(repo):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main"
    req = urllib.request.Request(url, headers=UA)
    entries = json.load(urllib.request.urlopen(req, timeout=30))
    for e in entries:
        size = e.get("size")
        print(f"  {e.get('type','?'):<9} {e.get('path'):<50} "
              f"{'' if size is None else f'{size/1e6:,.1f} MB'}")
    return entries


def download(repo, path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, os.path.basename(path))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"already present: {dest} ({os.path.getsize(dest)/1e6:,.1f} MB)")
        return dest
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    print(f"downloading {url}\n         -> {dest}")
    req = urllib.request.Request(url, headers=UA)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = last = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got - last >= 100 << 20:
                last = got
                el = max(time.time() - t0, 1e-6)
                pct = f"{100*got/total:.0f}%" if total else "?"
                print(f"  {got/1e6:,.0f} MB / {total/1e6:,.0f} MB ({pct}) "
                      f"{got/1e6/el:.1f} MB/s", flush=True)
    print(f"done: {dest} ({os.path.getsize(dest)/1e6:,.1f} MB in "
          f"{time.time()-t0:.0f}s)")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--file")
    ap.add_argument("--out", default=os.path.join(HERE, "hagrid"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.file:
        list_repo(args.repo)
        return
    download(args.repo, args.file, args.out)


if __name__ == "__main__":
    main()
