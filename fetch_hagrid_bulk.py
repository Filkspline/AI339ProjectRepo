"""Pull a large, resumable subset of the real HaGRIDv2 512px dataset.

The official HaGRID host (SberCloud) is not resolvable from this environment, so
this uses the Hugging Face mirror `testdummyvt/hagRIDv2_512px`, which holds the
full set as 1503 train + 301 test shards of ~64 MB each (64 GB total; the
`..._10GB` sibling repo the subset name came from holds the same shards).

Downloads sequentially with per-file retries and skips anything already complete,
so it can be interrupted and restarted. Stops once `--gb` have been fetched.

Usage:
    .venv-blazepalm\\Scripts\\python.exe fetch_hagrid_bulk.py --gb 10
    ... --list-only
"""
import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "testdummyvt/hagRIDv2_512px"
UA = {"User-Agent": "Mozilla/5.0"}


def tree(repo=REPO):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    entries = json.load(urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=60))
    out = []
    for e in entries:
        if e.get("type") == "file" and e["path"].endswith(".parquet"):
            out.append((e["path"], e.get("size") or 0))
    # train shards first (that is where the volume is), then test
    out.sort(key=lambda t: (0 if "/train/" in t[0] else 1, t[0]))
    return out


def fetch(repo, path, out_dir, tries=4):
    dest = os.path.join(out_dir, os.path.basename(path))
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                total = int(r.headers.get("Content-Length") or 0)
                tmp = dest + ".part"
                got = 0
                t0 = time.time()
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                if total and got != total:
                    raise IOError(f"short read {got}/{total}")
                os.replace(tmp, dest)
                return got, time.time() - t0
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    retry {attempt}/{tries} for {path}: {exc} "
                  f"(sleeping {wait}s)", flush=True)
            time.sleep(wait)
    return 0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--out", default=os.path.join(HERE, "hagrid512"))
    ap.add_argument("--gb", type=float, default=10.0)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    files = tree(args.repo)
    total = sum(s for _p, s in files)
    print(f"repo {args.repo}: {len(files)} parquet shards, "
          f"{total/1e9:.1f} GB total")
    if args.list_only:
        for p, s in files[:5]:
            print(f"   {p}  {s/1e6:,.1f} MB")
        print("   ...")
        return

    os.makedirs(args.out, exist_ok=True)
    have = sum(os.path.getsize(os.path.join(args.out, os.path.basename(p)))
               for p, _s in files
               if os.path.isfile(os.path.join(args.out, os.path.basename(p))))
    print(f"already on disk in {args.out}: {have/1e9:.2f} GB")
    target = args.gb * 1e9
    fetched = have
    t_start = time.time()
    for path, size in files:
        if fetched >= target:
            break
        dest = os.path.join(args.out, os.path.basename(path))
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            continue
        got, dt = fetch(args.repo, path, args.out)
        if not got:
            print(f"  FAILED: {path}", flush=True)
            continue
        fetched += got
        el = time.time() - t_start
        print(f"  {os.path.basename(path):<38} {got/1e6:7,.1f} MB  "
              f"total {fetched/1e9:5.2f} GB  "
              f"{got/1e6/max(dt,1e-6):5.1f} MB/s  elapsed {el/60:.1f} min",
              flush=True)
    print(f"\ndone: {fetched/1e9:.2f} GB in {args.out} "
          f"({(time.time()-t_start)/60:.1f} min)")


if __name__ == "__main__":
    main()
