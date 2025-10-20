# search_index.py
import argparse
import json
import mmap
import os
import sys
import lmdb

SEP = "\x1f"

def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as r:
        return json.load(r)

def open_mmaps(files_meta):
    files = []
    mmaps = []
    for meta in files_meta:
        p = meta["path"]
        f = open(p, "rb")
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        files.append(f)
        mmaps.append(m)
    return files, mmaps

def close_mmaps(files, mmaps):
    for m in mmaps:
        m.close()
    for f in files:
        f.close()

def lookup(lmdb_path, key_path, value, manifest_path="manifest.json", limit=None):
    meta = load_manifest(manifest_path)
    files_meta = meta["files"]
    files, mmaps = open_mmaps(files_meta)

    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, max_dbs=2)
    kv_db = env.open_db(b"kv")

    k = f"{key_path}{SEP}{value}".encode("utf-8")
    results = []
    with env.begin(db=kv_db) as txn, txn.cursor(db=kv_db) as cur:
        if cur.set_key(k):
            vv = cur.value()
            while True:
                file_id_s, offset_s = vv.decode("utf-8").split(SEP, 1)
                file_id, offset = int(file_id_s), int(offset_s)
                mm = mmaps[file_id]
                mm.seek(offset)
                line = mm.readline()
                try:
                    obj = json.loads(line)
                    results.append(obj)
                except json.JSONDecodeError:
                    pass

                if limit and len(results) >= limit:
                    break
                if not cur.next_dup():
                    break

    close_mmaps(files, mmaps)
    env.close()
    return results

def write_output(objs, output, fmt="json", only_source=False, indent=2, ensure_ascii=False):
    # 선택적으로 _source만 추출
    if only_source:
        objs = [o.get("_source") for o in objs if isinstance(o, dict) and "_source" in o]

    if output:
        # 파일로 저장 (UTF-8, \n 고정)
        with open(output, "w", encoding="utf-8", newline="\n") as f:
            if fmt == "ndjson":
                for o in objs:
                    json.dump(o, f, ensure_ascii=ensure_ascii)
                    f.write("\n")
            else:
                json.dump(objs, f, ensure_ascii=ensure_ascii, indent=indent)
        # 요약은 stderr로만 (stdout 오염 방지)
        print(f"Saved {len(objs)} docs to {output} ({fmt})", file=sys.stderr)
    else:
        # stdout 출력
        if fmt == "ndjson":
            # NDJSON을 표준출력
            out = sys.stdout
            # Windows PowerShell 등에서 UTF-16 문제 회피
            try:
                out.reconfigure(encoding="utf-8")  # py3.7+
            except Exception:
                pass
            for o in objs:
                out.write(json.dumps(o, ensure_ascii=ensure_ascii))
                out.write("\n")
        else:
            print(json.dumps(objs, ensure_ascii=ensure_ascii, indent=indent))

def main():
    ap = argparse.ArgumentParser(description="Exact-match lookup and save results (JSON / NDJSON).")
    ap.add_argument("--lmdb", default="kv_index.lmdb")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--key", required=True, help='Key path (e.g., "_source.user.id", "items[].sku")')
    ap.add_argument("--value", required=True, help="Exact value to match (stringified)")
    ap.add_argument("--limit", type=int, default=0, help="Optional max results")
    # 저장/출력 옵션
    ap.add_argument("--output", default="", help="Output file path. If omitted, prints to stdout.")
    ap.add_argument("--format", choices=["json", "ndjson"], default="json", help="Output format (default: json)")
    ap.add_argument("--only-source", action="store_true", help="Emit only the `_source` field if present")
    ap.add_argument("--no-pretty", action="store_true", help="Compact JSON (no indent)")
    ap.add_argument("--ascii", action="store_true", help="Escape non-ASCII (ensure_ascii=True)")
    args = ap.parse_args()

    res = lookup(args.lmdb, args.key, args.value, args.manifest, args.limit or None)
    indent = None if args.no_pretty else 2
    write_output(
        res,
        output=args.output or "",
        fmt=args.format,
        only_source=args.only_source,
        indent=indent,
        ensure_ascii=args.ascii
    )

if __name__ == "__main__":
    main()