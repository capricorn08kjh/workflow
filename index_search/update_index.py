# update_index.py
import argparse
import json
import os
from pathlib import Path
import lmdb

SEP = "\x1f"

# ------- create_index.py와 동일한 유틸 일부 재사용 -------
def _is_scalar(x):
    return isinstance(x, (str, int, float, bool)) or x is None

def _to_index_str(x):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float, str)):
        return str(x)
    return None

def extract_values(obj, path):
    parts = path.split(".")
    curr = [obj]
    for part in parts:
        is_arr = part.endswith("[]")
        key = part[:-2] if is_arr else part
        next_level = []
        for node in curr:
            if isinstance(node, dict):
                if key not in node:
                    continue
                val = node[key]
                if is_arr:
                    if isinstance(val, list):
                        next_level.extend(val)
                    else:
                        continue
                else:
                    next_level.append(val)
            elif isinstance(node, list):
                for el in node:
                    if isinstance(el, dict) and key in el:
                        v = el[key]
                        if is_arr:
                            if isinstance(v, list):
                                next_level.extend(v)
                        else:
                            next_level.append(v)
        curr = next_level

    out = []
    stack = list(curr)
    while stack:
        v = stack.pop()
        if _is_scalar(v):
            s = _to_index_str(v)
            if s is not None:
                out.append(s)
    return out

# ------- 매니페스트 -------
def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as r:
        return json.load(r)

def write_manifest(manifest_path, files_meta, keys):
    meta = {"files": files_meta, "keys": keys}
    with open(manifest_path, "w", encoding="utf-8") as w:
        json.dump(meta, w, ensure_ascii=False, indent=2)

def discover_new_files(data_dir, files_glob, known_paths):
    base = Path(data_dir)
    all_paths = sorted([str(p) for p in base.glob(files_glob)])
    return [p for p in all_paths if p not in known_paths]

# ------- 증분 인덱싱 -------
def append_index(lmdb_path, new_files, keys, start_file_id, map_size_gb=64, commit_interval=1_000_000):
    env = lmdb.open(
        lmdb_path,
        map_size=map_size_gb * (1024**3),
        subdir=False,
        max_dbs=2,
        lock=True,
    )
    kv_db = env.open_db(b"kv", dupsort=True)

    total_put = 0
    with env.begin(db=kv_db, write=True) as txn:
        for add, p in enumerate(new_files):
            file_id = start_file_id + add
            with open(p, "rb") as f:
                offset = 0
                line = f.readline()
                while line:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        pass
                    else:
                        for key_path in keys:
                            vals = extract_values(obj, key_path)
                            for val in vals:
                                k = f"{key_path}{SEP}{val}".encode("utf-8")
                                v = f"{file_id}{SEP}{offset}".encode("utf-8")
                                txn.put(k, v, db=kv_db, dupdata=True)
                                total_put += 1
                                if total_put % commit_interval == 0:
                                    txn.commit()
                                    txn = env.begin(db=kv_db, write=True)
                    offset += len(line)
                    line = f.readline()
        txn.commit()

    env.sync()
    env.close()
    return total_put

def main():
    ap = argparse.ArgumentParser(description="Incrementally index new JSONL files (nested keys supported).")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--files-glob", default="*.jsonl")
    ap.add_argument("--lmdb", default="kv_index.lmdb")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--map-size-gb", type=int, default=64)
    ap.add_argument("--commit-interval", type=int, default=1_000_000)
    args = ap.parse_args()

    meta = load_manifest(args.manifest)
    keys = meta["keys"]
    known_files = [f["path"] for f in meta["files"]]
    new_files = discover_new_files(args.data_dir, args.files_glob, known_files)

    if not new_files:
        print("[update_index] No new files.")
        return

    start_file_id = len(known_files)
    print(f"[update_index] New files: {len(new_files)} (start_file_id={start_file_id})")

    total = append_index(args.lmdb, new_files, keys, start_file_id, args.map_size_gb, args.commit_interval)
    print(f"[update_index] Appended entries: {total:,}")

    files_meta = meta["files"] + [{"id": start_file_id + i, "path": p, "size": os.path.getsize(p)} for i, p in enumerate(new_files)]
    write_manifest(args.manifest, files_meta, keys)
    print("[update_index] Manifest updated.")

if __name__ == "__main__":
    main()