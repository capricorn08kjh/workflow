# create_index.py
import argparse
import json
import os
from pathlib import Path
import lmdb

SEP = "\x1f"  # 키/값 구분자

# ---------- 경로(중첩/배열) 지원 유틸 ----------
def _is_scalar(x):
    return isinstance(x, (str, int, float, bool)) or x is None

def _to_index_str(x):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float, str)):
        return str(x)
    # dict/list 등 비스칼라는 기본 skip
    return None

def extract_values(obj, path):
    """
    dotted path + [] (배열 와일드카드) 지원.
    예: "user.id", "items[].sku", "events[].attrs.type"
    반환: 스칼라값 문자열 리스트 (없으면 빈 리스트)
    """
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
                # 리스트 원소들에 대해 같은 규칙 적용
                for el in node:
                    if isinstance(el, dict) and key in el:
                        v = el[key]
                        if is_arr:
                            if isinstance(v, list):
                                next_level.extend(v)
                        else:
                            next_level.append(v)
        curr = next_level

    # 스칼라만 인덱싱 (object/array는 skip)
    out = []
    stack = list(curr)
    while stack:
        v = stack.pop()
        if _is_scalar(v):
            s = _to_index_str(v)
            if s is not None:
                out.append(s)
    return out

# ---------- 인덱스 빌드 ----------
def discover_files(data_dir, files_glob):
    base = Path(data_dir)
    paths = sorted([str(p) for p in base.glob(files_glob)])
    if not paths:
        raise SystemExit(f"No files matched: {base}/{files_glob}")
    return paths

def write_manifest(manifest_path, files, keys):
    meta = {
        "files": [{"id": i, "path": f, "size": os.path.getsize(f)} for i, f in enumerate(files)],
        "keys": keys
    }
    with open(manifest_path, "w", encoding="utf-8") as w:
        json.dump(meta, w, ensure_ascii=False, indent=2)

def build_index(lmdb_path, files, keys, map_size_gb=64, commit_interval=1_000_000):
    env = lmdb.open(
        lmdb_path,
        map_size=map_size_gb * (1024**3),
        subdir=False,
        max_dbs=2,
        lock=True,
    )
    kv_db = env.open_db(b"kv", dupsort=True)  # 동일 키에 중복값 허용

    total_put = 0
    with env.begin(db=kv_db, write=True) as txn:
        for file_id, p in enumerate(files):
            with open(p, "rb") as f:
                offset = 0
                line = f.readline()
                while line:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # 깨진 라인은 스킵
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
    ap = argparse.ArgumentParser(description="Build LMDB offset index for JSONL files (nested keys supported).")
    ap.add_argument("--data-dir", required=True, help="Directory containing JSONL files")
    ap.add_argument("--files-glob", default="*.jsonl", help="Glob for JSONL files (default: *.jsonl)")
    ap.add_argument("--keys", nargs="+", required=True, help="Key paths to index (dotted; [] for arrays)")
    ap.add_argument("--lmdb", default="kv_index.lmdb", help="LMDB file path (default: kv_index.lmdb)")
    ap.add_argument("--manifest", default="manifest.json", help="Manifest JSON path (default: manifest.json)")
    ap.add_argument("--map-size-gb", type=int, default=64, help="LMDB map size in GB (default: 64)")
    ap.add_argument("--commit-interval", type=int, default=1_000_000, help="Commit every N puts (default: 1e6)")
    args = ap.parse_args()

    files = discover_files(args.data_dir, args.files_glob)
    print(f"[create_index] Found {len(files)} files")
    write_manifest(args.manifest, files, args.keys)
    total = build_index(args.lmdb, files, args.keys, args.map_size_gb, args.commit_interval)
    print(f"[create_index] Done. Indexed entries: {total:,}")

if __name__ == "__main__":
    main()