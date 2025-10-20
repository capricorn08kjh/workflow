# create_index.py
import argparse
import json
import os
from pathlib import Path
import hashlib
import lmdb

SEP = "\x1f"
MAX_KEY_BYTES = 480  # LMDB 키 길이 제한 여유

def make_key_bytes(key_path: str, value: str) -> bytes:
    raw = f"{key_path}{SEP}{value}".encode("utf-8")
    if len(raw) <= MAX_KEY_BYTES:
        return raw
    h = hashlib.sha256(raw).hexdigest()
    return f"{key_path}{SEP}#h:{h}".encode("utf-8")

# ---------- 스칼라/문자열 유틸 ----------
def _is_scalar(x):
    return isinstance(x, (str, int, float, bool)) or x is None

def _to_index_str(x):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float, str)):
        return str(x)
    return None  # dict/list 등 비스칼라는 skip

# ---------- 지정 경로 추출 (dotted + [] 지원) ----------
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

# ---------- 전체 재귀 추출 (_source 전용) ----------
def extract_all_kv(obj, base_path=""):
    result = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{base_path}.{k}" if base_path else k
            if isinstance(v, (dict, list)):
                result.extend(extract_all_kv(v, path))
            elif _is_scalar(v):
                s = _to_index_str(v)
                if s is not None:
                    result.append((path, s))
    elif isinstance(obj, list):
        for el in obj:
            path = f"{base_path}[]" if base_path else "[]"
            if isinstance(el, (dict, list)):
                result.extend(extract_all_kv(el, path))
            elif _is_scalar(el):
                s = _to_index_str(el)
                if s is not None:
                    result.append((path, s))
    return result

# ---------- 엘라스틱 덤프 payload 루트 ----------
def get_payload_root(obj, elasticdump=False, payload_path=None):
    if elasticdump and "index" in obj and len(obj) == 1:
        return None
    if payload_path:
        base = obj
        for part in payload_path.split("."):
            if isinstance(base, dict) and part in base:
                base = base[part]
            else:
                return None
        return base
    if elasticdump and isinstance(obj, dict) and "_source" in obj:
        return obj["_source"]
    return obj

# ---------- N-gram ----------
def parse_ngrams(specs):
    """
    ["_source.morps[].text:2", "title:3"] -> dict {path: {2,3}}
    """
    table = {}
    for s in specs or []:
        path, _, n_s = s.partition(":")
        n = int(n_s or "2")
        table.setdefault(path, set()).add(n)
    return table

def ngrams(text: str, n: int):
    text = str(text)
    L = len(text)
    if L == 0:
        return []
    if L < n:
        return [text]
    return [text[i:i+n] for i in range(L - n + 1)]

# ---------- 인덱스 빌드 ----------
def discover_files(data_dir, files_glob):
    base = Path(data_dir)
    paths = sorted([str(p) for p in base.glob(files_glob)])
    if not paths:
        raise SystemExit(f"No files matched: {base}/{files_glob}")
    return paths

def write_manifest(manifest_path, files, keys, elasticdump, payload_path, ngram_map):
    meta = {
        "files": [{"id": i, "path": f, "size": os.path.getsize(f)} for i, f in enumerate(files)],
        "keys": keys,
        "elasticdump": bool(elasticdump),
        "payload_path": payload_path or "",
        "ngrams": {k: sorted(list(v)) for k, v in ngram_map.items()},
    }
    with open(manifest_path, "w", encoding="utf-8") as w:
        json.dump(meta, w, ensure_ascii=False, indent=2)

def build_index(lmdb_path, files, keys, elasticdump=False, payload_path=None,
                map_size_gb=64, commit_interval=1_000_000, ngram_specs=None):
    ngram_map = parse_ngrams(ngram_specs)
    env = lmdb.open(
        lmdb_path,
        map_size=map_size_gb * (1024**3),
        subdir=False,
        max_dbs=2,
        lock=True,
    )
    kv_db = env.open_db(b"kv", dupsort=True)

    total_put = 0
    txn = env.begin(db=kv_db, write=True)

    try:
        for file_id, p in enumerate(files):
            with open(p, "rb") as f:
                offset = 0
                line = f.readline()
                while line:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        pass
                    else:
                        base = get_payload_root(obj, elasticdump=elasticdump, payload_path=payload_path)
                        if base is not None:
                            # 1) `_source` 전체 인덱싱
                            if len(keys) == 1 and keys[0] == "_source":
                                for path, val in extract_all_kv(base, base_path="_source"):
                                    # 정확 일치 키
                                    k = make_key_bytes(path, val)
                                    v = f"{file_id}{SEP}{offset}".encode("utf-8")
                                    txn.put(k, v, db=kv_db, dupdata=True)
                                    total_put += 1
                                    # n-gram 키 (해당 path 지정 시)
                                    for n in ngram_map.get(path, []):
                                        for ng in ngrams(val, n):
                                            k2 = make_key_bytes(f"{path}#ng{n}", ng)
                                            txn.put(k2, v, db=kv_db, dupdata=True)
                                            total_put += 1
                                    if commit_interval > 0 and (total_put % commit_interval == 0):
                                        txn.commit()
                                        txn = env.begin(db=kv_db, write=True)
                            # 2) 지정 경로만 인덱싱
                            else:
                                for key_path in keys:
                                    vals = extract_values(base, key_path)
                                    for val in vals:
                                        # 정확 일치 키
                                        k = make_key_bytes(key_path, val)
                                        v = f"{file_id}{SEP}{offset}".encode("utf-8")
                                        txn.put(k, v, db=kv_db, dupdata=True)
                                        total_put += 1
                                        # n-gram 키
                                        for n in ngram_map.get(key_path, []):
                                            for ng in ngrams(val, n):
                                                k2 = make_key_bytes(f"{key_path}#ng{n}", ng)
                                                txn.put(k2, v, db=kv_db, dupdata=True)
                                                total_put += 1
                                        if commit_interval > 0 and (total_put % commit_interval == 0):
                                            txn.commit()
                                            txn = env.begin(db=kv_db, write=True)
                    offset += len(line)
                    line = f.readline()
        txn.commit()
    finally:
        env.sync()
        env.close()

    return total_put, ngram_map

def main():
    ap = argparse.ArgumentParser(description="Build LMDB index (exact + optional N-gram, elasticdump & nested).")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--files-glob", default="*.json*")
    ap.add_argument("--keys", nargs="+", required=True, help="Key paths (or `_source` for full-recursive)")
    ap.add_argument("--lmdb", default="kv_index.lmdb")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--map-size-gb", type=int, default=64)
    ap.add_argument("--commit-interval", type=int, default=1_000_000)
    ap.add_argument("--elasticdump", action="store_true")
    ap.add_argument("--payload-path", default="")
    # N-gram 지정: --ngram "_source.morps[].text:2" --ngram "title:3"
    ap.add_argument("--ngram", action="append", default=[], help="N-gram spec 'path:n' (repeatable)")
    args = ap.parse_args()

    files = discover_files(args.data_dir, args.files_glob)
    print(f"[create_index] Found {len(files)} files")

    total, ngram_map = build_index(
        args.lmdb, files, args.keys,
        elasticdump=args.elasticdump,
        payload_path=(args.payload_path or None),
        map_size_gb=args.map_size_gb,
        commit_interval=args.commit_interval,
        ngram_specs=args.ngram,
    )
    write_manifest(args.manifest, files, args.keys, args.elasticdump, args.payload_path or "", ngram_map)
    print(f"[create_index] Done. Indexed entries: {total:,}")
    if ngram_map:
        print(f"[create_index] N-gram enabled for {len(ngram_map)} paths.")
        for k, ns in ngram_map.items():
            print(f"  - {k}: n={sorted(list(ns))}")

if __name__ == "__main__":
    main()