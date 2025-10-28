# create_index.py
import argparse
import json
import os
from pathlib import Path
import hashlib
import lmdb

# ------------------ 공용 상수/유틸 ------------------
SEP = "\x1f"
MAX_KEY_BYTES = 480  # LMDB 키 길이 제한(≈511B) 대비 여유
META_ACTION_KEYS = ("index", "create", "update")  # 2-라인 elasticdump 메타라인 키

def make_key_bytes(key_path: str, value: str) -> bytes:
    raw = f"{key_path}{SEP}{value}".encode("utf-8")
    if len(raw) <= MAX_KEY_BYTES:
        return raw
    h = hashlib.sha256(raw).hexdigest()
    return f"{key_path}{SEP}#h:{h}".encode("utf-8")

def _is_scalar(x):
    return isinstance(x, (str, int, float, bool)) or x is None

def _to_index_str(x):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float, str)):
        return str(x)
    return None  # dict/list 등 비스칼라는 skip (루트 메타는 normalize 사용)

def normalize_id(val):
    """_id를 문자열로 일관되게 정규화"""
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return str(val)
    if isinstance(val, dict):
        for k in ("$oid", "oid", "_id", "id", "value"):
            if k in val and isinstance(val[k], (str, int, float, bool)):
                return str(val[k])
        import json as _json
        return _json.dumps(val, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(val)

# ------------------ 경로 추출 ------------------
def extract_values(obj, path):
    """ dotted path + [] (배열 와일드카드) 지원 """
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

def extract_all_kv(obj, base_path=""):
    """ dict/list 재귀 순회 → (key_path, scalar_value) 목록 """
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

# ------------------ elasticdump payload 루트 ------------------
def is_meta_line(obj):
    return isinstance(obj, dict) and len(obj) == 1 and any(k in obj for k in META_ACTION_KEYS)

def get_payload_root(obj, elasticdump=False, payload_path=None):
    """
    - elasticdump=True: {"index"/"create"/"update":...} 단독 라인은 None (skip)
    - payload_path 지정 시: 해당 dotted 경로
    - elasticdump=True & payload_path 미지정: 기본 '_source'
    - 일반 NDJSON/JSONL: obj 자체
    """
    if elasticdump and is_meta_line(obj):
        return None  # 메타라인 스킵
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

# ------------------ N-gram ------------------
def parse_ngrams(specs):
    """ ["_source.title:2", "title:3"] -> {path: {2,3}} """
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

# ------------------ 파일/매니페스트 ------------------
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

# ------------------ 인덱스 빌드 ------------------
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

    want_source_all = ("_source" in keys)
    want_root_id    = ("_id" in keys)
    want_root_index = ("_index" in keys)

    total_put = 0
    txn = env.begin(db=kv_db, write=True)

    try:
        for file_id, p in enumerate(files):
            with open(p, "rb") as f:
                offset = 0
                line = f.readline()
                pending_id = None
                pending_index = None
                while line:
                    obj = None
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        pass

                    # 메타라인: {"index"/"create"/"update":{...}} → 다음 문서라인에 붙일 값 보관
                    if elasticdump and is_meta_line(obj):
                        meta = obj.get("index") or obj.get("create") or obj.get("update") or {}
                        pending_id = meta.get("_id", pending_id)
                        pending_index = meta.get("_index", pending_index)
                        offset += len(line)
                        line = f.readline()
                        continue

                    # 문서라인 처리
                    base = get_payload_root(obj, elasticdump=elasticdump, payload_path=payload_path) if obj is not None else None
                    if base is not None:
                        v = f"{file_id}{SEP}{offset}".encode("utf-8")

                        # (A) _source 전체 재귀
                        if want_source_all:
                            for path, val in extract_all_kv(base, base_path="_source"):
                                txn.put(make_key_bytes(path, val), v, db=kv_db, dupdata=True)
                                total_put += 1
                                for n in ngram_map.get(path, []):
                                    for ng in ngrams(val, n):
                                        txn.put(make_key_bytes(f"{path}#ng{n}", ng), v, db=kv_db, dupdata=True)
                                        total_put += 1
                                if commit_interval > 0 and (total_put % commit_interval == 0):
                                    txn.commit(); txn = env.begin(db=kv_db, write=True)

                        # (B) 지정 경로("_source","_id","_index" 제외)
                        explicit_paths = [kp for kp in keys if kp not in ("_source", "_id", "_index")]
                        for key_path in explicit_paths:
                            vals = extract_values(base, key_path)
                            for val in vals:
                                txn.put(make_key_bytes(key_path, val), v, db=kv_db, dupdata=True)
                                total_put += 1
                                for n in ngram_map.get(key_path, []):
                                    for ng in ngrams(val, n):
                                        txn.put(make_key_bytes(f"{key_path}#ng{n}", ng), v, db=kv_db, dupdata=True)
                                        total_put += 1
                                if commit_interval > 0 and (total_put % commit_interval == 0):
                                    txn.commit(); txn = env.begin(db=kv_db, write=True)

                        # (C1) 루트 _id 인덱싱 (1라인/2라인 모두 지원, 정규화 포함)
                        if want_root_id:
                            root_id_val = None
                            if elasticdump:
                                root_id_val = pending_id
                                if root_id_val is None and isinstance(obj, dict) and "_id" in obj:
                                    root_id_val = obj.get("_id")
                            else:
                                if isinstance(obj, dict) and "_id" in obj:
                                    root_id_val = obj.get("_id")
                            norm_id = normalize_id(root_id_val)
                            if norm_id is not None:
                                txn.put(make_key_bytes("_id", norm_id), v, db=kv_db, dupdata=True)
                                total_put += 1
                                if commit_interval > 0 and (total_put % commit_interval == 0):
                                    txn.commit(); txn = env.begin(db=kv_db, write=True)

                        # (C2) 루트 _index 인덱싱 (1라인/2라인 모두 지원)
                        if want_root_index:
                            root_index_val = None
                            if elasticdump:
                                root_index_val = pending_index
                                if root_index_val is None and isinstance(obj, dict) and "_index" in obj:
                                    root_index_val = obj.get("_index")
                            else:
                                if isinstance(obj, dict) and "_index" in obj:
                                    root_index_val = obj.get("_index")
                            if root_index_val is not None:
                                txn.put(make_key_bytes("_index", str(root_index_val)), v, db=kv_db, dupdata=True)
                                total_put += 1
                                if commit_interval > 0 and (total_put % commit_interval == 0):
                                    txn.commit(); txn = env.begin(db=kv_db, write=True)

                        # 메타 보관값 사용 후 비우기
                        pending_id = None
                        pending_index = None

                    offset += len(line)
                    line = f.readline()

        txn.commit()
    finally:
        env.sync()
        env.close()

    return total_put, ngram_map

# ------------------ CLI 엔트리 ------------------
def main():
    ap = argparse.ArgumentParser(description="Build LMDB index (exact + N-gram, elasticdump 1/2-line, root _id/_index).")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--files-glob", default="*.json*")
    ap.add_argument("--keys", nargs="+", required=True, help="예: _source _id _index (루트 메타 포함 가능)")
    ap.add_argument("--lmdb", default="kv_index.lmdb")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--map-size-gb", type=int, default=64)
    ap.add_argument("--commit-interval", type=int, default=1_000_000)
    ap.add_argument("--elasticdump", action="store_true")
    ap.add_argument("--payload-path", default="", help="루트가 _source가 아닐 때 경로 지정")
    ap.add_argument("--ngram", action="append", default=[], help="예: --ngram \"_source.title:2\" (반복 가능)")
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
        print(f"[create_index] N-gram enabled for {len(ngram_map)} paths:")
        for k, ns in ngram_map.items():
            print(f"  - {k}: n={sorted(list(ns))}")

if __name__ == "__main__":
    main()