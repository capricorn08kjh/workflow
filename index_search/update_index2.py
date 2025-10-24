# update_index.py
import argparse
import json
import os
from pathlib import Path
import hashlib
import lmdb

# ------------------ 공용 상수/유틸 ------------------
SEP = "\x1f"
MAX_KEY_BYTES = 480  # LMDB 키 길이 제한(≈511B) 대비 여유

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
    return None  # dict/list 등 비스칼라는 skip

# ------------------ 경로 추출 ------------------
def extract_values(obj, path):
    """
    dotted path + [] (배열 와일드카드) 지원.
    예: "user.id", "items[].sku", "events[].attrs.type"
    반환: 스칼라 문자열 리스트
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
    """
    dict/list 재귀 순회 → (key_path, scalar_value) 목록
    배열 경로는 base_path에 "[]" 표기
    """
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
def get_payload_root(obj, elasticdump=False, payload_path=None):
    """
    - elasticdump=True: {"index":...} 단독 라인은 None (skip)
    - payload_path 지정 시: 해당 dotted 경로
    - elasticdump=True & payload_path 미지정: 기본 '_source'
    - 일반 NDJSON/JSONL: obj 자체
    """
    if elasticdump and isinstance(obj, dict) and "index" in obj and len(obj) == 1:
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
def ngrams(text: str, n: int):
    text = str(text)
    L = len(text)
    if L == 0:
        return []
    if L < n:
        return [text]
    return [text[i:i+n] for i in range(L - n + 1)]

# ------------------ 매니페스트 ------------------
def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as r:
        return json.load(r)

def write_manifest(manifest_path, files_meta, keys, elasticdump, payload_path, ngram_map):
    meta = {
        "files": files_meta,
        "keys": keys,
        "elasticdump": bool(elasticdump),
        "payload_path": payload_path or "",
        "ngrams": ngram_map or {}
    }
    with open(manifest_path, "w", encoding="utf-8") as w:
        json.dump(meta, w, ensure_ascii=False, indent=2)

def discover_new_files(data_dir, files_glob, known_paths):
    base = Path(data_dir)
    all_paths = sorted([str(p) for p in base.glob(files_glob)])
    return [p for p in all_paths if p not in known_paths]

# ------------------ 증분 인덱싱 (엘라스틱 _id 통합) ------------------
def append_index(lmdb_path, new_files, keys, elasticdump=False, payload_path=None,
                 start_file_id=0, map_size_gb=64, commit_interval=1_000_000, ngram_map=None):
    ngram_map = ngram_map or {}
    env = lmdb.open(
        lmdb_path,
        map_size=map_size_gb * (1024**3),
        subdir=False,
        max_dbs=2,
        lock=True,
    )
    kv_db = env.open_db(b"kv", dupsort=True)

    want_source_all = ("_source" in keys)
    want_root_id   = ("_id" in keys)

    total_put = 0
    txn = env.begin(db=kv_db, write=True)
    try:
        for add, p in enumerate(new_files):
            file_id = start_file_id + add
            with open(p, "rb") as f:
                offset = 0
                line = f.readline()
                pending_id = None  # 메타라인 _id 저장
                while line:
                    obj = None
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        pass

                    # 메타라인: {"index":{"_id":...}}
                    if elasticdump and isinstance(obj, dict) and "index" in obj and len(obj) == 1:
                        idx_meta = obj.get("index") or {}
                        pending_id = idx_meta.get("_id")
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
                                for n in (ngram_map.get(path) or []):
                                    for ng in ngrams(val, n):
                                        txn.put(make_key_bytes(f"{path}#ng{n}", ng), v, db=kv_db, dupdata=True)
                                        total_put += 1
                                if commit_interval > 0 and (total_put % commit_interval == 0):
                                    txn.commit(); txn = env.begin(db=kv_db, write=True)

                        # (B) 명시적 경로들("_source","_id" 제외)
                        explicit_paths = [kp for kp in keys if kp not in ("_source", "_id")]
                        for key_path in explicit_paths:
                            vals = extract_values(base, key_path)
                            for val in vals:
                                txn.put(make_key_bytes(key_path, val), v, db=kv_db, dupdata=True)
                                total_put += 1
                                for n in (ngram_map.get(key_path) or []):
                                    for ng in ngrams(val, n):
                                        txn.put(make_key_bytes(f"{key_path}#ng{n}", ng), v, db=kv_db, dupdata=True)
                                        total_put += 1
                                if commit_interval > 0 and (total_put % commit_interval == 0):
                                    txn.commit(); txn = env.begin(db=kv_db, write=True)

                        # (C) 루트 _id (elasticdump 메타 → 문서라인 연결)
                        if want_root_id and elasticdump and (pending_id is not None):
                            txn.put(make_key_bytes("_id", str(pending_id)), v, db=kv_db, dupdata=True)
                            total_put += 1
                            if commit_interval > 0 and (total_put % commit_interval == 0):
                                txn.commit(); txn = env.begin(db=kv_db, write=True)

                        pending_id = None  # 한 번 쓰고 비우기

                    offset += len(line)
                    line = f.readline()

        txn.commit()
    finally:
        env.sync()
        env.close()

    return total_put

# ------------------ CLI 엔트리 ------------------
def main():
    ap = argparse.ArgumentParser(description="Incrementally index (exact + N-gram, elasticdump & nested, root _id support).")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--files-glob", default="*.json*")
    ap.add_argument("--lmdb", default="kv_index.lmdb")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--map-size-gb", type=int, default=64)
    ap.add_argument("--commit-interval", type=int, default=1_000_000)
    ap.add_argument("--elasticdump", action="store_true", help="Treat inputs as elasticdump NDJSON (skip meta lines; payload=_source)")
    ap.add_argument("--payload-path", default="", help="Override payload root dotted path")
    args = ap.parse_args()

    meta = load_manifest(args.manifest)
    keys = meta["keys"]
    known_files = [f["path"] for f in meta["files"]]

    # manifest 우선 + CLI override
    elasticdump = args.elasticdump or meta.get("elasticdump", False)
    payload_path = args.payload_path or meta.get("payload_path") or ""
    ngram_map = meta.get("ngrams", {})

    new_files = discover_new_files(args.data_dir, args.files_glob, known_files)
    if not new_files:
        print("[update_index] No new files.")
        return

    start_file_id = len(known_files)
    print(f"[update_index] New files: {len(new_files)} (start_file_id={start_file_id})")
    total = append_index(
        args.lmdb, new_files, keys,
        elasticdump=elasticdump,
        payload_path=(payload_path or None),
        start_file_id=start_file_id,
        map_size_gb=args.map_size_gb,
        commit_interval=args.commit_interval,
        ngram_map=ngram_map
    )
    print(f"[update_index] Appended entries: {total:,}")

    files_meta = meta["files"] + [{"id": start_file_id + i, "path": p, "size": os.path.getsize(p)} for i, p in enumerate(new_files)]
    write_manifest(args.manifest, files_meta, keys, elasticdump, payload_path, ngram_map)
    print("[update_index] Manifest updated.")

if __name__ == "__main__":
    main()