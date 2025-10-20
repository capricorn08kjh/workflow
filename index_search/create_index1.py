# create_index.py
import argparse
import json
import os
from pathlib import Path
import lmdb

SEP = "\x1f"  # (key_path, value) 구분자

# 파일 상단 근처 공용 상수/함수로 추가하세요
SEP = "\x1f"
MAX_KEY_BYTES = 480  # 511 여유 고려 (내부 오버헤드 감안)

def make_key_bytes(key_path: str, value: str) -> bytes:
    """
    (key_path + SEP + value)를 UTF-8로 인코딩.
    총 길이가 LMDB 한계를 넘으면 value를 해시로 치환.
    """
    raw = f"{key_path}{SEP}{value}".encode("utf-8")
    if len(raw) <= MAX_KEY_BYTES:
        return raw
    import hashlib
    h = hashlib.sha256(f"{key_path}{SEP}{value}".encode("utf-8")).hexdigest()
    # 해시 사용 표식(optional): '#h:' 접두어로 디버깅 용이
    return f"{key_path}{SEP}#h:{h}".encode("utf-8")


# ---------- 유틸: 스칼라 판별/정규화 ----------
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
    """
    dict/list 를 재귀 순회하여 모든 (key_path, scalar_value) 목록 반환
    배열 경로는 base_path에 "[]"를 덧붙여 표기
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

# ---------- 엘라스틱 덤프 payload 루트 결정 ----------
def get_payload_root(obj, elasticdump=False, payload_path=None):
    """
    - elasticdump=True: {"index":...} 단독 라인은 None (skip)
    - payload_path 지정 시: 해당 dotted 경로로 진입
    - elasticdump=True & payload_path 미지정: 기본 '_source'
    - 일반 NDJSON/JSONL: obj 자체 반환
    """
    if elasticdump and "index" in obj and len(obj) == 1:
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

# ---------- 인덱스 빌드 ----------
def discover_files(data_dir, files_glob):
    base = Path(data_dir)
    paths = sorted([str(p) for p in base.glob(files_glob)])
    if not paths:
        raise SystemExit(f"No files matched: {base}/{files_glob}")
    return paths

def write_manifest(manifest_path, files, keys, elasticdump, payload_path):
    meta = {
        "files": [{"id": i, "path": f, "size": os.path.getsize(f)} for i, f in enumerate(files)],
        "keys": keys,
        "elasticdump": bool(elasticdump),
        "payload_path": payload_path or ""
    }
    with open(manifest_path, "w", encoding="utf-8") as w:
        json.dump(meta, w, ensure_ascii=False, indent=2)

def build_index(lmdb_path, files, keys, elasticdump=False, payload_path=None,
                map_size_gb=64, commit_interval=1_000_000):
    env = lmdb.open(
        lmdb_path,
        map_size=map_size_gb * (1024**3),
        subdir=False,
        max_dbs=2,
        lock=True,
    )
    kv_db = env.open_db(b"kv", dupsort=True)  # 동일 key_path+value에 중복값 허용

    total_put = 0
    txn = env.begin(db=kv_db, write=True)  # 첫 트랜잭션

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
                            # 1) `_source` 전체 인덱싱 모드
                            if len(keys) == 1 and keys[0] == "_source":
                                for path, val in extract_all_kv(base, base_path="_source"):
                                    k = f"{path}{SEP}{val}".encode("utf-8")
                                    v = f"{file_id}{SEP}{offset}".encode("utf-8")
                                    txn.put(k, v, db=kv_db, dupdata=True)
                                    total_put += 1
                                    if commit_interval > 0 and (total_put % commit_interval == 0):
                                        txn.commit()
                                        txn = env.begin(db=kv_db, write=True)
                            # 2) 지정 경로만 인덱싱
                            else:
                                for key_path in keys:
                                    vals = extract_values(base, key_path)
                                    for val in vals:
                                        k = f"{key_path}{SEP}{val}".encode("utf-8")
                                        v = f"{file_id}{SEP}{offset}".encode("utf-8")
                                        txn.put(k, v, db=kv_db, dupdata=True)
                                        total_put += 1
                                        if commit_interval > 0 and (total_put % commit_interval == 0):
                                            txn.commit()
                                            txn = env.begin(db=kv_db, write=True)

                    offset += len(line)
                    line = f.readline()

        txn.commit()  # 마지막 커밋
    finally:
        env.sync()
        env.close()

    return total_put

def main():
    ap = argparse.ArgumentParser(description="Build LMDB offset index for NDJSON/JSONL (nested, elasticdump, and `_source` full indexing).")
    ap.add_argument("--data-dir", required=True, help="Directory containing NDJSON/JSONL files")
    ap.add_argument("--files-glob", default="*.json*", help='Glob for files (default: "*.json*")')
    ap.add_argument("--keys", nargs="+", required=True, help='Key paths (e.g., user.id items[].sku) or `_source` for full-recursive indexing')
    ap.add_argument("--lmdb", default="kv_index.lmdb", help="LMDB file path (default: kv_index.lmdb)")
    ap.add_argument("--manifest", default="manifest.json", help="Manifest JSON path (default: manifest.json)")
    ap.add_argument("--map-size-gb", type=int, default=64, help="LMDB map size in GB (default: 64)")
    ap.add_argument("--commit-interval", type=int, default=1_000_000, help="Commit every N puts (default: 1e6)")
    ap.add_argument("--elasticdump", action="store_true", help="Skip meta index lines; default payload=_source")
    ap.add_argument("--payload-path", default="", help="Override payload root dotted path (e.g., _source, hits.hits[])")
    args = ap.parse_args()

    files = discover_files(args.data_dir, args.files_glob)
    print(f"[create_index] Found {len(files)} files")
    write_manifest(args.manifest, files, args.keys, args.elasticdump, args.payload_path or "")
    total = build_index(
        args.lmdb, files, args.keys,
        elasticdump=args.elasticdump,
        payload_path=(args.payload_path or None),
        map_size_gb=args.map_size_gb,
        commit_interval=args.commit_interval
    )
    print(f"[create_index] Done. Indexed entries: {total:,}")

if __name__ == "__main__":
    main()