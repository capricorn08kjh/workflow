# add_id_index.py
import argparse, json, lmdb, os
from pathlib import Path
import hashlib

SEP = "\x1f"
MAX_KEY_BYTES = 480

def make_key_bytes(key_path: str, value: str) -> bytes:
    raw = f"{key_path}{SEP}{value}".encode("utf-8")
    if len(raw) <= MAX_KEY_BYTES:
        return raw
    h = hashlib.sha256(raw).hexdigest()
    return f"{key_path}{SEP}#h:{h}".encode("utf-8")

def load_manifest(p):
    with open(p, "r", encoding="utf-8") as r:
        return json.load(r)

def open_env(lmdb_path, map_size_gb=64):
    return lmdb.open(
        lmdb_path,
        map_size=map_size_gb * (1024**3),
        subdir=False,
        max_dbs=2,
        lock=True,
    )

def add_id_postings(lmdb_path, manifest_path, commit_interval=1_000_000):
    meta = load_manifest(manifest_path)
    files_meta = meta["files"]
    total_put = 0

    env = open_env(lmdb_path)
    kv_db = env.open_db(b"kv", dupsort=True)
    txn = env.begin(db=kv_db, write=True)

    try:
        for fm in files_meta:
            path = fm["path"]
            if not os.path.exists(path):
                print(f"[skip] not found: {path}")
                continue

            with open(path, "rb") as f:
                offset = 0
                last_id = None   # 메타라인의 _id
                while True:
                    line = f.readline()
                    if not line:
                        break
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        offset += len(line)
                        continue

                    # 메타라인: {"index":{"_id":"..."}}
                    if isinstance(obj, dict) and "index" in obj and len(obj) == 1:
                        idx_meta = obj.get("index") or {}
                        last_id = idx_meta.get("_id")
                        offset += len(line)
                        continue

                    # 문서라인: 직전 메타의 _id를 현재 문서 오프셋과 매칭
                    if last_id is not None:
                        k = make_key_bytes("_id", str(last_id))
                        v = f"{fm['id']}{SEP}{offset}".encode("utf-8")
                        txn.put(k, v, db=kv_db, dupdata=True)
                        total_put += 1
                        last_id = None

                        if total_put % commit_interval == 0:
                            txn.commit()
                            txn = env.begin(db=kv_db, write=True)

                    offset += len(line)

        txn.commit()
    finally:
        env.sync()
        env.close()

    # manifest에 keys 목록 보강(표시 목적)
    keys = set(meta.get("keys", []))
    if "_id" not in keys:
        keys.add("_id")
        meta["keys"] = sorted(keys)
        with open(manifest_path, "w", encoding="utf-8") as w:
            json.dump(meta, w, ensure_ascii=False, indent=2)

    print(f"[add_id_index] Added postings: {total_put:,}")

def main():
    ap = argparse.ArgumentParser(description="Append `_id` postings from elasticdump NDJSON meta lines.")
    ap.add_argument("--lmdb", default="kv_index.lmdb")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--commit-interval", type=int, default=1_000_000)
    args = ap.parse_args()
    add_id_postings(args.lmdb, args.manifest, args.commit_interval)

if __name__ == "__main__":
    main()