# search_index.py
import argparse
import json
import mmap
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

def main():
    ap = argparse.ArgumentParser(description="Exact-match lookup from LMDB offset index (nested key paths).")
    ap.add_argument("--lmdb", default="kv_index.lmdb")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--key", required=True, help='Key path (e.g., "user.id", "items[].sku")')
    ap.add_argument("--value", required=True, help="Exact value to match (stringified)")
    ap.add_argument("--limit", type=int, default=0, help="Optional max results")
    args = ap.parse_args()

    res = lookup(args.lmdb, args.key, args.value, args.manifest, args.limit or None)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
    
"""
python search_index.py \
  --lmdb kv_index.lmdb \
  --manifest manifest.json \
  --key "_source.user.id" \
  --value "u01" \
  --limit 100

python search_index.py --lmdb kv_index.lmdb --manifest manifest.json --key "items[].sku" --value "SKU-123"

"""