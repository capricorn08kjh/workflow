# probe_postings.py
import lmdb, hashlib

SEP = "\x1f"
MAX_KEY_BYTES = 480

def make_key_bytes(key_path: str, value: str) -> bytes:
    raw = f"{key_path}{SEP}{value}".encode("utf-8")
    if len(raw) <= MAX_KEY_BYTES:
        return raw
    h = hashlib.sha256(raw).hexdigest()
    return f"{key_path}{SEP}#h:{h}".encode("utf-8")

def count_postings(lmdb_path, key_path, value):
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, max_dbs=2)
    kv = env.open_db(b"kv")
    k = make_key_bytes(key_path, value)
    n = 0
    with env.begin(db=kv) as txn, txn.cursor(db=kv) as cur:
        if cur.set_key(k):
            n += 1
            while cur.next_dup():
                n += 1
    env.close()
    return n

if __name__ == "__main__":
    import sys
    lmdb_path, key_path, value = sys.argv[1], sys.argv[2], sys.argv[3]
    print(count_postings(lmdb_path, key_path, value))
# python probe_postings.py kv_index.lmdb "_source.morps[].text" "서울역"
# python probe_postings.py kv_index.lmdb "_source.title" "서울역"