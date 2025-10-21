# api_search.py
import argparse
import json
import mmap
import hashlib
from typing import List, Optional, Set, Tuple, Dict, Any

import lmdb
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

SEP = "\x1f"
MAX_KEY_BYTES = 480

def make_key_bytes(key_path: str, value: str) -> bytes:
    raw = f"{key_path}{SEP}{value}".encode("utf-8")
    if len(raw) <= MAX_KEY_BYTES:
        return raw
    h = hashlib.sha256(raw).hexdigest()
    return f"{key_path}{SEP}#h:{h}".encode("utf-8")

def ngrams(text: str, n: int):
    text = str(text)
    L = len(text)
    if L == 0:
        return []
    if L < n:
        return [text]
    return [text[i:i+n] for i in range(L - n + 1)]

class SearchRuntime:
    def __init__(self, lmdb_path: str, manifest_path: str):
        self.manifest = self._load_manifest(manifest_path)
        self.files_meta = self.manifest["files"]
        self.files, self.mmaps = self._open_mmaps(self.files_meta)
        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, max_dbs=2)
        self.kv_db = self.env.open_db(b"kv")
        # ngrams 설정
        self.ngram_map: Dict[str, List[int]] = self.manifest.get("ngrams", {})

    @staticmethod
    def _load_manifest(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as r:
            return json.load(r)

    @staticmethod
    def _open_mmaps(files_meta: List[Dict[str, Any]]):
        files = []
        mmaps = []
        for meta in files_meta:
            p = meta["path"]
            f = open(p, "rb")
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            files.append(f)
            mmaps.append(m)
        return files, mmaps

    def close(self):
        for m in self.mmaps:
            m.close()
        for f in self.files:
            f.close()
        self.env.close()

    # ---- postings ----
    def postings_for_exact(self, key_path: str, value: str) -> Set[Tuple[int, int]]:
        k = make_key_bytes(key_path, str(value))
        out: Set[Tuple[int, int]] = set()
        with self.env.begin(db=self.kv_db) as txn, txn.cursor(db=self.kv_db) as cur:
            if cur.set_key(k):
                while True:
                    file_id_s, offset_s = cur.value().decode("utf-8").split(SEP, 1)
                    out.add((int(file_id_s), int(offset_s)))
                    if not cur.next_dup():
                        break
        return out

    def postings_for_contains(self, key_path: str, substr: str, n: int) -> Set[Tuple[int, int]]:
        """
        부분매칭: query를 n-gram으로 쪼개 각 gram의 postings를 AND 교집합.
        인덱스는 key_path#ng{n}로 생성되어 있어야 합니다.
        """
        grams = ngrams(substr, n)
        if not grams:
            return set()
        inter: Optional[Set[Tuple[int, int]]] = None
        ng_key_prefix = f"{key_path}#ng{n}"
        for g in grams:
            s = self.postings_for_exact(ng_key_prefix, g)
            if inter is None:
                inter = s
            else:
                inter &= s
            if not inter:
                break
        return inter or set()

    def fetch_docs(self, postings: Set[Tuple[int, int]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = sorted(postings)
        results: List[Dict[str, Any]] = []
        for (file_id, offset) in items:
            mm = self.mmaps[file_id]
            mm.seek(offset)
            line = mm.readline()
            try:
                obj = json.loads(line)
                results.append(obj)
            except json.JSONDecodeError:
                pass
            if limit and len(results) >= limit:
                break
        return results


app = FastAPI(title="KV Index Search API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
RUNTIME: Optional[SearchRuntime] = None

"""
POST /search 바디 형식(추가됨):
- any / all 조건 각각에서
  - 정확일치: {"key": "...", "value": "foo"} 또는 {"key":"...","values":["a","b"]}
  - 부분일치: {"key": "...", "contains": "foo", "ngram": 2}
    * ngram 생략 시 manifest의 ngrams에서 해당 key에 등록된 첫 번째 n 사용
"""

def union_sets(sets: List[Set[Tuple[int,int]]]) -> Set[Tuple[int,int]]:
    if not sets:
        return set()
    out = set()
    for s in sets:
        out |= s
    return out

def intersect_sets(sets: List[Set[Tuple[int,int]]]) -> Set[Tuple[int,int]]:
    if not sets:
        return set()
    it = iter(sets)
    out = set(next(it))
    for s in it:
        out &= s
        if not out:
            break
    return out

def eval_conditions(runtime: SearchRuntime, conds: List[Dict[str, Any]], as_and: bool) -> Set[Tuple[int,int]]:
    buckets: List[Set[Tuple[int,int]]] = []
    for c in conds:
        key = c["key"]
        # contains(부분매칭)
        if "contains" in c:
            substr = str(c["contains"])
            n = int(c.get("ngram") or (runtime.ngram_map.get(key, [None])[0] or 2))
            postings = runtime.postings_for_contains(key, substr, n)
            buckets.append(postings)
        # values (OR)
        elif "values" in c:
            vals = c["values"]
            s: Set[Tuple[int,int]] = set()
            for v in vals:
                s |= runtime.postings_for_exact(key, str(v))
            buckets.append(s)
        # value (exact)
        else:
            v = str(c["value"])
            buckets.append(runtime.postings_for_exact(key, v))
    return intersect_sets(buckets) if as_and else union_sets(buckets)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/search")
def search(
    body: Dict[str, Any] = Body(
        ...,
        example={
            "any": [
                {"key": "_source.morps[].text", "contains": "서울역", "ngram": 2},
                {"key": "_source.title", "values": ["서울", "용산"]},
            ],
            "all": [
                {"key": "_source.morps[].pos", "values": ["NNP", "NNG"]},
            ],
            "limit": 100,
            "only_source": False,
            "format": "json"
        },
    )
):
    if RUNTIME is None:
        raise HTTPException(500, "Runtime is not initialized")

    any_conds = body.get("any") or []
    all_conds = body.get("all") or []
    limit = body.get("limit")
    only_source = bool(body.get("only_source", False))
    out_format = (body.get("format") or "json").lower()
    if out_format not in ("json","ndjson"):
        raise HTTPException(400, "format must be 'json' or 'ndjson'")
    if not any_conds and not all_conds:
        raise HTTPException(400, "Provide at least one of 'any' or 'all'.")

    sets_any = eval_conditions(RUNTIME, any_conds, as_and=False) if any_conds else None
    sets_all = eval_conditions(RUNTIME, all_conds, as_and=True)  if all_conds else None

    if sets_any is not None and sets_all is not None:
        final_set = sets_any & sets_all
    elif sets_any is not None:
        final_set = sets_any
    else:
        final_set = sets_all or set()

    docs = RUNTIME.fetch_docs(final_set, limit=limit)
    if only_source:
        docs = [d.get("_source") for d in docs if isinstance(d, dict) and "_source" in d]

    if out_format == "ndjson":
        lines = [json.dumps(d, ensure_ascii=False) for d in docs]
        return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""), media_type="application/x-ndjson; charset=utf-8")
    else:
        return JSONResponse(docs)

def parse_args():
    ap = argparse.ArgumentParser(description="Run KV Search API server (exact + contains).")
    ap.add_argument("--lmdb", default="kv_index.lmdb")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    return ap.parse_args()

def create_app(lmdb_path: str, manifest_path: str) -> FastAPI:
    global RUNTIME
    RUNTIME = SearchRuntime(lmdb_path, manifest_path)
    return app

if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    create_app(args.lmdb, args.manifest)
    uvicorn.run(app, host=args.host, port=args.port)
    
"""
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "any": [
      { "key": "_source.morps[].text", "contains": "서울역", "ngram": 2 }
    ],
    "limit": 50
  }' -o contains_results.json
"""