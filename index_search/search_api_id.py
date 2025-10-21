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

# ------------------ 공용 상수/유틸 ------------------
SEP = "\x1f"
MAX_KEY_BYTES = 480  # LMDB key length(~511B) 대비 여유

def make_key_bytes(key_path: str, value: str) -> bytes:
    raw = f"{key_path}{SEP}{value}".encode("utf-8")
    if len(raw) <= MAX_KEY_BYTES:
        return raw
    h = hashlib.sha256(raw).hexdigest()
    return f"{key_path}{SEP}#h:{h}".encode("utf-8")

def ngrams(text: str, n: int) -> List[str]:
    text = str(text)
    L = len(text)
    if L == 0:
        return []
    if L < n:
        return [text]
    return [text[i:i+n] for i in range(L - n + 1)]

# ------------------ 런타임 ------------------
class SearchRuntime:
    def __init__(self, lmdb_path: str, manifest_path: str):
        self.manifest = self._load_manifest(manifest_path)
        self.files_meta = self.manifest["files"]
        self.files, self.mmaps = self._open_mmaps(self.files_meta)
        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, max_dbs=2)
        self.kv_db = self.env.open_db(b"kv")
        # n-gram 설정(예: {"_source.title":[2]})
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
        """정확 일치 포스팅 집합: {(file_id, offset), ...}"""
        k = make_key_bytes(key_path, str(value))
        out: Set[Tuple[int, int]] = set()
        with self.env.begin(db=self.kv_db) as txn, txn.cursor(db=self.kv_db) as cur:
            if cur.set_key(k):
                while True:
                    # ⚠️ 매회 cur.value() 갱신 (중복 모두 읽힘)
                    file_id_s, offset_s = cur.value().decode("utf-8").split(SEP, 1)
                    out.add((int(file_id_s), int(offset_s)))
                    if not cur.next_dup():
                        break
        return out

    def postings_for_contains(self, key_path: str, substr: str, n: Optional[int]) -> Set[Tuple[int, int]]:
        """
        부분매칭: substr을 n-gram으로 쪼개 각 gram의 postings를 AND 교집합.
        인덱스는 key_path#ng{n} 로 생성되어 있어야 함.
        n을 생략하면 manifest의 설정에서 첫 n 사용, 없으면 2.
        """
        if n is None:
            n = (self.ngram_map.get(key_path) or [2])[0]
        grams = ngrams(substr, n)
        if not grams:
            return set()
        inter: Optional[Set[Tuple[int, int]]] = None
        ng_key = f"{key_path}#ng{n}"
        for g in grams:
            s = self.postings_for_exact(ng_key, g)
            inter = s if inter is None else (inter & s)
            if not inter:
                break
        return inter or set()

    def fetch_docs(self, postings: Set[Tuple[int, int]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """오프셋에서 실제 문서 라인 로드 → JSON 디코드"""
        items = sorted(postings)  # 재현성
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

# ------------------ FastAPI ------------------
app = FastAPI(title="KV Index Search API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
RUNTIME: Optional[SearchRuntime] = None

def _union(sets: List[Set[Tuple[int,int]]]) -> Set[Tuple[int,int]]:
    out: Set[Tuple[int,int]] = set()
    for s in sets:
        out |= s
    return out

def _intersect(sets: List[Set[Tuple[int,int]]]) -> Set[Tuple[int,int]]:
    if not sets:
        return set()
    it = iter(sets)
    out = set(next(it))
    for s in it:
        out &= s
        if not out:
            break
    return out

def _eval_group(runtime: SearchRuntime, conds: List[Dict[str, Any]], as_and: bool) -> Set[Tuple[int,int]]:
    """
    cond: 
      - exact  : {"key":"...", "value":"X"}
      - multi  : {"key":"...", "values":["A","B"]}  # 내부 OR
      - contains: {"key":"...", "contains":"서을", "ngram":2}
    """
    buckets: List[Set[Tuple[int,int]]] = []
    for c in conds:
        key = c["key"]
        if "contains" in c:
            substr = str(c["contains"])
            n = c.get("ngram")
            postings = runtime.postings_for_contains(key, substr, n)
            buckets.append(postings)
        elif "values" in c:
            acc: Set[Tuple[int,int]] = set()
            for v in c["values"]:
                acc |= runtime.postings_for_exact(key, str(v))
            buckets.append(acc)
        else:
            v = str(c["value"])
            buckets.append(runtime.postings_for_exact(key, v))
    return _intersect(buckets) if as_and else _union(buckets)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/search")
def search(
    body: Dict[str, Any] = Body(
        ...,
        example={
            "any": [
                {"key": "_source.title", "contains": "특가", "ngram": 2},   # 부분매칭
                {"key": "_id", "value": "abc123"}                          # 정확매칭
            ],
            "all": [
                {"key": "_source.morps[].text", "value": "서울역"}          # 정확매칭
            ],
            "limit": 100,
            "only_source": False,
            "format": "json",   # 또는 "ndjson"
            "with_meta": True
        },
    )
):
    """
    Body:
      any: [ {key, value|values|contains(,ngram)}, ... ]   # OR
      all: [ ... ]                                         # AND
      limit: int
      only_source: bool
      format: "json" | "ndjson"
      with_meta: bool  # json 응답일 때 매칭/반환 개수 메타 포함
    """
    if RUNTIME is None:
        raise HTTPException(500, "Runtime is not initialized")

    any_conds = body.get("any") or []
    all_conds = body.get("all") or []
    limit = body.get("limit")
    only_source = bool(body.get("only_source", False))
    out_format = (body.get("format") or "json").lower()
    with_meta = bool(body.get("with_meta", False))

    if out_format not in ("json", "ndjson"):
        raise HTTPException(400, "format must be 'json' or 'ndjson'")
    if not any_conds and not all_conds:
        raise HTTPException(400, "Provide at least one of 'any' or 'all'.")

    set_any = _eval_group(RUNTIME, any_conds, as_and=False) if any_conds else None
    set_all = _eval_group(RUNTIME, all_conds, as_and=True)  if all_conds else None

    if set_any is not None and set_all is not None:
        final_postings = set_any & set_all
    elif set_any is not None:
        final_postings = set_any
    else:
        final_postings = set_all or set()

    docs = RUNTIME.fetch_docs(final_postings, limit=limit)
    if only_source:
        docs = [d.get("_source") for d in docs if isinstance(d, dict) and "_source" in d]

    if out_format == "ndjson":
        lines = [json.dumps(d, ensure_ascii=False) for d in docs]
        return PlainTextResponse(
            "\n".join(lines) + ("\n" if lines else ""),
            media_type="application/x-ndjson; charset=utf-8"
        )
    else:
        if with_meta:
            meta = {"matched_postings": len(final_postings), "returned_docs": len(docs)}
            return JSONResponse({"meta": meta, "docs": docs})
        return JSONResponse(docs)

# ------------------ Entrypoint ------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Run KV Search API server (exact + contains, root _id).")
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