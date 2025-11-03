# api_multi.py
import argparse
import json
import mmap
import hashlib
from typing import List, Optional, Set, Tuple, Dict, Any

import lmdb
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ===================== 공용 유틸 =====================
SEP = "\x1f"
MAX_KEY_BYTES = 480

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

def s_union(sets: List[Set[Tuple[int,int]]]) -> Set[Tuple[int,int]]:
    out: Set[Tuple[int,int]] = set()
    for s in sets:
        out |= s
    return out

def s_intersect(sets: List[Set[Tuple[int,int]]]) -> Set[Tuple[int,int]]:
    if not sets:
        return set()
    it = iter(sets)
    out = set(next(it))
    for s in it:
        out &= s
        if not out:
            break
    return out

# ===================== 런타임(인덱스 1개) =====================
class SearchRuntime:
    def __init__(self, lmdb_path: str, manifest_path: str):
        self.manifest = self._load_manifest(manifest_path)
        self.files_meta = self.manifest["files"]
        self.files, self.mmaps = self._open_mmaps(self.files_meta)
        # 다중 동시 읽기 + 다른 프로세스에서 라이터 가능
        self.env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=True,
            max_dbs=2,
            max_readers=1024
        )
        self.kv_db = self.env.open_db(b"kv")
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

    def postings_for_contains(self, key_path: str, substr: str, n: Optional[int]) -> Set[Tuple[int, int]]:
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

# ===================== 쿼리 평가 로직(재사용) =====================
def eval_single_condition(runtime: SearchRuntime, cond: Dict[str, Any]) -> Tuple[Optional[Set[Tuple[int,int]]], Optional[Set[Tuple[int,int]]]]:
    """
    반환: (include_set, exclude_set)
      - contains/values+logic/exact 모두 지원
    """
    key = cond["key"]
    logic = cond.get("logic", "or").lower()  # or|and|not

    def postings_exact_vals(vals: List[str], combine: str) -> Set[Tuple[int,int]]:
        bags = [runtime.postings_for_exact(key, str(v)) for v in vals]
        return s_intersect(bags) if combine == "and" else s_union(bags)

    def postings_contains_vals(vals: List[str], n: Optional[int], combine: str) -> Set[Tuple[int,int]]:
        bags = [runtime.postings_for_contains(key, str(v), n) for v in vals]
        return s_intersect(bags) if combine == "and" else s_union(bags)

    # contains 단일
    if "contains" in cond:
        result = runtime.postings_for_contains(key, str(cond["contains"]), cond.get("ngram"))
        return (None, result) if logic == "not" else (result, None)

    # values 리스트
    if "values" in cond:
        vals = cond["values"]
        if not isinstance(vals, list) or not vals:
            return set(), None
        match = cond.get("match", "exact").lower()
        if match == "contains":
            result = postings_contains_vals(vals, cond.get("ngram"), "and" if logic == "and" else "or")
        else:
            result = postings_exact_vals(vals, "and" if logic == "and" else "or")
        return (None, result) if logic == "not" else (result, None)

    # 단일 value (정확)
    if "value" in cond:
        result = runtime.postings_for_exact(key, str(cond["value"]))
        return (None, result) if logic == "not" else (result, None)

    return set(), None

def eval_group(runtime: SearchRuntime, conds: List[Dict[str, Any]], as_and: bool) -> Tuple[Set[Tuple[int,int]], Set[Tuple[int,int]]]:
    include_parts: List[Set[Tuple[int,int]]] = []
    exclude_parts: List[Set[Tuple[int,int]]] = []
    for c in conds:
        inc, exc = eval_single_condition(runtime, c)
        if inc is not None:
            include_parts.append(inc)
        if exc is not None:
            exclude_parts.append(exc)
    include_final = s_intersect(include_parts) if as_and else s_union(include_parts)
    exclude_union = s_union(exclude_parts) if exclude_parts else set()
    return include_final, exclude_union

def combine_any_all(runtime: SearchRuntime, body: Dict[str, Any]) -> Tuple[Set[Tuple[int,int]], int]:
    any_conds = body.get("any") or []
    all_conds = body.get("all") or []
    limit = body.get("limit")

    any_inc, any_exc = eval_group(runtime, any_conds, as_and=False) if any_conds else (set(), set())
    all_inc, all_exc = eval_group(runtime, all_conds, as_and=True)  if all_conds else (set(), set())

    # 포함 결합
    if any_conds and all_conds:
        final_postings = any_inc & all_inc
    elif any_conds:
        final_postings = any_inc
    else:
        final_postings = all_inc

    # 제외 차감
    exclusions = s_union([any_exc, all_exc])
    if exclusions:
        final_postings -= exclusions

    return final_postings, (limit or None)

# ===================== 앱 초기화/수명 관리 =====================
RUNTIMES: Dict[str, SearchRuntime] = {}  # {"nw": runtime, "rp": runtime}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 초기화 (args는 main()에서 주입)
    yield
    # 종료 시 정리
    for rt in RUNTIMES.values():
        rt.close()

app = FastAPI(title="KV Index Search API (multi)", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    ok = {name: True for name in RUNTIMES.keys()}
    return {"ok": ok, "runtimes": list(RUNTIMES.keys())}

# --------- 개별 인덱스 검색: NW ----------
@app.post("/search_nw")
def search_nw(body: Dict[str, Any] = Body(...)):
    if "nw" not in RUNTIMES:
        raise HTTPException(500, "NW runtime not initialized")
    runtime = RUNTIMES["nw"]
    final_postings, limit = combine_any_all(runtime, body)
    only_source = bool(body.get("only_source", False))
    out_format = (body.get("format") or "json").lower()
    with_meta = bool(body.get("with_meta", False))

    docs = runtime.fetch_docs(final_postings, limit=limit)
    if only_source:
        docs = [d.get("_source") for d in docs if isinstance(d, dict) and "_source" in d]

    if out_format == "ndjson":
        lines = [json.dumps(d, ensure_ascii=False) for d in docs]
        return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""),
                                 media_type="application/x-ndjson; charset=utf-8")
    else:
        return JSONResponse({"meta": {"matched_postings": len(final_postings), "returned_docs": len(docs)}, "docs": docs} if with_meta else docs)

# --------- 개별 인덱스 검색: RP ----------
@app.post("/search_rp")
def search_rp(body: Dict[str, Any] = Body(...)):
    if "rp" not in RUNTIMES:
        raise HTTPException(500, "RP runtime not initialized")
    runtime = RUNTIMES["rp"]
    final_postings, limit = combine_any_all(runtime, body)
    only_source = bool(body.get("only_source", False))
    out_format = (body.get("format") or "json").lower()
    with_meta = bool(body.get("with_meta", False))

    docs = runtime.fetch_docs(final_postings, limit=limit)
    if only_source:
        docs = [d.get("_source") for d in docs if isinstance(d, dict) and "_source" in d]

    if out_format == "ndjson":
        lines = [json.dumps(d, ensure_ascii=False) for d in docs]
        return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""),
                                 media_type="application/x-ndjson; charset=utf-8")
    else:
        return JSONResponse({"meta": {"matched_postings": len(final_postings), "returned_docs": len(docs)}, "docs": docs} if with_meta else docs)

# --------- 두 인덱스 동시에 검색해서 합치는 엔드포인트(선택) ----------
@app.post("/search_both")
def search_both(body: Dict[str, Any] = Body(...)):
    # mode: union | intersect | rp_only | nw_only (기본 union)
    mode = (body.get("mode") or "union").lower()
    if "nw" not in RUNTIMES or "rp" not in RUNTIMES:
        raise HTTPException(500, "Both runtimes must be initialized")

    rt_nw = RUNTIMES["nw"]
    rt_rp = RUNTIMES["rp"]

    postings_nw, limit = combine_any_all(rt_nw, body)
    postings_rp, _     = combine_any_all(rt_rp, body)

    if mode == "intersect":
        final_postings = postings_nw & postings_rp
        # 주의: 서로 다른 파일셋이므로 교집합은 보통 빈집합이 됩니다.
        # 동일 문서를 양쪽에 복사한 경우가 아니라면 union을 권장합니다.
    elif mode == "rp_only":
        final_postings = postings_rp
        runtime = rt_rp
    elif mode == "nw_only":
        final_postings = postings_nw
        runtime = rt_nw
    else:
        final_postings = postings_nw | postings_rp
        # fetch 시 어떤 런타임에서 읽을지 결정 필요 → 단순화를 위해
        # 여기서는 "nw 먼저, 그 다음 rp" 순서로 읽습니다.
    only_source = bool(body.get("only_source", False))
    out_format = (body.get("format") or "json").lower()
    with_meta = bool(body.get("with_meta", False))

    # postings가 어느 런타임의 파일인지 구분 필요:
    # 간단하게 "file_id 범위"가 서로 다르다고 가정하지 말고,
    # 실제 문서 복원을 각 런타임별로 분리 fetch 후 결합합니다.
    docs = []
    if mode in ("nw_only", "intersect", "union"):
        docs += rt_nw.fetch_docs(postings_nw, limit=None)
    if mode in ("rp_only", "intersect", "union"):
        docs += rt_rp.fetch_docs(postings_rp, limit=None)

    # limit을 여기서 적용
    if limit:
        docs = docs[:limit]

    if only_source:
        docs = [d.get("_source") for d in docs if isinstance(d, dict) and "_source" in d]

    if out_format == "ndjson":
        lines = [json.dumps(d, ensure_ascii=False) for d in docs]
        return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""),
                                 media_type="application/x-ndjson; charset=utf-8")
    else:
        meta = {
            "matched_postings_nw": len(postings_nw),
            "matched_postings_rp": len(postings_rp),
            "returned_docs": len(docs),
            "mode": mode
        }
        return JSONResponse({"meta": meta, "docs": docs} if with_meta else docs)

# ===================== 실행/초기화 =====================
def parse_args():
    ap = argparse.ArgumentParser(description="Run KV Search API (two LMDB runtimes).")
    ap.add_argument("--lmdb-nw", default="kv_index_nw.lmdb")
    ap.add_argument("--manifest-nw", default="manifest_nw.json")
    ap.add_argument("--lmdb-rp", default="kv_index_rp.lmdb")
    ap.add_argument("--manifest-rp", default="manifest_rp.json")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    return ap.parse_args()

def create_app(args) -> FastAPI:
    # 두 런타임을 동시에 메모리에 올림 (mmap + LMDB readonly)
    RUNTIMES["nw"] = SearchRuntime(args.lmdb_nw, args.manifest_nw)
    RUNTIMES["rp"] = SearchRuntime(args.lmdb_rp, args.manifest_rp)
    return app

if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)