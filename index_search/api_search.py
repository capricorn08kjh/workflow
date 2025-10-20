# api_search.py
import argparse
import json
import mmap
from typing import List, Optional, Set, Tuple, Dict, Any, Union

import lmdb
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

SEP = "\x1f"

# ------------------------------
# Start-up: load manifest, open mmaps, open LMDB
# ------------------------------
class SearchRuntime:
    def __init__(self, lmdb_path: str, manifest_path: str):
        self.manifest = self._load_manifest(manifest_path)
        self.files_meta = self.manifest["files"]
        self.files, self.mmaps = self._open_mmaps(self.files_meta)
        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, max_dbs=2)
        self.kv_db = self.env.open_db(b"kv")

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

    # ---- core: postings ----
    def postings_for(self, key_path: str, value: str) -> Set[Tuple[int, int]]:
        """Return set of (file_id, offset) for exact-match term."""
        k = f"{key_path}{SEP}{value}".encode("utf-8")
        out: Set[Tuple[int, int]] = set()
        with self.env.begin(db=self.kv_db) as txn, txn.cursor(db=self.kv_db) as cur:
            if cur.set_key(k):
                vv = cur.value()
                while True:
                    file_id_s, offset_s = vv.decode("utf-8").split(SEP, 1)
                    out.add((int(file_id_s), int(offset_s)))
                    if not cur.next_dup():
                        break
        return out

    def fetch_docs(self, postings: Set[Tuple[int, int]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read lines at offsets and decode JSON."""
        # 정렬해서 읽기(파일 순/오프셋 순) → 재현성
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


app = FastAPI(title="KV Index Search API", version="1.0.0")
RUNTIME: Optional[SearchRuntime] = None

# ------------------------------
# Query model
# ------------------------------
"""
쿼리 형태(간단/실용 설계):

- any:  OR 그룹 (이 안의 조건들 중 하나라도 매칭되면 통과)
- all:  AND 그룹 (이 안의 조건들을 모두 만족해야 통과)
- 각 조건은 다음 둘 중 하나:
    {"key":"_source.morps[].text", "value":"서울역"}
    {"key":"_source.morps[].pos", "values":["NNP","NNG"]}    # values = 여러 값 OR

최종 결과 집합 = (ANY 그룹의 합집합) ∩ (ALL 그룹의 교집합)
- any/all 둘 중 하나만 주어도 동작
"""

class _Sentinel:
    pass

SENTINEL = _Sentinel()  # 내부용


def compute_postings_union(runtime: SearchRuntime, conds: List[Dict[str, Any]]) -> Set[Tuple[int, int]]:
    """여러 조건을 OR로 합집합."""
    union_set: Set[Tuple[int, int]] = set()
    first = True
    for c in conds:
        key = c["key"]
        if "values" in c:
            vals = c["values"]
            s: Set[Tuple[int, int]] = set()
            for v in vals:
                s |= runtime.postings_for(key, str(v))
        else:
            v = c["value"]
            s = runtime.postings_for(key, str(v))
        if first:
            union_set = set(s)
            first = False
        else:
            union_set |= s
    return union_set


def compute_postings_intersection(runtime: SearchRuntime, conds: List[Dict[str, Any]]) -> Set[Tuple[int, int]]:
    """여러 조건을 AND로 교집합."""
    inter_set: Set[Tuple[int, int]] = set()
    first = True
    for c in conds:
        key = c["key"]
        if "values" in c:
            # 같은 key에 여러 values → 내부적으로 OR 후, 전체 AND에 참여
            vals = c["values"]
            s: Set[Tuple[int, int]] = set()
            for v in vals:
                s |= runtime.postings_for(key, str(v))
        else:
            v = c["value"]
            s = runtime.postings_for(key, str(v))
        if first:
            inter_set = set(s)
            first = False
        else:
            inter_set &= s
        # 일찍 끝내기
        if not inter_set:
            break
    return inter_set


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/search")
def search(
    body: Dict[str, Any] = Body(
        ...,
        example={
            "any": [
                {"key": "_source.morps[].text", "value": "서울역"},
                {"key": "_source.title", "values": ["서울역", "용산역"]},
            ],
            "all": [
                {"key": "_source.morps[].pos", "values": ["NNP", "NNG"]},
            ],
            "limit": 100,
            "only_source": False,
            "format": "json"  # "json" | "ndjson"
        },
    )
):
    """
    POST /search
    Body:
      any:  [ {key, value|values}, ... ]    # OR
      all:  [ {key, value|values}, ... ]    # AND
      limit: int
      only_source: bool
      format: "json" | "ndjson"

    반환:
      - format=json  → JSON 배열
      - format=ndjson→ NDJSON 문자열(한 줄 한 문서)
    """
    if RUNTIME is None:
        raise HTTPException(500, "Runtime is not initialized")

    any_conds: List[Dict[str, Any]] = body.get("any") or []
    all_conds: List[Dict[str, Any]] = body.get("all") or []
    limit: Optional[int] = body.get("limit")
    only_source: bool = bool(body.get("only_source", False))
    out_format: str = (body.get("format") or "json").lower()
    if out_format not in ("json", "ndjson"):
        raise HTTPException(400, "format must be 'json' or 'ndjson'")

    # 비어있는 쿼리 방지
    if not any_conds and not all_conds:
        raise HTTPException(400, "Provide at least one of 'any' or 'all'.")

    # 1) any = 합집합
    any_set: Optional[Set[Tuple[int, int]]] = None
    if any_conds:
        any_set = compute_postings_union(RUNTIME, any_conds)

    # 2) all = 교집합
    all_set: Optional[Set[Tuple[int, int]]] = None
    if all_conds:
        all_set = compute_postings_intersection(RUNTIME, all_conds)

    # 3) 결합
    if any_set is not None and all_set is not None:
        final_set = any_set & all_set
    elif any_set is not None:
        final_set = any_set
    else:
        final_set = all_set or set()

    # 4) 문서 읽기
    docs = RUNTIME.fetch_docs(final_set, limit=limit)

    # 5) only_source 옵션
    if only_source:
        docs = [d.get("_source") for d in docs if isinstance(d, dict) and "_source" in d]

    # 6) 형식화하여 반환
    if out_format == "ndjson":
        # FastAPI는 문자열 그대로 내려주기
        lines = []
        for d in docs:
            lines.append(json.dumps(d, ensure_ascii=False))
        payload = "\n".join(lines) + ("\n" if lines else "")
        return PlainTextResponse(payload, media_type="application/x-ndjson; charset=utf-8")
    else:
        return JSONResponse(docs)


# ------------------------------
# Entrypoint
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Run KV Search API server.")
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
    # uvicorn api_search:app --host 0.0.0.0 --port 8000  (외부 실행 방식)
    # 또는 아래처럼 직접 run 하려면:
    import uvicorn
    args = parse_args()
    create_app(args.lmdb, args.manifest)
    uvicorn.run(app, host=args.host, port=args.port)