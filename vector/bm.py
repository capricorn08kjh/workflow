from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import chromadb

# (선택) 쿼리 임베딩용 — 이미 쓰고 계신 걸 그대로 쓰세요 (bge-m3 등)
from langchain_community.embeddings import HuggingFaceEmbeddings
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

CHROMA_PATH = "./your_chroma_path"
COLLECTION = "your_collection_name"

# 1) 기존 컬렉션 재사용
client = chromadb.PersistentClient(path=CHROMA_PATH)
vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION,
    embedding_function=embedder,  # similarity_search에 필요
)

# 2) 컬렉션에서 모든 Document 복원 (BM25 구축용)
def load_all_docs_from_chroma(vs: Chroma, page_size: int = 5000) -> List[Document]:
    coll = vs._collection
    total = coll.count()
    out: List[Document] = []
    for offset in range(0, total, page_size):
        batch = coll.get(
            include=["documents", "metadatas", "ids"],
            limit=page_size,
            offset=offset
        )
        for doc, meta, _id in zip(batch["documents"], batch["metadatas"], batch["ids"]):
            meta = (meta or {}) | {"_id": _id}
            out.append(Document(page_content=doc, metadata=meta))
    return out

docs_chunked = load_all_docs_from_chroma(vectorstore)
print(f"Loaded {len(docs_chunked)} docs from Chroma")



from collections import defaultdict

def _meta_filter(d: Document, filt: Dict[str, Any]) -> bool:
    if not filt:
        return True
    m = d.metadata or {}
    for k, v in filt.items():
        if m.get(k) != v:
            return False
    return True

def hybrid_search(
    query: str,
    k_dense: int = 8,
    k_bm25: int = 8,
    meta_filter: Dict[str, Any] | None = None,
    mode: str = "rrf",           # "rrf" or "weighted"
    w_dense: float = 0.5,        # weighted 모드일 때
    w_bm25: float = 0.5
):
    # 1) Dense (점수 포함)
    dense_results = vectorstore.similarity_search_with_relevance_scores(
        query, k=k_dense, filter=meta_filter
    )  # -> List[(Document, score: float 0~1)]

    # 2) BM25 (후처리로 메타 필터)
    bm25_raw = bm25.invoke(query)  # List[Document]
    bm25_results = [d for d in bm25_raw if _meta_filter(d, meta_filter)]

    # 3) 결합
    if mode == "rrf":
        # RRF: 1/(k + rank)
        K = 60
        def doc_key(d: Document):
            return d.metadata.get("_id") or (d.metadata.get("source","") + "::" + d.metadata.get("chunk_id",""))

        score_map = defaultdict(float)
        order_map = {}

        # Dense 순위 반영
        for r, item in enumerate(dense_results, start=1):
            d = item[0]
            key = doc_key(d)
            score_map[key] += 1.0 / (K + r)
            order_map[key] = d

        # BM25 순위 반영
        for r, d in enumerate(bm25_results, start=1):
            key = doc_key(d)
            score_map[key] += 1.0 / (K + r)
            order_map.setdefault(key, d)

        fused = sorted(order_map.keys(), key=lambda k: score_map[k], reverse=True)
        docs = [order_map[k] for k in fused]
        return docs

    else:  # weighted
        def doc_key(d: Document):
            return d.metadata.get("_id") or (d.metadata.get("source","") + "::" + d.metadata.get("chunk_id",""))

        dense_score = {}
        dense_docs = {}
        for d, s in dense_results:
            k = doc_key(d); dense_score[k] = float(s); dense_docs[k] = d

        bm25_score = {}
        bm25_docs = {}
        for rank, d in enumerate(bm25_results, start=1):
            k = doc_key(d); bm25_score[k] = 1.0 / rank; bm25_docs[k] = d

        keys = set(dense_score.keys()) | set(bm25_score.keys())
        merged = []
        for k in keys:
            ds = dense_score.get(k, 0.0)
            bs = bm25_score.get(k, 0.0)
            merged.append((w_dense*ds + w_bm25*bs, k))
        merged.sort(key=lambda x: x[0], reverse=True)

        # 원본 문서 복원
        docs = []
        for _, k in merged:
            d = dense_docs.get(k) or bm25_docs.get(k)
            docs.append(d)
        return docs
        
        
        
        
q = "25년도 12주차 주간보고 요약"
top_docs = hybrid_search(
    q,
    k_dense=8, k_bm25=8,
    meta_filter={"team": "R&D"},   # 필요없으면 None
    mode="rrf"                     # 또는 "weighted"
)[:10]

for i, d in enumerate(top_docs, 1):
    print(f"[{i}] {d.metadata} :: {d.page_content[:80]}...")
    
    
    


