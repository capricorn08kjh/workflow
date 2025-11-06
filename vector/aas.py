import pickle
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from typing import List

BM25_PATH = "./bm25_index.pkl"

# -----------------------------
# 1. BM25 retriever 생성 및 저장
# -----------------------------
def build_and_save_bm25(docs: List[Document], save_path: str = BM25_PATH):
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 8
    with open(save_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"✅ BM25 retriever saved to {save_path}")
    return bm25

# -----------------------------
# 2. BM25 retriever 불러오기
# -----------------------------
def load_bm25(save_path: str = BM25_PATH):
    with open(save_path, "rb") as f:
        bm25 = pickle.load(f)
    print(f"✅ BM25 retriever loaded from {save_path}")
    return bm25
    
    
# 사용예시
from pathlib import Path

if Path(BM25_PATH).exists():
    bm25 = load_bm25(BM25_PATH)
else:
    # load_all_docs_from_chroma()로 복원한 문서 리스트 사용
    bm25 = build_and_save_bm25(docs_chunked, BM25_PATH)

# 이후 검색은 바로 가능
results = bm25.invoke("25년도 12주차 주간보고")
for r in results[:3]:
    print(r.page_content[:80])
    
    



