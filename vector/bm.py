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