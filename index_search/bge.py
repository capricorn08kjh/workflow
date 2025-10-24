from __future__ import annotations
from typing import List, Iterable, Dict, Any
from dataclasses import dataclass

from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import chromadb

# BGE-M3 (dense + sparse) from FlagEmbedding
from FlagEmbedding import BGEM3FlagModel  # pip install FlagEmbedding

# -----------------------------
# 0) 청킹 유틸 (문자 기준; 필요시 토큰 기준으로 교체)
# -----------------------------
def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size>0, 0<=overlap<chunk_size 조건을 확인하세요.")
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks

def chunk_documents(
    docs: Iterable[Document],
    chunk_size: int = 600,
    overlap: int = 100
) -> List[Document]:
    out = []
    for d in docs:
        for idx, ch in enumerate(chunk_text(d.page_content, chunk_size, overlap)):
            md = dict(d.metadata) if d.metadata else {}
            md.update({"chunk_id": idx})
            out.append(Document(page_content=ch, metadata=md))
    return out

# -----------------------------
# 1) BGE-M3 임베딩 래퍼
#    - encode(..., return_dense=True, return_sparse=True) 사용
#    - dense_vecs: List[List[float]]
#    - lexical_weights: List[Dict[str, float]]  (sparse 표현)
# -----------------------------
@dataclass
class BGEM3Embedder:
    model_name: str = "BAAI/bge-m3"
    use_fp16: bool = True
    max_length: int = 8192
    batch_size: int = 1024

    def __post_init__(self):
        self.model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)

    def encode_texts(self, texts: List[str]):
        """
        return:
          dense_vecs: List[List[float]]
          lexical_weights: List[Dict[str, float]]
        """
        out = self.model.encode(
            sentences=texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        # FlagEmbedding의 BGEM3는 위 옵션을 주면 dict로 반환하며
        # 'dense_vecs'와 'lexical_weights' 키를 포함합니다.
        dense_vecs = out["dense_vecs"]
        lexical_weights = out["lexical_weights"]
        return dense_vecs, lexical_weights

    def encode_query_dense(self, q: str) -> List[float]:
        # 검색 시 dense 질의 임베딩 (간단 버전)
        out = self.model.encode(
            sentences=[q],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"][0]

    def encode_query_sparse(self, q: str) -> Dict[str, float]:
        out = self.model.encode(
            sentences=[q],
            batch_size=1,
            max_length=self.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return out["lexical_weights"][0]

    def lexical_match_score(self, q_lex: Dict[str, float], d_lex: Dict[str, float]) -> float:
        # BGEM3가 제공하는 sparse 간 점수 계산 유틸을 사용
        return self.model.compute_lexical_matching_score(q_lex, d_lex)

# -----------------------------
# 2) Chroma 구축 함수 (LangChain VectorStore)
#    - dense는 embeddings 인자로 저장
#    - sparse는 metadata['lexical_weights']에 함께 저장
# -----------------------------
def make_chromadb_with_bge_m3(
    db_name: str,
    documents: List[Document],
    persist_path: str = "./chroma_bge_m3",
    batch_size: int = 1024,
    chunk_size: int = 600,
    overlap: int = 100,
):
    # Chroma Persistent Client
    client = chromadb.PersistentClient(path=persist_path)

    # embedding_function은 생성 시 필요하지만,
    # 아래에서 add_documents(..., embeddings=...)로 직접 주입하므로 호출되지 않습니다.
    dummy_embed_fn = lambda x: [_ for _ in x]  # 사용되지 않음

    vectorstore = Chroma(
        client=client,
        collection_name=db_name,
        embedding_function=dummy_embed_fn,
    )

    # 1) 청킹
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

    # 2) 임베딩: BGE-M3 dense + sparse
    embedder = BGEM3Embedder(batch_size=batch_size)

    # 3) 배치 업서트
    total = len(chunks)
    for i in tqdm(range(0, total, batch_size), desc="Upserting to Chroma"):
        batch_docs = chunks[i : i + batch_size]
        texts = [d.page_content for d in batch_docs]

        dense_vecs, lexical_weights = embedder.encode_texts(texts)

        # sparse를 metadata에 같이 저장
        metadatas = []
        for d, lw in zip(batch_docs, lexical_weights):
            md = dict(d.metadata) if d.metadata else {}
            md["lexical_weights"] = lw  # sparse 표현 저장
            metadatas.append(md)

        # embeddings 인자로 dense 주입 (LangChain Chroma가 그대로 사용)
        vectorstore.add_documents(
            documents=batch_docs,
            embeddings=dense_vecs,
            metadatas=metadatas,
        )

    return vectorstore, embedder

# -----------------------------
# 3) 예시 사용
# -----------------------------
if __name__ == "__main__":
    # 예시 문서
    raw_docs = [
        Document(page_content="지능형 CCTV는 화재·침입·배회 등의 이상행동을 탐지한다.", metadata={"source":"cctv_guide"}),
        Document(page_content="BGE-M3는 dense, sparse, colbert 다기능 임베딩을 지원한다.", metadata={"source":"bge_m3"}),
    ]
    vs, emb = make_chromadb_with_bge_m3(
        db_name="hybrid_bge_m3",
        documents=raw_docs,
        persist_path="./db_bge_m3",
        batch_size=1024,
        chunk_size=600,
        overlap=100,
    )
    print("✅ Chroma 구축 완료")