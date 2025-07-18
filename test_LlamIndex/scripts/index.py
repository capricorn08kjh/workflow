from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

faiss_index = faiss.IndexFlatL2(1536)  # 임베딩 차원
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)
index.storage_context.persist(persist_dir="vector_store/faiss_index")
