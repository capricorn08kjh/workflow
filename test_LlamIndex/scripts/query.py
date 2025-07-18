from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever

index = VectorStoreIndex.load_from_persist_dir(persist_dir="vector_store/faiss_index")
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    filters={"participants": "김영철", "roles": {"name": "김영철", "role": "고객사 담당자"}}
)
query = "회의록 중 고객사 담당자인 김영철씨가 언급된 회의록을 찾아주고 내용을 요약해줘"
nodes = retriever.retrieve(QueryBundle(query_str=query))
synthesizer = get_response_synthesizer(llm=Settings.llm, response_mode="compact")
response = synthesizer.synthesize(query, nodes)
print(response)
