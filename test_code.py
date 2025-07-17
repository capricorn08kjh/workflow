# 1. 라이브러리 로드
from llama_index import (
    SimpleDirectoryReader, GPTVectorStoreIndex,
    SQLDatabase, SQLDatabaseChain,
    LLMPredictor, ServiceContext, PromptTemplate
)
from openai import OpenAI
import json

# 2. 데이터 로드 및 인덱싱
# -- 비정형
docs = SimpleDirectoryReader("data/unstructured/").load_data()
index_unstruct = GPTVectorStoreIndex.from_documents(docs)

# -- 정형
db = SQLDatabase.from_uri("postgresql://user:pass@host:5432/dbname")
index_struct = SQLDatabaseChain.from_llm(
    llm=OpenAI(temperature=0), database=db
)

# 3. LLM 셋업
service_ctx = ServiceContext.from_defaults(
    llm_predictor=LLMPredictor(OpenAI(temperature=0))
)
orch_prompt = PromptTemplate(
    # 툴 호출 스펙을 JSON으로 생성하라
    template="""
당신은 툴 오케스트레이터입니다.
사용자 질문에 맞춰 아래 형식의 JSON 리스트를 반환하세요.
{
  "steps": [
    {"tool": "VectorSearch", "input": {"query": "...", "filter": {}}, "output_key": "..."},
    {"tool": "SQLSearch",    "input": {"query": "..."},               "output_key": "..."},
    {"tool": "ChartGenerator","input": {"source_key": "...", "chart_type": "bar"}, "output_key": "..."}
  ]
}
질문: {query}
""",
    input_variables=["query"]
)
orch_llm = LLMPredictor(OpenAI(temperature=0))

resp_prompt = PromptTemplate(
    # 최종 결과를 자연어 + 출처까지 합성하라
    template="""
다음 도구 출력 결과를 바탕으로,
1) 사용자에게 자연어로 답변 작성
2) 각 결과마다 [출처: source_id] 형태로 표기

{context}
""",
    input_variables=["context"]
)
resp_llm = LLMPredictor(OpenAI(temperature=0))

# 4. Orchestrator → MCP(JSON) 생성
user_query = "2024년 3분기 DRAM 실적을 차트로 보여줘"
mcp_json = orch_llm.predict(prompt=orch_prompt, query=user_query).strip()
mcp = json.loads(mcp_json)

# 5. MCP Executor
# (실제로는 instantiate chart function 등 추가)
def make_chart(df, chart_type="bar"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if chart_type=="bar":
        ax.bar(df["period"], df["value"])
    else:
        ax.plot(df["period"], df["value"])
    return fig

results = {}
for step in mcp["steps"]:
    tool = step["tool"]
    out = None

    if tool == "VectorSearch":
        out = index_unstruct.query(
            step["input"]["query"],
            filter_metadata=step["input"].get("filter", {})
        )

    elif tool == "SQLSearch":
        out = index_struct.run(step["input"]["query"])

    elif tool == "ChartGenerator":
        df = results[step["input"]["source_key"]]
        out = make_chart(df, chart_type=step["input"]["chart_type"])

    # 출력 저장
    results[step["output_key"]] = out

# 6. Response LLM: 최종 합성 + 출처 표기
context_str = json.dumps(
    {k: (v if not hasattr(v, "columns") else "DataFrame(...)") for k,v in results.items()},
    ensure_ascii=False
)
final_response = resp_llm.predict(prompt=resp_prompt, context=context_str)

print(final_response)
