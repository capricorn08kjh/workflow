### 1. **구현 환경 가정**
- **모델**: 회사 서버에서 제공하는 OpenAI-like API (예: LLaMA, GPT-4o 호환 엔드포인트).
- **LlamaIndex**: OpenAI-like API를 지원하는 LlamaIndex의 `OpenAILike` 설정 사용.
- **문서**: 수천~수만 건의 회의록 (PDF, Word, 텍스트 등).
- **목표**: "김영철"이 고객사 담당자로 언급된 문서를 검색하고 요약.
- **벡터DB**: LlamaIndex가 지원하는 벡터DB (예: FAISS, Weaviate, Chroma 등) 중 하나 선택.
- **제약**: 메타데이터를 수동으로 생성하지 않고 자동화.

### 1.1 **폴더 구조**
```
project_root/
├── config/
│   ├── config.yaml          # 설정 파일 (API 키, 엔드포인트, 벡터DB 설정 등)
│   └── prompts.yaml         # LLM 프롬프트 템플릿 (메타데이터 추출, 요약 등)
├── data/
│   ├── raw/                # 원본 회의록 파일 (PDF, Word, 텍스트)
│   ├── processed/          # 전처리된 텍스트 파일 (옵션)
│   └── metadata/           # 추출된 메타데이터 JSON 파일
├── scripts/
│   ├── preprocess.py       # 문서 로드 및 전처리 스크립트
│   ├── extract_metadata.py # 메타데이터 추출 (LLM/NER)
│   ├── index.py            # 벡터DB 인덱싱
│   ├── query.py            # 검색 및 요약 쿼리 실행
│   └── utils.py            # 공통 유틸리티 함수 (텍스트 정규화, 캐싱 등)
├── vector_store/
│   ├── faiss_index/        # FAISS 벡터 인덱스 저장
│   └── cache/              # 검색 결과 캐싱 (예: Redis 또는 로컬 JSON)
├── logs/
│   ├── processing.log      # 문서 처리 로그
│   └── query.log           # 쿼리 및 검색 로그
├── requirements.txt         # Python 종속성
└── README.md               # 프로젝트 설명 및 실행 방법
```

### 2. **LlamaIndex 설정 및 OpenAI-like API 통합**
LlamaIndex에서 회사 서버의 OpenAI-like API를 사용하려면, `OpenAILike` 설정을 통해 커스텀 엔드포인트를 지정합니다.

```python
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings

# OpenAI-like API 설정
llm = OpenAILike(
    api_base="https://your-company-api-endpoint",  # 회사 서버 API 엔드포인트
    api_key="your-api-key",  # 회사 제공 API 키
    model="your-model-name"  # 예: gpt-4o, llama-3
)

# 임베딩 모델 설정 (회사 API가 임베딩도 지원한다고 가정)
embedding_model = OpenAILike(
    api_base="https://your-company-api-endpoint",
    api_key="your-api-key",
    model="your-embedding-model"  # 예: text-embedding-ada-002 호환 모델
)

# LlamaIndex 글로벌 설정
Settings.llm = llm
Settings.embed_model = embedding_model
```

- **참고**: 회사 API가 OpenAI와 호환되지 않는 경우, LlamaIndex의 커스텀 LLM 어댑터를 구현해야 할 수 있습니다. API 스펙(엔드포인트, 요청/응답 형식)을 확인하세요.

---

### 3. **자동 메타데이터 추출**
수천~수만 건의 문서를 처리하기 위해 메타데이터(참석자, 역할, 고객사 등)를 자동으로 추출합니다. LlamaIndex의 `MetadataExtractor`와 회사 LLM을 결합해 구현합니다.

#### (1) **문서 로드 및 전처리**
- **문서 로드**: LlamaIndex의 `SimpleDirectoryReader`로 PDF, Word, 텍스트 파일 로드.
- **텍스트 추출**: PDF/Word의 경우, `PyPDF2`, `python-docx`, 또는 `pdfplumber`로 텍스트 추출.
- 예:
  ```python
  from llama_index.core import SimpleDirectoryReader
  from pdfplumber import open as pdf_open

  # PDF/Word에서 텍스트 추출
  def extract_text(file_path):
      if file_path.endswith(".pdf"):
          with pdf_open(file_path) as pdf:
              return "\n".join(page.extract_text() for page in pdf.pages)
      elif file_path.endswith(".docx"):
          from docx import Document
          doc = Document(file_path)
          return "\n".join(p.text for p in doc.paragraphs)
      return open(file_path).read()

  # 문서 로드
  documents = SimpleDirectoryReader(
      input_dir="meeting_notes/",
      file_extractor={".pdf": extract_text, ".docx": extract_text}
  ).load_data()
  ```

#### (2) **메타데이터 추출**
- **LLM 기반 추출**: LlamaIndex의 `LLMMetadataExtractor`를 사용해 회사 LLM으로 참석자, 역할, 고객사 등을 추출.
- **프롬프트 설계**:
  ```python
  from llama_index.core.extractors import LLMMetadataExtractor
  from llama_index.core.node_parser import SentenceSplitter

  # LLM 기반 메타데이터 추출기
  extractor = LLMMetadataExtractor(
      llm=llm,
      prompt_template="다음 텍스트에서 참석자, 역할(예: 고객사 담당자), 고객사 이름을 JSON으로 추출:\n{text}\n출력 형식: {'participants': [], 'roles': {}, 'customer_company': null}"
  )

  # 문서 분할 (청크 단위 처리)
  node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
  nodes = node_parser.get_nodes_from_documents(documents)

  # 메타데이터 추출
  for node in nodes:
      metadata = extractor.extract([node])[0]
      node.metadata.update(metadata)
  ```
- **예시 출력**:
  ```json
  {
      "participants": ["김영철", "이민수"],
      "roles": {"김영철": "고객사 담당자"},
      "customer_company": "ABC Corp"
  }
  ```

#### (3) **NER 보완 (옵션)**
- 회사 API가 NER을 지원하지 않는 경우, 오픈소스 NER 모델(예: `klue/ner`)을 로컬에서 실행해 보완.
- 예:
  ```python
  from transformers import pipeline
  ner = pipeline("ner", model="klue/ner")
  def extract_entities(text):
      entities = ner(text)
      metadata = {"participants": [], "customer_company": None}
      for entity in entities:
          if entity["entity"].startswith("B-PER"):
              metadata["participants"].append(entity["word"])
          elif entity["entity"].startswith("B-ORG"):
              metadata["customer_company"] = entity["word"]
      return metadata
  ```

---

### 4. **벡터DB 설정 및 인덱싱**
LlamaIndex는 FAISS, Weaviate, Chroma 등 다양한 벡터DB를 지원합니다. 대규모 문서 처리에 적합한 **FAISS**를 예로 들어 설명합니다.

#### (1) **벡터 인덱스 생성**
- 문서와 메타데이터를 벡터화하여 FAISS에 저장.
- 예:
  ```python
  from llama_index.core import VectorStoreIndex, StorageContext
  from llama_index.vector_stores.faiss import FaissVectorStore
  import faiss

  # FAISS 벡터 스토어 초기화
  faiss_index = faiss.IndexFlatL2(1536)  # 임베딩 차원 (회사 API에 맞게 조정)
  vector_store = FaissVectorStore(faiss_index=faiss_index)
  storage_context = StorageContext.from_defaults(vector_store=vector_store)

  # 벡터 인덱스 생성
  index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embedding_model)
  ```

#### (2) **메타데이터 인덱싱**
- LlamaIndex는 메타데이터를 자동으로 인덱싱하며, 검색 시 필터링 가능.
- 메타데이터 필드(`participants`, `roles`, `customer_company`)를 쿼리 필터로 사용.

---

### 5. **검색 및 하이브리드 검색**
"김영철이 고객사 담당자로 언급된 회의록"을 검색하기 위해 메타데이터 필터링과 벡터 검색을 결 biblio합합니다.

#### (1) **쿼리 처리**
- 쿼리: "회의록 중 고객사 담당자인 김영철씨가 언급된 회의록을 찾아주고 내용을 요약해줘"
- 메타데이터 필터: `participants`에 "김영철", `roles`에 "고객사 담당자" 포함.
- 벡터 검색: 쿼리 전체를 임베딩해 유사 문서 검색.

#### (2) **구현**
```python
from llama_index.core import QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever

# 검색 엔진 설정
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    filters={
        "participants": "김영철",
        "roles": {"name": "김영철", "role": "고객사 담당자"}
    }
)

# 쿼리 실행
query = "회의록 중 고객사 담당자인 김영철씨가 언급된 회의록"
nodes = retriever.retrieve(QueryBundle(query_str=query))

# 검색 결과 확인
for node in nodes:
    print(f"Document: {node.metadata}, Score: {node.score}")
```

#### (3) **하이브리드 검색 (옵션)**
- LlamaIndex는 기본적으로 벡터 검색을 사용하지만, 키워드 검색(BM25)을 결합하려면 `BM25Retriever`를 추가로 사용.
- 예:
  ```python
  from llama_index.core.retrievers import BM25Retriever
  from llama_index.core import get_response_synthesizer
  from llama_index.core.query_engine import RetrieverQueryEngine

  # BM25 리트리버
  bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

  # 하이브리드 검색: 벡터 + BM25
  from llama_index.core.retrievers import BaseRetriever
  class HybridRetriever(BaseRetriever):
      def __init__(self, vector_retriever, bm25_retriever):
          self.vector_retriever = vector_retriever
          self.bm25_retriever = bm25_retriever

      def _retrieve(self, query_bundle):
          vector_nodes = self.vector_retriever.retrieve(query_bundle)
          bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
          # 점수 결합 (예: 가중치 평균)
          combined_nodes = vector_nodes[:5] + bm25_nodes[:5]  # 간단 예시
          return combined_nodes

  hybrid_retriever = HybridRetriever(retriever, bm25_retriever)
  ```

---

### 6. **요약 생성**
검색된 문서를 회사 LLM에 입력해 요약을 생성합니다.

```python
from llama_index.core.response_synthesizers import get_response_synthesizer

# 응답 합성기 설정
synthesizer = get_response_synthesizer(
    llm=llm,
    response_mode="compact"  # 간결한 요약
)

# 쿼리 엔진 설정
query_engine = RetrieverQueryEngine(
    retriever=hybrid_retriever,
    response_synthesizer=synthesizer
)

# 쿼리 실행 및 요약
response = query_engine.query(
    "회의록 중 고객사 담당자인 김영철씨가 언급된 회의록을 찾아주고 내용을 요약해줘"
)
print(response)  # 예: "김영철씨는 ABC Corp 담당자로, 2025년 6월 10일 회의에서 프로젝트 일정 논의."
```

---

### 7. **대규모 문서 처리 최적화**
- **배치 처리**:
  - 문서 로드 및 메타데이터 추출을 병렬화. Python의 `multiprocessing` 또는 회사 서버의 분산 컴퓨팅 자원 활용.
  - 예: `concurrent.futures`로 병렬 처리.
    ```python
    from concurrent.futures import ProcessPoolExecutor
    def process_document(doc):
        node = node_parser.get_nodes_from_documents([doc])[0]
        metadata = extractor.extract([node])[0]
        node.metadata.update(metadata)
        return node

    with ProcessPoolExecutor() as executor:
        nodes = list(executor.map(process_document, documents))
    ```
- **증분 인덱싱**:
  - LlamaIndex의 `index.insert`로 새 문서만 추가.
  - 예: `index.insert_nodes(new_nodes)`
- **캐싱**:
  - 자주 검색되는 쿼리(예: "김영철") 결과를 Redis 또는 로컬 캐시에 저장.
- **스케일링**:
  - FAISS는 대규모 문서에 적합하지만, 메모리 사용량이 많을 수 있음. Weaviate나 Chroma로 전환 가능.
  - 회사 서버의 GPU/TPU 자원을 활용해 임베딩 속도 향상.

---

### 8. **추가 고려사항**
- **한국어 최적화**:
  - 회사 API가 한국어 특화 모델(예: KoBERT, HyperCLOVA 호환)을 제공한다면, 이를 사용해 임베딩 및 요약 품질 향상.
  - "김영철" 같은 이름의 동음이의어 처리: 메타데이터(`customer_company`)로 구체화.
- **프라이버시**:
  - 개인정보(예: "김영철") 처리 시, 회사 서버의 데이터 마스킹 또는 암호화 정책 준수.
- **비용 관리**:
  - 회사 API 호출 횟수를 최소화하기 위해, 메타데이터 추출은 오프라인에서 한 번만 실행.
  - LlamaIndex의 `ResponseSynthesizer`에서 `max_tokens`를 설정해 요약 길이 제한.
- **오타 처리**:
  - "김영철"의 오타(예: "김영철 ")를 처리하기 위해, 텍스트 정규화 후 인덱싱.
  - 예: `node.text = node.text.replace("김영철 ", "김영철")`

---

### 9. **구체적 워크플로우**
1. **문서 로드**:
   - `SimpleDirectoryReader`로 회의록 로드 (PDF/Word 지원).
2. **메타데이터 추출**:
   - `LLMMetadataExtractor`로 회사 LLM을 사용해 참석자, 역할, 고객사 추출.
   - NER(로컬 또는 회사 API)로 보완.
3. **인덱싱**:
   - `VectorStoreIndex`와 FAISS로 문서와 메타데이터 인덱싱.
4. **검색**:
   - 메타데이터 필터(`participants: 김영철, roles: 고객사 담당자`) + 벡터 검색.
   - 하이브리드 검색으로 키워드(BM25)와 벡터 검색 결합.
5. **요약**:
   - 검색된 문서를 회사 LLM에 입력해 요약 생성.
6. **출력**:
   - 예: "김영철씨는 ABC Corp 담당자로, 2025년 6월 10일 회의에서 프로젝트 일정 논의."

---

---
### 10. **실행방법**
####실행 방법
1. config/config.yaml에 API 설정 입력.
2. python scripts/preprocess.py
3. python scripts/extract_metadata.py
4. python scripts/index.py
5. python scripts/query.py
---
