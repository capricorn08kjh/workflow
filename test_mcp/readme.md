# MCP RAG 서버 설치 및 설정 가이드

## 1. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
# MCP 관련
mcp>=0.4.0

# 벡터DB 및 임베딩
chromadb>=0.4.0
sentence-transformers>=2.2.0

# NLP 처리
spacy>=3.7.0
transformers>=4.30.0
torch>=2.0.0

# 한국어 모델
ko-sentence-transformers

# 기타 유틸리티
numpy>=1.24.0
regex>=2023.0.0
```

## 2. 한국어 모델 설치

```bash
# spaCy 한국어 모델 설치
python -m spacy download ko_core_news_sm

# 또는 pip으로 설치
pip install https://github.com/explosion/spacy-models/releases/download/ko_core_news_sm-3.7.0/ko_core_news_sm-3.7.0-py3-none-any.whl
```

## 3. 디렉토리 구조

```
your-project/
├── mcp_rag_server.py      # MCP 서버 메인 파일
├── requirements.txt        # 의존성 목록
├── config.json            # MCP 클라이언트 설정
├── chroma_db/             # ChromaDB 데이터 저장소 (자동 생성)
└── test_documents/        # 테스트 문서들
```

## 4. 사용 방법

### 4.1 서버 실행
```bash
python mcp_rag_server.py
```

### 4.2 Claude Desktop에서 사용
1. Claude Desktop 설정에서 config.json 파일 경로 지정
2. Claude Desktop 재시작
3. 채팅에서 MCP 도구 사용

### 4.3 사용 예시

**문서 업로드**:
```
"회의록을 업로드하고 메타데이터를 추출해줘"
```

**검색 실행**:
```
"회의록 중 고객사 담당자인 김영철씨가 언급된 회의록을 찾아주고 내용을 요약해줘"
```

**고급 검색**:
```
"김영철과 관련된 프로젝트 회의록을 찾아줘"
```

## 5. 커스터마이징

### 5.1 새로운 패턴 추가
```python
# DocumentProcessor 클래스의 patterns 딕셔너리에 추가
self.patterns = {
    # 기존 패턴들...
    "project_names": r"프로젝트[:\s]*([^\s,\n]+)",
    "budget": r"예산[:\s]*([0-9,]+(?:원|달러|만원))",
    "deadlines": r"마감[:\s]*(\d{4}[-/]\d{1,2}[-/]\d{1,2})"
}
```

### 5.2 다른 임베딩 모델 사용
```python
# 다른 한국어 모델로 변경
self.embedding_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
```

### 5.3 외부 LLM 연동 (선택사항)
```python
# OpenAI 또는 다른 LLM 서비스 연동
import openai

def extract_with_llm(self, text: str) -> Dict[str, Any]:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user", 
            "content": f"다음 텍스트에서 메타데이터 추출: {text}"
        }]
    )
    # 응답 파싱 및 반환
```

## 6. 성능 최적화

### 6.1 배치 처리 활성화
```python
# 대량 문서 처리시 배치 크기 조정
BATCH_SIZE = 100

async def process_documents_batch(self, documents: List[str]):
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        await self.process_batch(batch)
```

### 6.2 캐싱 설정
```python
# 임베딩 캐시 설정
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(self, text: str):
    return self.embedding_model.encode(text).tolist()
```

## 7. 문제 해결

### 7.1 일반적인 오류
- **spaCy 모델 없음**: `python -m spacy download ko_core_news_sm`
- **ChromaDB 권한 오류**: 디렉토리 권한 확인 또는 경로 변경
- **메모리 부족**: 배치 크기 감소 또는 경량 모델 사용

### 7.2 성능 이슈
- **검색 속도 느림**: 인덱스 크기 조정, 더 작은 임베딩 모델 사용
- **메타데이터 추출 느림**: 패턴 매칭 우선 사용, NER 선택적 적용

### 7.3 정확도 개선
- **검색 정확도 낮음**: 쿼리 확장, 다중 검색 전략 사용
- **엔티티 인식 오류**: 커스텀 패턴 추가, 도메인 특화 모델 사용
