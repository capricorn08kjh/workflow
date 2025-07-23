
## 🚀 주요 기능

### 1. **지능형 질의 분류**

- `IntelligentQueryClassifier`로 질의를 자동 분류
- 정형 데이터, 문서 검색, 하이브리드, 일반 질의 구분
- LLM 기반 분류와 키워드 기반 폴백 지원

### 2. **다중 데이터 소스 지원**

- SQL 데이터베이스 (정형 데이터)
- ChromaDB 벡터 데이터베이스 (문서)
- 하이브리드 질의로 두 소스 동시 활용

### 3. **스마트 시각화**

- `SmartVisualizationDecider`로 질의 의도에 맞는 차트 자동 선택
- 라인차트, 바차트, 파이차트, 산점도 지원
- 질의 키워드와 데이터 특성 기반 결정

### 4. **ReAct 에이전트**

- 복잡한 다단계 질의 처리
- 도구들을 조합하여 종합적 분석 제공
- 자동화된 추론과 행동 수행

### 5. **OpenAI-like API 지원**

- 다양한 LLM 사용 가능 (로컬 LLM 포함)
- `llama_index.llms.openai_like.OpenAILike` 사용

## 📝 지원하는 질의 예시

### 정형 데이터 질의

- “공정별 실적을 찾아줘”
- “최근 4분기 매출 추이를 보여줘”
- “제품별 매출 비교 분석해줘”

### 문서 검색 질의

- “회의록 중 김영철씨가 언급된 것을 찾아줘”
- “최근 보고서에서 품질 개선 관련 내용을 요약해줘”

### 하이브리드 질의

- “4월 월간보고서와 결산 실적을 같이 보여줘”
- “공정 효율성 데이터와 관련 회의록 내용을 비교해줘”

## 🔧 사용 방법

```python
# 시스템 초기화
system = ImprovedQuerySystem(
    openai_api_base="http://localhost:1234/v1",  # Local LLM API
    openai_api_key="fake-key",
    model_name="gpt-3.5-turbo"
)

# 질의 처리
result = system.process_query("공정별 실적을 차트로 보여줘")
formatted_result = system.format_response(result)
```
