# process_query 메서드 상세 분석

## 🔄 전체 처리 파이프라인

`process_query` 메서드는 사용자의 자연어 질의를 받아서 적절한 답변과 시각화를 제공하는 6단계 파이프라인입니다.

## 📋 단계별 상세 분석

### 1단계: 질의 분류 및 파라미터 추출

```python
def process_query(self, user_query: str) -> QueryResult:
    logger.info(f"질의 처리 시작: {user_query}")
    
    try:
        # 1. 질의 분류 및 파라미터 추출
        table_info = self.db_manager.get_table_info()
        query_type, parameters = self.query_classifier.classify_query(user_query, table_info)
```

**목적**: 사용자 질의의 **의도**와 **유형**을 파악합니다.

**동작 과정**:
- `IntelligentQueryClassifier`가 LLM을 사용하여 질의를 분석
- 데이터베이스 테이블 정보를 참조하여 더 정확한 분류
- 4가지 유형으로 분류:
  - `STRUCTURED_DATA`: SQL 데이터 조회 필요
  - `DOCUMENT_SEARCH`: 문서 검색 필요  
  - `HYBRID`: 정형+비정형 데이터 모두 필요
  - `GENERAL_QUERY`: 일반적인 대화

**예시**:
```python
# 입력: "공정별 실적을 찾아줘"
# 결과: QueryType.STRUCTURED_DATA, {"tables": ["process_performance"], "analysis_type": "performance"}

# 입력: "회의록 중 김영철씨가 언급된 것을 찾아줘" 
# 결과: QueryType.DOCUMENT_SEARCH, {"person_names": ["김영철"], "document_types": ["meeting_minutes"]}

# 입력: "4월 월간보고서와 결산 실적을 같이 보여줘"
# 결과: QueryType.HYBRID, {"document_types": ["monthly_report"], "tables": ["quarterly_performance"]}
```

### 2단계: 일반 질의 처리 (해당하는 경우)

```python
# 2. 질의 유형별 처리
if query_type == QueryType.GENERAL_QUERY:
    # 일반 질의는 LLM으로 직접 처리
    response = self.llm.complete(user_query)
    return QueryResult(
        answer=response.text,
        query_type=query_type,
        visualization_type=VisualizationType.TABLE,
        metadata={"parameters": parameters}
    )
```

**목적**: 데이터 조회가 필요 없는 일반적인 질문을 간단히 처리합니다.

**처리 대상**:
- "안녕하세요"
- "회사 현황을 전반적으로 설명해주세요"
- "어떤 질문을 할 수 있나요?"

### 3단계: ReAct 에이전트를 통한 복합 질의 처리

```python
# 3. 에이전트를 통한 복합 질의 처리
agent_response = self.agent.chat(user_query)
```

**목적**: 복잡한 질의를 **자동화된 추론과 행동**으로 처리합니다.

**ReAct 에이전트의 동작**:
1. **Reasoning**: 질의를 분석하고 필요한 도구를 결정
2. **Acting**: 적절한 도구(SQL 조회, 문서 검색, 시각화)를 실행
3. **Observing**: 도구 실행 결과를 관찰
4. 필요시 2-3단계를 반복

**사용 가능한 도구들**:
- `analyze_structured_data`: SQL 데이터베이스 조회
- `search_documents`: 문서 검색
- `create_visualization`: 차트 생성
- `hybrid_analysis`: 정형+비정형 데이터 통합 분석

**에이전트 처리 예시**:
```
사용자: "공정별 실적을 차트로 보여줘"

에이전트 사고 과정:
1. Thought: 공정별 실적 데이터를 조회해야 하고, 차트로 시각화해야 한다
2. Action: analyze_structured_data("SELECT * FROM process_performance")
3. Observation: 데이터 조회 완료
4. Thought: 이제 이 데이터를 바차트로 시각화하자
5. Action: create_visualization(data, "bar_chart", "공정별 실적")
6. Observation: 차트 생성 완료
7. 최종 답변 생성
```

### 4단계: 응답에서 데이터 추출

```python
# 4. 응답에서 데이터 추출 시도
structured_data = self._extract_structured_data_from_response(str(agent_response))
document_results = self._extract_document_results_from_response(str(agent_response))
```

**목적**: 에이전트 응답에서 **구조화된 데이터**와 **문서 결과**를 추출합니다.

**추출 과정**:
- 정규표현식과 JSON 파싱을 사용
- 에이전트가 도구를 사용한 결과를 파싱
- DataFrame 형태의 정형 데이터 추출
- 문서 검색 결과 추출

**추출 예시**:
```python
# 에이전트 응답에서 이런 형태의 데이터를 찾아서 추출
{
    "data_summary": {
        "row_count": 5,
        "column_count": 3,
        "columns": ["process_name", "efficiency", "output"]
    },
    "data": [
        {"process_name": "A공정", "efficiency": 92, "output": 1500},
        {"process_name": "B공정", "efficiency": 78, "output": 1200}
    ]
}
```

### 5단계: 시각화 결정 및 생성

```python
# 5. 시각화 결정 및 생성
viz_type = VisualizationType.TABLE
chart_path = None

if structured_data is not None and not structured_data.empty:
    viz_type, viz_params = self.viz_decider.decide_visualization(
        structured_data, user_query, parameters
    )
    
    if viz_type != VisualizationType.TABLE:
        chart_path = self.chart_generator.generate_chart(
            structured_data, 
            viz_type, 
            f"{user_query} - 분석 결과",
            viz_params.get('x_column'),
            viz_params.get('y_column')
        )
```

**목적**: **스마트한 시각화 결정**과 차트 생성을 수행합니다.

**SmartVisualizationDecider의 판단 기준**:

| 질의 키워드 | 시각화 타입 | 예시 |
|------------|------------|------|
| "추이", "변화", "트렌드" | LINE_CHART | "매출 추이를 보여줘" |
| "비율", "구성", "점유율" | PIE_CHART | "제품별 점유율을 보여줘" |
| "비교", "대비", "차이" | BAR_CHART | "공정별 실적을 비교해줘" |
| "상관관계", "관계" | SCATTER_PLOT | "효율성과 매출의 관계" |

**데이터 특성 기반 결정**:
- 시계열 데이터 → LINE_CHART
- 카테고리 데이터 (20개 이하) → BAR_CHART
- 기본값 → TABLE

### 6단계: 최종 결과 구성

```python
# 6. 최종 결과 구성
return QueryResult(
    answer=str(agent_response),
    query_type=query_type,
    visualization_type=viz_type,
    structured_data=structured_data,
    document_results=document_results,
    chart_path=chart_path,
    metadata={
        "parameters": parameters,
        "processing_steps": ["classification", "agent_processing", "visualization"]
    }
)
```

**목적**: 모든 처리 결과를 **구조화된 형태**로 반환합니다.

**QueryResult 구성 요소**:
- `answer`: 자연어 답변
- `query_type`: 질의 유형
- `visualization_type`: 사용된 시각화 타입
- `structured_data`: 조회된 DataFrame
- `document_results`: 문서 검색 결과
- `chart_path`: 생성된 차트 파일 경로
- `metadata`: 처리 과정 메타데이터

## 🔧 오류 처리

```python
except Exception as e:
    logger.error(f"질의 처리 중 오류 발생: {str(e)}")
    return QueryResult(
        answer=f"질의 처리 중 오류가 발생했습니다: {str(e)}",
        query_type=QueryType.UNKNOWN,
        visualization_type=VisualizationType.TABLE,
        metadata={"error": str(e)}
    )
```

**강건한 오류 처리**로 시스템이 중단되지 않고 사용자에게 유용한 오류 정보를 제공합니다.

## 📊 실제 처리 예시

### 예시 1: 정형 데이터 질의
```
입력: "공정별 실적을 차트로 보여줘"

1. 분류: STRUCTURED_DATA
2. 에이전트 처리:
   - SQL 조회: "SELECT * FROM process_performance"
   - 데이터 추출: DataFrame(5행 3열)
3. 시각화 결정: BAR_CHART (비교 키워드 감지)
4. 차트 생성: bar_chart_12345.png
5. 결과 반환: 답변 + 데이터 + 차트
```

### 예시 2: 문서 검색 질의
```
입력: "회의록 중 김영철씨가 언급된 것을 찾아줘"

1. 분류: DOCUMENT_SEARCH
2. 에이전트 처리:
   - 문서 검색: "김영철" + "회의록"
   - 3개 관련 문서 발견
3. 시각화: TABLE (문서 결과)
4. 결과 반환: 답변 + 문서 목록
```

### 예시 3: 하이브리드 질의
```
입력: "4월 월간보고서와 결산 실적을 같이 보여줘"

1. 분류: HYBRID
2. 에이전트 처리:
   - 문서 검색: "4월 월간보고서"
   - SQL 조회: "SELECT * FROM quarterly_performance WHERE quarter='2024Q1'"
   - 결과 통합
3. 시각화: 데이터에 따라 결정
4. 결과 반환: 통합 답변 + 데이터 + 문서
```

## ⚡ 핵심 장점

1. **지능형 분류**: LLM을 활용한 정확한 의도 파악
2. **자동화된 처리**: ReAct 에이전트의 자율적 추론과 행동
3. **스마트 시각화**: 질의 의도에 맞는 자동 차트 생성
4. **통합 처리**: 정형+비정형 데이터의 seamless한 통합
5. **강건성**: 포괄적인 오류 처리와 로깅
6. **확장성**: 새로운 도구와 데이터 소스 쉽게 추가 가능

이러한 아키텍처를 통해 사용자는 자연어로 복잡한 질의를 던지기만 하면, 시스템이 자동으로 최적의 처리 경로를 선택하고 적절한 답변과 시각화를 제공합니다.