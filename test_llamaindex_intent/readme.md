위 문서를 분석하여 LlamaIndex 기반 LLM 시스템의 정형 데이터 질의 처리 워크플로우를 파이썬으로 구현하겠습니다.문서에서 설명된 LlamaIndex 기반 정형 데이터 질의 처리 시스템을 완전히 구현했습니다. 주요 특징은 다음과 같습니다:

## 구현된 주요 구성 요소

### 1. **QueryClassifier** - 의도 분류
- LLM을 사용해 사용자 질문을 정형 데이터 질의 vs 일반 질의로 분류
- 조회 대상, 조건, 파라미터 자동 추출

### 2. **DatabaseManager** - 데이터베이스 관리
- SQLite 기반 샘플 데이터 (실적, 제품별 매출 테이블)
- LlamaIndex SQLDatabase 연동
- OracleDB로 쉽게 변경 가능한 구조

### 3. **ChartGenerator** - 차트 생성
- matplotlib/seaborn 기반 서버 측 차트 렌더링
- 선형 차트, 막대 차트, 원형 차트 지원
- 고해상도 이미지 파일로 저장

### 4. **LlamaIndexQuerySystem** - 메인 시스템
- **NLSQLTableQueryEngine**: 자연어를 SQL로 변환
- **RouterQueryEngine**: 질의 유형에 따른 엔진 자동 선택
- **ReActAgent**: 툴 호출을 통한 다단계 작업 처리

## 워크플로우 구현

1. **의도 분류**: 사용자 질문 → LLM 분석 → 정형/일반 질의 구분
2. **SQL 생성**: LlamaIndex NLSQLTableQueryEngine으로 자연어→SQL 변환
3. **데이터 조회**: 생성된 SQL로 데이터베이스 조회
4. **시각화 결정**: 질의 내용과 데이터 특성에 따라 표/차트 선택
5. **차트 생성**: 필요시 matplotlib로 차트 이미지 생성
6. **응답 구성**: LLM이 결과 해석 및 사용자 친화적 설명 생성

## Tool Calling 구현

- **execute_sql**: SQL 직접 실행 툴
- **generate_chart**: 차트 생성 툴
- ReActAgent가 상황에 따라 적절한 툴 선택적 호출

## 사용 방법

```python
# 시스템 초기화
system = LlamaIndexQuerySystem("your-openai-api-key")

# 질의 처리
result = system.process_query("우리 회사 분기별 실적을 알려줘")
formatted_result = system.format_response(result)
```

이 구현은 문서의 모든 핵심 개념을 포함하며, 실제 운영 환경에서 OpenAI API 키만 설정하면 바로 사용 가능합니다. OracleDB 연동, 추가 차트 유형, 프론트엔드 JSON 데이터 전송 등으로 확장할 수 있습니다.
