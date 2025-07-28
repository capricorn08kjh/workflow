# Text2SQL 시스템

Oracle 데이터베이스와 연동하여 자연어 질의를 SQL로 변환하고 실행하는 시스템

## 🌟 주요 기능

### 1. 지능적 의도 분석
- **다양한 질의 유형 지원**: 데이터 조회, 분석, 스키마 문의, 일반 질문 등
- **신뢰도 기반 판단**: 질의 의도를 분석하고 신뢰도를 측정
- **SQL 필요성 자동 판단**: Text2SQL 변환이 필요한지 자동으로 결정

### 2. 대화형 명확화 시스템
- **모호한 질의 처리**: 불명확한 표현을 구체적으로 명확화
- **단계별 질문**: 테이블 선택, 조건 설정, 시간 범위 등 체계적 질문
- **컨텍스트 유지**: 대화 맥락을 기억하며 연속적인 질의 처리

### 3. 강화된 SQL 생성
- **조인 지원**: 여러 테이블 간의 복잡한 관계 분석 및 조인 쿼리 생성
- **관계 분석**: 외래키 제약조건과 공통 컬럼 기반 테이블 관계 추론
- **성능 최적화**: ROWNUM 제한, 인덱스 활용 등 Oracle 최적화 기법 적용

### 4. 사용자 친화적 인터페이스
- **대화형 콘솔**: 터미널 기반 실시간 대화 인터페이스
- **웹 API**: REST API를 통한 웹 애플리케이션 연동
- **상태 관리**: 세션별 대화 상태와 이력 관리

## 📋 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    ConversationHandler                      │
│                   (대화 흐름 관리)                        │
└─────────────────┬─────────────────┬─────────────────────────┘
                  │                 │
                  ▼                 ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│    QueryIntentAnalyzer      │ │   ClarificationManager      │
│     (의도 분석)              │ │     (명확화 관리)            │
└─────────────────┬───────────┘ └─────────────────┬───────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │       Text2SQLOracle        │
                    │    (SQL 생성 및 실행)       │
                    └─────────────────────────────┘
```

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone <repository-url>
cd enhanced-text2sql

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 설정

`config_example.json`을 참고하여 설정 파일을 작성하세요:

```json
{
  "oracle_username": "your_username",
  "oracle_password": "your_password",
  "oracle_host": "localhost",
  "oracle_port": 1521,
  "oracle_service_name": "SMIP_DEV",
  
  "openai_api_base": "http://localhost:8000/v1",
  "openai_api_key": "your-api-key",
  "model_name": "gpt-3.5-turbo"
}
```

### 3. 실행

#### 대화형 콘솔
```bash
python enhanced_text2sql_main.py
```

#### 데모 실행
```bash
python demo_usage.py
```

## 💡 사용 예시

### 기본 질의
```
사용자: "모든 사용자 정보를 보여주세요"
시스템: ✅ 쿼리가 성공적으로 실행되었습니다!

생성된 SQL:
SELECT * FROM USERS WHERE ROWNUM <= 100

결과: 85개 행이 조회되었습니다.
```

### 명확화 과정
```
사용자: "최근 매출이 높은 고객을 찾아주세요"
시스템: 질문을 더 정확히 이해하기 위해 몇 가지 질문을 드리겠습니다.

❓ 어떤 테이블의 데이터를 조회하고 싶으신가요?
선택 옵션:
  1. CUSTOMERS (고객 정보)
  2. SALES (매출 정보)
  3. ORDERS (주문 정보)

사용자: "2"
시스템: ❓ '최근'은 구체적으로 언제부터를 말씀하시는 건가요?

사용자: "최근 30일"
시스템: ❓ '높은 매출'의 구체적인 기준을 알려주시겠어요?

사용자: "100만원 이상"
시스템: ✅ 명확화가 완료되었습니다!

정제된 질의: SALES 테이블에서 최근 30일간 100만원 이상의 매출을 가진 고객을 조회해주세요
```

### 복잡한 분석
```
사용자: "부서별 직원 수와 평균 급여를 함께 보여주세요"
시스템: ✅ 쿼리가 성공적으로 실행되었습니다!

생성된 SQL:
SELECT d.dept_name, COUNT(e.emp_id) as employee_count, 
       ROUND(AVG(e.salary), 2) as avg_salary
FROM employees e 
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY d.dept_name, d.dept_id
ORDER BY employee_count DESC

결과: 8개 행이 조회되었습니다.
```

## 📁 파일 구조

```
enhanced-text2sql/
├── text2sql_oracle.py          # 기존 Text2SQL 시스템
├── query_intent_analyzer.py    # 의도 분석 모듈
├── clarification_manager.py    # 명확화 관리 모듈
├── conversation_handler.py     # 대화 처리 모듈
├── enhanced_text2sql_main.py   # 메인 시스템
├── demo_usage.py              # 사용 예시 및 데모
├── config_example.json        # 설정 파일 예시
├── requirements.txt           # 패키지 의존성
└── README.md                  # 이 파일
```

## 🎯 주요 모듈 설명

### QueryIntentAnalyzer
- **역할**: 자연어 질의의 의도를 분석하고 분류
- **기능**: 
  - 의도 분류 (데이터 조회, 분석, 스키마 문의 등)
  - 복잡도 분석 (단순, 중간, 복잡)
  - 엔터티 추출 (테이블명, 조건, 시간 등)
  - SQL 필요성 판단

### ClarificationManager
- **역할**: 모호한 질의를 명확화하는 대화 과정 관리
- **기능**:
  - 질문 생성 (테이블 선택, 조건 설정 등)
  - 답변 검증 및 저장
  - 정제된 질의 생성
  - 세션 상태 관리

### ConversationHandler
- **역할**: 전체적인 대화 흐름을 조정하고 관리
- **기능**:
  - 사용자 메시지 라우팅
  - 상태 기반 처리 로직
  - 세션 생명주기 관리
  - 결과 포맷팅

## 🔧 고급 기능

### 1. 테이블 관계 분석
시스템은 다음 방법으로 테이블 간 관계를 자동으로 분석합니다:
- **외래키 제약조건**: Oracle 시스템 카탈로그에서 FK 관계 조회
- **공통 컬럼 매칭**: 컬럼명 패턴 분석 (ID, CODE, NO 등)
- **조인 조건 생성**: 관계를 바탕으로 최적의 조인 조건 자동 생성

### 2. 복잡도 기반 처리
- **단순**: 단일 테이블 조회
- **중간**: 집계 함수, 조건 필터링, 정렬
- **복잡**: 다중 테이블 조인, 서브쿼리, 윈도우 함수

### 3. 성능 최적화
- **행 수 제한**: 자동으로 ROWNUM 조건 추가
- **인덱스 힌트**: 적절한 Oracle 힌트 사용
- **쿼리 계획**: 효율적인 실행 계획 유도

## 🌐 웹 API 사용

### FastAPI 연동 예시

```python
from fastapi import FastAPI
from enhanced_text2sql_main import WebAPIInterface

app = FastAPI()
text2sql_api = WebAPIInterface(config)

@app.post("/api/session")
async def create_session():
    return text2sql_api.create_session()

@app.post("/api/query")
async def process_query(request: dict):
    return text2sql_api.process_query(
        request.get("session_id"), 
        request.get("query")
    )

@app.get("/api/tables")
async def get_tables():
    return text2sql_api.get_available_tables()
```

### API 엔드포인트

| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| POST | `/api/session` | 새 세션 생성 |
| POST | `/api/query` | 질의 처리 |
| GET | `/api/session/{id}` | 세션 정보 조회 |
| DELETE | `/api/session/{id}` | 세션 종료 |
| GET | `/api/tables` | 테이블 목록 |
| GET | `/api/health` | 시스템 상태 |

## 🛠️ 설정 옵션

### Oracle 데이터베이스
```json
{
  "oracle_username": "사용자명",
  "oracle_password": "비밀번호",
  "oracle_host": "호스트 주소",
  "oracle_port": 1521,
  "oracle_service_name": "서비스명"
}
```

### LLM 설정
```json
{
  "openai_api_base": "API 베이스 URL",
  "openai_api_key": "API 키",
  "model_name": "모델명",
  "temperature": 0.1,
  "max_tokens": 1500
}
```

### 시스템 설정
```json
{
  "system": {
    "max_rows_default": 100,
    "session_timeout_minutes": 30,
    "max_active_sessions": 100
  }
}
```

## 🧪 테스트

### 단위 테스트 실행
```bash
pytest tests/ -v
```

### 통합 테스트
```bash
python demo_usage.py
```

### 성능 테스트
```bash
python -c "from demo_usage import demo_performance_test; demo_performance_test()"
```

## 📊 모니터링

### 시스템 통계 확인
```python
system = EnhancedText2SQLSystem(config)
stats = system.get_system_stats()
print(f"활성 세션: {stats['active_sessions']}")
print(f"DB 연결 상태: {stats['database_connected']}")
```

### 로그 설정
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 🚨 문제 해결

### 일반적인 문제들

1. **Oracle 연결 실패**
   ```
   해결방법: 
   - Oracle 클라이언트 설치 확인
   - 연결 정보 (호스트, 포트, 서비스명) 확인
   - 방화벽 설정 확인
   ```

2. **LLM API 오류**
   ```
   해결방법:
   - API 서버 실행 상태 확인
   - API 키 유효성 확인
   - 네트워크 연결 확인
   ```

3. **의도 분석 정확도 낮음**
   ```
   해결방법:
   - 더 구체적인 질의 사용
   - 테이블명이나 컬럼명 포함
   - 명확화 과정 적극 활용
   ```

### 로그 확인
```bash
# 시스템 로그
tail -f system.log

# 오류 로그만 필터링
grep ERROR system.log
```

## 🤝 기여 방법

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 지원

- **이슈 리포트**: GitHub Issues를 통해 버그나 기능 요청
- **문서**: 추가 문서는 `docs/` 디렉토리 참조
- **예시**: `demo_usage.py`에서 다양한 사용 예시 확인

## 🔄 업데이트 로그

### v2.0.0 - 향상된 Text2SQL
- ✨ 의도 분석 시스템 추가
- ✨ 대화형 명확화 과정 구현
- ✨ 다중 테이블 조인 지원
- ✨ 웹 API 인터페이스 제공
- 🔧 성능 최적화 및 안정성 개선

### v1.0.0 - 기본 Text2SQL
- 📦 Oracle 연동 기본 기능
- 📦 단순 자연어-SQL 변환
