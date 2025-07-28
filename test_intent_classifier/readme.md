-----

# 대화형 Oracle Text-to-SQL 쿼리 시스템

자연어(한국어)로 질문하면 이를 Oracle SQL 쿼리로 변환하고 실행하여 결과를 보여주는 대화형 시스템입니다. 사용자의 질문이 모호할 경우, 시스템이 능동적으로 되물어 명확한 정보를 얻어내는 피드백 루프를 통해 보다 정확한 답변을 제공하는 것을 목표로 합니다.

## ✨ 주요 기능

  - **자연어 to SQL 변환**: 한국어 질문을 Oracle 데이터베이스에서 실행 가능한 SQL 쿼리로 자동 변환합니다.
  - **의도 기반 처리**: 사용자의 입력이 단순 인사인지, 데이터 조회 요청인지, 아니면 이전 질문에 대한 추가 정보 제공인지 의도를 먼저 파악하여 적절하게 대응합니다.
  - **대화형 명확화 루프(Interactive Clarification Loop)**: "최근 매출 보여줘"와 같이 모호한 질문에 대해 "어느 기간을 원하시나요?" 와 같이 되물어, 사용자와의 대화를 통해 쿼리에 필요한 구체적인 조건을 완성합니다.
  - **모듈식 아키텍처**: 기능별로 코드를 분리하여 시스템의 확장성과 유지보수성을 높였습니다.
  - **설정 분리**: 데이터베이스 및 LLM API와 관련된 민감한 정보를 별도의 `config.py` 파일로 분리하여 관리가 용이합니다.

## 🏗️ 시스템 아키텍처 (파일 구조)

본 시스템은 각기 다른 책임을 가진 여러 Python 모듈로 구성되어 있습니다.

```
.
├── config.py               # 🅐 설정 파일 (DB, LLM API 정보)
├── oracle_connector.py     # 🅑 Oracle DB 연결 및 실행 관리
├── intent_classifier.py    # 🅒 사용자 의도 분류기
├── query_analyzer.py       # 🅓 쿼리 분석 및 명확화 질문 생성기
├── text2sql_engine.py      # 🅔 Text-to-SQL 변환 엔진
└── main.py                 # 🅕 메인 실행 파일 (대화형 루프)
```

  - **`main.py`**: 전체 모듈을 조립하고 대화형 루프를 실행하는 메인 컨트롤러입니다.
  - **`intent_classifier.py`**: 사용자의 입력 의도를 '데이터 조회', '인사', '추가 정보' 등으로 분류합니다.
  - **`query_analyzer.py`**: 데이터 조회 요청이 SQL로 변환되기에 충분히 명확한지 분석하고, 불충분할 경우 사용자에게 되물을 질문을 생성합니다.
  - **`text2sql_engine.py`**: 명확화된 사용자 요청을 최종적으로 SQL 쿼리로 변환하는 핵심 엔진입니다.
  - **`oracle_connector.py`**: Oracle 데이터베이스와의 모든 상호작용(연결, 쿼리 실행, 스키마 조회 등)을 담당합니다.
  - **`config.py`**: DB 접속 정보, LLM API 키 등 외부 설정을 관리합니다.

## ⚙️ 설치 및 설정

### 사전 요구사항

  - Python 3.8 이상
  - Oracle Instant Client: `oracledb` 라이브러리가 동작하기 위해 반드시 설치되어 있어야 합니다. [Oracle Instant Client 다운로드](https://www.oracle.com/database/technologies/instant-client/downloads.html)

### 설치 과정

1.  **프로젝트 파일 다운로드**
    모든 `.py` 파일을 동일한 디렉토리에 다운로드합니다.

2.  **필요한 라이브러리 설치**
    터미널 또는 명령 프롬프트에서 아래 명령어를 실행하여 필요한 Python 패키지를 설치합니다.

    ```bash
    pip install oracledb pandas llama-index-llms-openai-like
    ```

3.  **설정 파일 구성 (`config.py`)**
    `config.py` 파일을 열어 본인의 환경에 맞게 정보를 수정합니다.

    ```python
    # config.py

    # --- Oracle Database Configuration ---
    ORACLE_CONFIG = {
        'username': 'YOUR_ORACLE_USERNAME',  # Oracle 사용자 이름
        'password': 'YOUR_ORACLE_PASSWORD',  # Oracle 비밀번호
        'host': 'YOUR_ORACLE_HOST_IP',       # Oracle DB 호스트 주소
        'port': 1521,
        'service_name': 'YOUR_SERVICE_NAME', # Oracle 서비스 이름
    }

    # --- Large Language Model (LLM) Configuration ---
    LLM_CONFIG = {
        # 로컬 LLM(LM Studio, Ollama 등) 또는 OpenAI-like API 엔드포인트
        'openai_api_base': 'http://localhost:8000/v1',
        'openai_api_key': 'your-api-key', # API 키 (필요시)
        'model_name': 'gpt-4o',           # 사용할 LLM 모델 이름
    }
    ```

## ▶️ 실행 방법

모든 설정이 완료된 후, 터미널에서 다음 명령어를 입력하여 챗봇을 실행합니다.

```bash
python main.py
```

## 💬 사용 예시

아래는 시스템 실행 후 사용자와의 대화 예시입니다.

```
==================================================
Oracle DB 대화형 쿼리 시스템에 오신 것을 환영합니다.
('quit' 또는 'exit'를 입력하여 종료)
==================================================
You: 안녕
AI: 안녕하세요! Oracle 데이터베이스에 대해 무엇이 궁금하신가요?

You: 직원 정보 좀 보여줘
AI: 생성된 SQL 쿼리입니다.
---
SELECT * FROM (
    SELECT *
    FROM EMPLOYEES
) WHERE ROWNUM <= 1000
---
AI: 쿼리 실행 결과입니다. (10개 행)
   EMPLOYEE_ID FIRST_NAME LAST_NAME      EMAIL  PHONE_NUMBER HIRE_DATE     JOB_ID  SALARY COMMISSION_PCT MANAGER_ID DEPARTMENT_ID
0          100     Steven      King      SKING  515.123.4567  2003-06-17  AD_PRES   24000            NaN        NaN            90
1          101      Neena   Kochhar    NKOCHHAR  515.123.4568  2005-09-21   AD_VP   17000            NaN        100            90
...

You: 최근 매출 실적 좀 알려줘
AI: 어느 기간의 매출 실적을 원하시나요? (예: "지난 한 달", "올해 1분기", "2024년")

You: 지난 달
AI: 생성된 SQL 쿼리입니다.
---
SELECT
    p.PRODUCT_NAME,
    c.CATEGORY_NAME,
    SUM(oi.QUANTITY * oi.UNIT_PRICE) AS TOTAL_SALES
FROM ORDERS o
JOIN ORDER_ITEMS oi ON o.ORDER_ID = oi.ORDER_ID
JOIN PRODUCTS p ON oi.PRODUCT_ID = p.PRODUCT_ID
JOIN CATEGORIES c ON p.CATEGORY_ID = c.CATEGORY_ID
WHERE o.ORDER_DATE >= TRUNC(ADD_MONTHS(SYSDATE, -1), 'MM') AND o.ORDER_DATE < TRUNC(SYSDATE, 'MM')
GROUP BY p.PRODUCT_NAME, c.CATEGORY_NAME
ORDER BY TOTAL_SALES DESC
---
AI: 쿼리 실행 결과입니다. (5개 행)
        PRODUCT_NAME CATEGORY_NAME  TOTAL_SALES
0   Oracle Database      Software      1500000
...

You: quit
AI: 안녕히 가세요!
```
