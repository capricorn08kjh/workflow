import cx_Oracle
import pandas as pd
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

# LlamaIndex imports

from llama_index.llms.openai_like import OpenAILike
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# 로깅 설정

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

@dataclass
class QueryResult:
“”“SQL 쿼리 결과 데이터 클래스”””
success: bool
data: Optional[pd.DataFrame] = None
sql_query: Optional[str] = None
error_message: Optional[str] = None
execution_time: Optional[float] = None
row_count: Optional[int] = None

class OracleConnectionManager:
“”“Oracle 데이터베이스 연결 관리 클래스”””

```
def __init__(self, 
             username: str,
             password: str, 
             dsn: str,
             encoding: str = "UTF-8"):
    """
    Oracle DB 연결 초기화
    
    Args:
        username: 사용자명
        password: 비밀번호
        dsn: 데이터 소스명 (예: localhost:1521/XE)
        encoding: 인코딩
    """
    self.username = username
    self.password = password
    self.dsn = dsn
    self.encoding = encoding
    
    # SQLAlchemy 엔진 생성
    self.connection_string = f"oracle+cx_oracle://{username}:{password}@{dsn}"
    self.engine = create_engine(
        self.connection_string,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    
    # 연결 테스트
    self.test_connection()

def test_connection(self) -> bool:
    """데이터베이스 연결 테스트"""
    try:
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT 1 FROM DUAL"))
            result.fetchone()
        logger.info("Oracle 데이터베이스 연결 성공")
        return True
    except Exception as e:
        logger.error(f"Oracle 데이터베이스 연결 실패: {str(e)}")
        raise

def get_engine(self) -> Engine:
    """SQLAlchemy 엔진 반환"""
    return self.engine

def get_table_schema(self, schema_name: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """테이블 스키마 정보 조회"""
    try:
        schema_filter = ""
        if schema_name:
            schema_filter = f"AND OWNER = '{schema_name.upper()}'"
        
        schema_query = f"""
        SELECT 
            OWNER,
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            DATA_LENGTH,
            NULLABLE,
            COLUMN_ID
        FROM ALL_TAB_COLUMNS 
        WHERE OWNER NOT IN ('SYS', 'SYSTEM', 'DBSNMP', 'SYSMAN', 'OUTLN', 'MDSYS', 'ORDSYS', 'EXFSYS', 'DMSYS', 'WMSYS', 'CTXSYS', 'ANONYMOUS', 'XDB', 'ORDPLUGINS', 'OLAPSYS')
        {schema_filter}
        ORDER BY OWNER, TABLE_NAME, COLUMN_ID
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(schema_query))
            rows = result.fetchall()
        
        # 스키마 정보 구조화
        schema_info = {}
        for row in rows:
            owner, table_name, column_name, data_type, data_length, nullable, column_id = row
            table_key = f"{owner}.{table_name}"
            
            if table_key not in schema_info:
                schema_info[table_key] = {
                    'columns': {},
                    'owner': owner,
                    'table_name': table_name
                }
            
            schema_info[table_key]['columns'][column_name] = {
                'data_type': data_type,
                'data_length': data_length,
                'nullable': nullable == 'Y',
                'column_id': column_id
            }
        
        logger.info(f"스키마 정보 조회 완료: {len(schema_info)}개 테이블")
        return schema_info
        
    except Exception as e:
        logger.error(f"스키마 정보 조회 실패: {str(e)}")
        return {}

def execute_query(self, sql_query: str) -> QueryResult:
    """SQL 쿼리 실행"""
    start_time = datetime.now()
    
    try:
        # SQL 쿼리 정리 및 검증
        cleaned_query = self._clean_sql_query(sql_query)
        
        # 안전성 검사
        if not self._is_safe_query(cleaned_query):
            return QueryResult(
                success=False,
                error_message="안전하지 않은 쿼리입니다. SELECT 문만 허용됩니다."
            )
        
        # 쿼리 실행
        with self.engine.connect() as conn:
            df = pd.read_sql(cleaned_query, conn)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"쿼리 실행 성공: {len(df)}개 행, {execution_time:.2f}초")
        
        return QueryResult(
            success=True,
            data=df,
            sql_query=cleaned_query,
            execution_time=execution_time,
            row_count=len(df)
        )
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        logger.error(f"쿼리 실행 실패: {error_msg}")
        
        return QueryResult(
            success=False,
            sql_query=cleaned_query if 'cleaned_query' in locals() else sql_query,
            error_message=error_msg,
            execution_time=execution_time
        )

def _clean_sql_query(self, sql_query: str) -> str:
    """SQL 쿼리 정리"""
    # 불필요한 공백 제거
    cleaned = re.sub(r'\s+', ' ', sql_query.strip())
    
    # 세미콜론 제거 (마지막에 있는 경우)
    if cleaned.endswith(';'):
        cleaned = cleaned[:-1]
    
    return cleaned

def _is_safe_query(self, sql_query: str) -> bool:
    """쿼리 안전성 검사"""
    query_upper = sql_query.upper().strip()
    
    # SELECT 문만 허용
    if not query_upper.startswith('SELECT'):
        return False
    
    # 위험한 키워드 차단
    dangerous_keywords = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
        'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE'
    ]
    
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return False
    
    return True
```

class Text2SQLConverter:
“”“자연어를 SQL로 변환하는 클래스”””

```
def __init__(self, 
             oracle_manager: OracleConnectionManager,
             openai_api_base: str,
             openai_api_key: str = "fake-key",
             model_name: str = "gpt-3.5-turbo"):
    """
    Text2SQL 변환기 초기화
    
    Args:
        oracle_manager: Oracle 연결 관리자
        openai_api_base: OpenAI-like API 베이스 URL
        openai_api_key: API 키
        model_name: 모델명
    """
    self.oracle_manager = oracle_manager
    
    # OpenAI-like LLM 설정
    self.llm = OpenAILike(
        model=model_name,
        api_base=openai_api_base,
        api_key=openai_api_key,
        temperature=0.1,
        max_tokens=1000
    )
    
    # 스키마 정보 로드
    self.schema_info = oracle_manager.get_table_schema()
    
    # LlamaIndex SQL 데이터베이스 설정
    self.sql_database = SQLDatabase(oracle_manager.get_engine())
    
    # NL2SQL 쿼리 엔진 설정
    self.setup_query_engine()

def setup_query_engine(self):
    """LlamaIndex NL2SQL 쿼리 엔진 설정"""
    try:
        # 사용 가능한 테이블 목록 생성
        table_names = [info['table_name'] for info in self.schema_info.values()]
        
        self.nl2sql_engine = NLSQLTableQueryEngine(
            sql_database=self.sql_database,
            tables=table_names[:10] if len(table_names) > 10 else table_names,  # 성능을 위해 최대 10개 테이블
            llm=self.llm,
            synthesize_response=False,  # SQL만 생성하고 응답은 별도 처리
            verbose=True
        )
        
        logger.info("NL2SQL 쿼리 엔진 설정 완료")
        
    except Exception as e:
        logger.error(f"쿼리 엔진 설정 실패: {str(e)}")
        self.nl2sql_engine = None

def convert_natural_language_to_sql(self, 
                                  natural_query: str,
                                  schema_hint: Optional[str] = None) -> str:
    """자연어를 SQL로 변환"""
    try:
        # 스키마 정보를 텍스트로 변환
        schema_context = self._build_schema_context(schema_hint)
        
        # 프롬프트 구성
        prompt = self._build_conversion_prompt(natural_query, schema_context)
        
        # LLM을 통한 SQL 생성
        response = self.llm.complete(prompt)
        sql_query = self._extract_sql_from_response(response.text)
        
        logger.info(f"자연어 → SQL 변환 완료")
        return sql_query
        
    except Exception as e:
        logger.error(f"SQL 변환 실패: {str(e)}")
        raise

def _build_schema_context(self, schema_hint: Optional[str] = None) -> str:
    """스키마 컨텍스트 구성"""
    if not self.schema_info:
        return "스키마 정보를 사용할 수 없습니다."
    
    # 관련 테이블만 선택 (성능 최적화)
    relevant_tables = self.schema_info
    if schema_hint:
        relevant_tables = {
            k: v for k, v in self.schema_info.items() 
            if schema_hint.lower() in k.lower()
        }
    
    # 스키마 정보를 텍스트로 변환
    schema_text = "=== 데이터베이스 스키마 정보 ===\n"
    
    for table_key, table_info in list(relevant_tables.items())[:5]:  # 최대 5개 테이블
        schema_text += f"\n테이블: {table_key}\n"
        schema_text += "컬럼:\n"
        
        for col_name, col_info in table_info['columns'].items():
            nullable = "NULL 가능" if col_info['nullable'] else "NOT NULL"
            schema_text += f"  - {col_name}: {col_info['data_type']} ({nullable})\n"
    
    return schema_text

def _build_conversion_prompt(self, natural_query: str, schema_context: str) -> str:
    """SQL 변환용 프롬프트 구성"""
    prompt = f"""
```

당신은 Oracle 데이터베이스 전문가입니다. 자연어 질의를 Oracle SQL로 변환해주세요.

{schema_context}

=== 변환 규칙 ===

1. Oracle SQL 문법을 정확히 사용하세요
1. 테이블명과 컬럼명은 대소문자를 정확히 맞춰주세요
1. 한국어 조건은 LIKE 연산자와 와일드카드(%)를 적절히 사용하세요
1. 날짜 조건은 TO_DATE 함수를 사용하세요
1. SELECT 문만 생성하세요 (INSERT, UPDATE, DELETE 금지)
1. 가능한 경우 적절한 ORDER BY를 추가하세요

=== 자연어 질의 ===
{natural_query}

=== Oracle SQL (코드 블록 없이 SQL만 반환) ===
“””
return prompt

```
def _extract_sql_from_response(self, response_text: str) -> str:
    """LLM 응답에서 SQL 추출"""
    # 코드 블록 제거
    if "```" in response_text:
        # ```sql ... ``` 형태에서 SQL 추출
        pattern = r'```(?:sql)?\s*(.*?)\s*```'
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 코드 블록이 없는 경우 전체 텍스트 사용
    return response_text.strip()

def process_natural_query(self, 
                        natural_query: str,
                        schema_hint: Optional[str] = None,
                        execute: bool = True) -> Dict[str, Any]:
    """자연어 질의 전체 처리 파이프라인"""
    result = {
        'natural_query': natural_query,
        'sql_query': None,
        'success': False,
        'data': None,
        'error_message': None,
        'execution_time': None,
        'row_count': None
    }
    
    try:
        # 1. 자연어 → SQL 변환
        logger.info(f"자연어 질의 처리 시작: {natural_query}")
        sql_query = self.convert_natural_language_to_sql(natural_query, schema_hint)
        result['sql_query'] = sql_query
        
        if not execute:
            result['success'] = True
            return result
        
        # 2. SQL 실행
        query_result = self.oracle_manager.execute_query(sql_query)
        
        # 3. 결과 통합
        result.update({
            'success': query_result.success,
            'data': query_result.data,
            'error_message': query_result.error_message,
            'execution_time': query_result.execution_time,
            'row_count': query_result.row_count
        })
        
        if query_result.success:
            logger.info(f"질의 처리 완료: {query_result.row_count}개 행 반환")
        else:
            logger.error(f"질의 처리 실패: {query_result.error_message}")
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"질의 처리 중 오류: {error_msg}")
        result['error_message'] = error_msg
        return result

def get_schema_summary(self) -> Dict[str, Any]:
    """스키마 요약 정보 반환"""
    if not self.schema_info:
        return {'error': '스키마 정보를 사용할 수 없습니다'}
    
    summary = {
        'total_tables': len(self.schema_info),
        'tables': []
    }
    
    for table_key, table_info in self.schema_info.items():
        table_summary = {
            'full_name': table_key,
            'owner': table_info['owner'],
            'table_name': table_info['table_name'],
            'column_count': len(table_info['columns']),
            'columns': list(table_info['columns'].keys())
        }
        summary['tables'].append(table_summary)
    
    return summary
```

class Text2SQLOracle:
“”“Oracle Text2SQL 메인 클래스”””

```
def __init__(self,
             username: str,
             password: str,
             dsn: str,
             openai_api_base: str,
             openai_api_key: str = "fake-key",
             model_name: str = "gpt-3.5-turbo"):
    """
    초기화
    
    Args:
        username: Oracle 사용자명
        password: Oracle 비밀번호  
        dsn: 데이터 소스명
        openai_api_base: OpenAI-like API URL
        openai_api_key: API 키
        model_name: 모델명
    """
    
    # Oracle 연결 설정
    self.oracle_manager = OracleConnectionManager(username, password, dsn)
    
    # Text2SQL 변환기 설정
    self.converter = Text2SQLConverter(
        self.oracle_manager,
        openai_api_base,
        openai_api_key,
        model_name
    )
    
    logger.info("Text2SQL Oracle 시스템 초기화 완료")

def query(self, natural_language: str, **kwargs) -> Dict[str, Any]:
    """자연어로 데이터베이스 조회"""
    return self.converter.process_natural_query(natural_language, **kwargs)

def get_sql_only(self, natural_language: str, **kwargs) -> str:
    """SQL만 생성 (실행하지 않음)"""
    result = self.converter.process_natural_query(
        natural_language, 
        execute=False, 
        **kwargs
    )
    return result.get('sql_query', '')

def get_schema_info(self) -> Dict[str, Any]:
    """데이터베이스 스키마 정보 조회"""
    return self.converter.get_schema_summary()

def test_connection(self) -> bool:
    """연결 테스트"""
    return self.oracle_manager.test_connection()
```

def main():
“”“사용 예시”””

```
# 설정 (실제 사용시 환경변수나 설정 파일 사용 권장)
config = {
    'username': 'your_username',
    'password': 'your_password', 
    'dsn': 'localhost:1521/XE',
    'openai_api_base': 'http://localhost:8000/v1',  # 예시 로컬 API
    'openai_api_key': 'fake-key',
    'model_name': 'gpt-3.5-turbo'
}

try:
    # Text2SQL 시스템 초기화
    print("Text2SQL Oracle 시스템 초기화 중...")
    text2sql = Text2SQLOracle(**config)
    
    # 스키마 정보 조회
    print("\n=== 데이터베이스 스키마 정보 ===")
    schema_info = text2sql.get_schema_info()
    print(f"총 테이블 수: {schema_info.get('total_tables', 0)}개")
    
    # 샘플 질의들
    sample_queries = [
        "모든 직원의 이름과 부서를 알려주세요",
        "급여가 5000 이상인 직원들을 찾아주세요",
        "부서별 평균 급여를 계산해주세요",
        "최근 3개월간 주문 현황을 보여주세요",
        "제품별 매출 순위를 알려주세요"
    ]
    
    print("\n=== 샘플 질의 테스트 ===")
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{i}. 자연어 질의: {query}")
        
        # SQL만 생성
        sql = text2sql.get_sql_only(query)
        print(f"   생성된 SQL: {sql}")
        
        # 실제 실행 (주석 해제하여 사용)
        # result = text2sql.query(query)
        # if result['success']:
        #     print(f"   결과: {result['row_count']}개 행")
        #     print(result['data'].head())
        # else:
        #     print(f"   오류: {result['error_message']}")
    
    print("\n실제 사용을 위해서는 Oracle DB 연결 정보와 OpenAI-like API 설정이 필요합니다.")
    
except Exception as e:
    print(f"초기화 실패: {str(e)}")
    print("Oracle 클라이언트와 cx_Oracle 패키지가 설치되었는지 확인하세요.")
```

if **name** == “**main**”:
main()