import oracledb
import pandas as pd
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

# LlamaIndex imports

from llama_index.llms.openai_like import OpenAILike

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
“”“Oracle 데이터베이스 연결 관리 클래스 (oracledb 사용)”””

```
def __init__(self, 
             username: str,
             password: str, 
             host: str,
             port: int = 1521,
             service_name: str = "SMIP_DEV"):
    """
    Oracle DB 연결 초기화
    
    Args:
        username: 사용자명
        password: 비밀번호
        host: 호스트 주소
        port: 포트 번호 (기본값: 1521)
        service_name: 서비스명 (기본값: SMIP_DEV)
    """
    self.username = username
    self.password = password
    self.host = host
    self.port = port
    self.service_name = service_name
    
    # DSN 구성
    self.dsn = oracledb.makedsn(host, port, service_name=service_name)
    
    # 연결 풀 설정
    self.pool = None
    self.setup_connection_pool()
    
    # 연결 테스트
    self.test_connection()

def setup_connection_pool(self):
    """연결 풀 설정"""
    try:
        self.pool = oracledb.create_pool(
            user=self.username,
            password=self.password,
            dsn=self.dsn,
            min=2,
            max=10,
            increment=1,
            encoding="UTF-8"
        )
        logger.info("Oracle 연결 풀 생성 완료")
    except Exception as e:
        logger.error(f"연결 풀 생성 실패: {str(e)}")
        raise

def test_connection(self) -> bool:
    """데이터베이스 연결 테스트"""
    try:
        with self.pool.acquire() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT 'Connection Test' FROM DUAL")
            result = cursor.fetchone()
            cursor.close()
        logger.info("Oracle 데이터베이스 연결 성공")
        return True
    except Exception as e:
        logger.error(f"Oracle 데이터베이스 연결 실패: {str(e)}")
        raise

def get_user_tables(self) -> List[str]:
    """사용자 소유 테이블 목록 조회"""
    try:
        with self.pool.acquire() as connection:
            cursor = connection.cursor()
            
            # 현재 사용자의 테이블 목록 조회
            cursor.execute("""
                SELECT TABLE_NAME 
                FROM USER_TABLES 
                ORDER BY TABLE_NAME
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
        
        logger.info(f"사용자 테이블 {len(tables)}개 조회됨")
        return tables
        
    except Exception as e:
        logger.error(f"테이블 목록 조회 실패: {str(e)}")
        return []

def get_all_accessible_tables(self, schema_filter: Optional[str] = None) -> Dict[str, str]:
    """접근 가능한 모든 테이블 조회 (스키마.테이블명 형태)"""
    try:
        with self.pool.acquire() as connection:
            cursor = connection.cursor()
            
            # 접근 가능한 모든 테이블 조회
            query = """
                SELECT OWNER, TABLE_NAME, COMMENTS
                FROM ALL_TAB_COMMENTS 
                WHERE OWNER NOT IN ('SYS', 'SYSTEM', 'DBSNMP', 'SYSMAN', 'OUTLN', 
                                   'MDSYS', 'ORDSYS', 'EXFSYS', 'DMSYS', 'WMSYS', 
                                   'CTXSYS', 'ANONYMOUS', 'XDB', 'ORDPLUGINS', 'OLAPSYS')
                AND TABLE_TYPE = 'TABLE'
            """
            
            if schema_filter:
                query += f" AND OWNER LIKE '%{schema_filter.upper()}%'"
            
            query += " ORDER BY OWNER, TABLE_NAME"
            
            cursor.execute(query)
            
            tables = {}
            for owner, table_name, comments in cursor.fetchall():
                full_name = f"{owner}.{table_name}"
                tables[full_name] = comments or "설명 없음"
            
            cursor.close()
        
        logger.info(f"접근 가능한 테이블 {len(tables)}개 조회됨")
        return tables
        
    except Exception as e:
        logger.error(f"테이블 목록 조회 실패: {str(e)}")
        return {}

def get_table_schema(self, table_name: str, schema_name: Optional[str] = None) -> Dict[str, Any]:
    """특정 테이블의 스키마 정보 조회"""
    try:
        with self.pool.acquire() as connection:
            cursor = connection.cursor()
            
            # 테이블 스키마 정보 조회
            if schema_name:
                table_ref = f"{schema_name}.{table_name}"
                schema_condition = f"AND OWNER = '{schema_name.upper()}'"
            else:
                table_ref = table_name
                schema_condition = f"AND OWNER = USER"
            
            query = f"""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    DATA_LENGTH,
                    DATA_PRECISION,
                    DATA_SCALE,
                    NULLABLE,
                    DATA_DEFAULT,
                    COLUMN_ID
                FROM ALL_TAB_COLUMNS 
                WHERE TABLE_NAME = '{table_name.upper()}'
                {schema_condition}
                ORDER BY COLUMN_ID
            """
            
            cursor.execute(query)
            columns = cursor.fetchall()
            
            # 테이블 코멘트 조회
            comment_query = f"""
                SELECT COMMENTS 
                FROM ALL_TAB_COMMENTS 
                WHERE TABLE_NAME = '{table_name.upper()}'
                {schema_condition}
            """
            cursor.execute(comment_query)
            comment_result = cursor.fetchone()
            table_comment = comment_result[0] if comment_result else "설명 없음"
            
            cursor.close()
        
        # 스키마 정보 구조화
        schema_info = {
            'table_name': table_ref,
            'comment': table_comment,
            'columns': []
        }
        
        for col in columns:
            col_name, data_type, data_length, data_precision, data_scale, nullable, default_val, col_id = col
            
            # 데이터 타입 문자열 구성
            if data_type in ['NUMBER']:
                if data_precision and data_scale:
                    type_str = f"{data_type}({data_precision},{data_scale})"
                elif data_precision:
                    type_str = f"{data_type}({data_precision})"
                else:
                    type_str = data_type
            elif data_type in ['VARCHAR2', 'CHAR', 'NVARCHAR2', 'NCHAR']:
                type_str = f"{data_type}({data_length})"
            else:
                type_str = data_type
            
            schema_info['columns'].append({
                'name': col_name,
                'type': type_str,
                'nullable': nullable == 'Y',
                'default': default_val,
                'position': col_id
            })
        
        logger.info(f"테이블 {table_ref} 스키마 조회 완료: {len(schema_info['columns'])}개 컬럼")
        return schema_info
        
    except Exception as e:
        logger.error(f"테이블 스키마 조회 실패 ({table_name}): {str(e)}")
        return {}

def execute_query(self, sql_query: str, max_rows: int = 1000) -> QueryResult:
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
        
        # 행 수 제한 추가
        if max_rows > 0 and "ROWNUM" not in cleaned_query.upper():
            if "ORDER BY" in cleaned_query.upper():
                # ORDER BY가 있는 경우 서브쿼리로 감싸기
                cleaned_query = f"""
                SELECT * FROM (
                    {cleaned_query}
                ) WHERE ROWNUM <= {max_rows}
                """
            else:
                # 단순히 WHERE 조건 추가
                if "WHERE" in cleaned_query.upper():
                    cleaned_query = cleaned_query + f" AND ROWNUM <= {max_rows}"
                else:
                    cleaned_query = cleaned_query + f" WHERE ROWNUM <= {max_rows}"
        
        # 쿼리 실행
        with self.pool.acquire() as connection:
            cursor = connection.cursor()
            cursor.execute(cleaned_query)
            
            # 컬럼명 가져오기
            columns = [desc[0] for desc in cursor.description]
            
            # 데이터 가져오기
            rows = cursor.fetchall()
            cursor.close()
        
        # DataFrame 생성
        df = pd.DataFrame(rows, columns=columns)
        
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
    
    # SELECT 문만 허용 (WITH 절도 허용)
    if not (query_upper.startswith('SELECT') or query_upper.startswith('WITH')):
        return False
    
    # 위험한 키워드 차단
    dangerous_keywords = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
        'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK'
    ]
    
    for keyword in dangerous_keywords:
        if f' {keyword} ' in f' {query_upper} ':
            return False
    
    return True

def close(self):
    """연결 풀 종료"""
    if self.pool:
        self.pool.close()
        logger.info("Oracle 연결 풀 종료됨")
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
        max_tokens=1500
    )
    
    # 접근 가능한 테이블 목록 로드
    self.available_tables = oracle_manager.get_all_accessible_tables()
    logger.info(f"사용 가능한 테이블 {len(self.available_tables)}개 로드됨")

def get_relevant_tables(self, natural_query: str, max_tables: int = 5) -> List[str]:
    """자연어 질의에서 관련 테이블 추출"""
    query_lower = natural_query.lower()
    relevant_tables = []
    
    # 테이블명 매칭 (부분 매칭 포함)
    for table_name, comment in self.available_tables.items():
        table_lower = table_name.lower()
        comment_lower = (comment or "").lower()
        
        # 테이블명이나 코멘트에서 키워드 매칭
        if any(word in table_lower or word in comment_lower 
               for word in query_lower.split() if len(word) > 2):
            relevant_tables.append(table_name)
        
        if len(relevant_tables) >= max_tables:
            break
    
    # 관련 테이블이 없으면 상위 몇 개 테이블 반환
    if not relevant_tables:
        relevant_tables = list(self.available_tables.keys())[:max_tables]
    
    return relevant_tables

def get_table_schemas_context(self, table_names: List[str]) -> str:
    """테이블들의 스키마 정보를 컨텍스트로 구성"""
    context = "=== 사용 가능한 테이블 스키마 ===\n\n"
    
    for table_name in table_names:
        # 스키마와 테이블명 분리
        if '.' in table_name:
            schema_name, tab_name = table_name.split('.', 1)
        else:
            schema_name, tab_name = None, table_name
        
        # 테이블 스키마 조회
        schema_info = self.oracle_manager.get_table_schema(tab_name, schema_name)
        
        if schema_info:
            context += f"테이블: {table_name}\n"
            context += f"설명: {schema_info.get('comment', '설명 없음')}\n"
            context += "컬럼:\n"
            
            for col in schema_info.get('columns', []):
                nullable = "NULL 가능" if col['nullable'] else "NOT NULL"
                context += f"  - {col['name']}: {col['type']} ({nullable})\n"
            
            context += "\n"
    
    return context

def convert_natural_language_to_sql(self, natural_query: str) -> str:
    """자연어를 Oracle SQL로 변환"""
    try:
        # 관련 테이블 찾기
        relevant_tables = self.get_relevant_tables(natural_query)
        
        # 스키마 컨텍스트 구성
        schema_context = self.get_table_schemas_context(relevant_tables)
        
        # 프롬프트 구성
        prompt = self._build_conversion_prompt(natural_query, schema_context)
        
        # LLM을 통한 SQL 생성
        response = self.llm.complete(prompt)
        sql_query = self._extract_sql_from_response(response.text)
        
        logger.info(f"자연어 → SQL 변환 완료")
        logger.debug(f"생성된 SQL: {sql_query}")
        
        return sql_query
        
    except Exception as e:
        logger.error(f"SQL 변환 실패: {str(e)}")
        raise

def _build_conversion_prompt(self, natural_query: str, schema_context: str) -> str:
    """SQL 변환용 프롬프트 구성"""
    prompt = f"""
```

당신은 Oracle 데이터베이스 전문가입니다. 자연어 질의를 Oracle SQL로 변환해주세요.

{schema_context}

=== Oracle SQL 변환 규칙 ===

1. Oracle SQL 문법을 정확히 사용하세요
1. 테이블명은 스키마.테이블명 형태로 정확히 사용하세요
1. 컬럼명과 테이블명은 대소문자를 정확히 맞춰주세요
1. 한국어 검색 조건은 LIKE 연산자와 와일드카드(%)를 사용하세요
1. 날짜 조건은 TO_DATE 함수나 DATE 리터럴을 사용하세요
1. SELECT 문만 생성하세요 (DML/DDL 금지)
1. 성능을 위해 적절한 WHERE 조건을 사용하세요
1. 결과가 많을 것 같으면 ROWNUM으로 제한하세요
1. 가능한 경우 적절한 ORDER BY를 추가하세요
1. 집계 함수 사용시 GROUP BY를 올바르게 사용하세요

=== 자연어 질의 ===
{natural_query}

=== 생성할 Oracle SQL ===
(SQL 쿼리만 반환하고, 설명이나 코드 블록 마크다운은 사용하지 마세요)
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
    cleaned_response = response_text.strip()
    
    # 불필요한 설명 제거 (SQL로 시작하는 부분만 추출)
    lines = cleaned_response.split('\n')
    sql_lines = []
    sql_started = False
    
    for line in lines:
        line_upper = line.strip().upper()
        if line_upper.startswith('SELECT') or line_upper.startswith('WITH'):
            sql_started = True
        
        if sql_started:
            sql_lines.append(line)
            # SQL 문이 끝나는 조건 (세미콜론이나 빈 줄)
            if line.strip().endswith(';'):
                break
    
    if sql_lines:
        return '\n'.join(sql_lines).strip()
    
    # 그래도 찾지 못하면 전체 응답 반환
    return cleaned_response

def process_natural_query(self, 
                        natural_query: str,
                        execute: bool = True,
                        max_rows: int = 1000) -> Dict[str, Any]:
    """자연어 질의 전체 처리 파이프라인"""
    result = {
        'natural_query': natural_query,
        'sql_query': None,
        'success': False,
        'data': None,
        'error_message': None,
        'execution_time': None,
        'row_count': None,
        'relevant_tables': []
    }
    
    try:
        # 1. 관련 테이블 찾기
        relevant_tables = self.get_relevant_tables(natural_query)
        result['relevant_tables'] = relevant_tables
        
        # 2. 자연어 → SQL 변환
        logger.info(f"자연어 질의 처리 시작: {natural_query}")
        sql_query = self.convert_natural_language_to_sql(natural_query)
        result['sql_query'] = sql_query
        
        if not execute:
            result['success'] = True
            return result
        
        # 3. SQL 실행
        query_result = self.oracle_manager.execute_query(sql_query, max_rows)
        
        # 4. 결과 통합
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
```

class Text2SQLOracle:
“”“Oracle Text2SQL 메인 클래스 (oracledb 사용)”””

```
def __init__(self,
             username: str,
             password: str,
             host: str,
             port: int = 1521,
             service_name: str = "SMIP_DEV",
             openai_api_base: str,
             openai_api_key: str = "fake-key",
             model_name: str = "gpt-3.5-turbo"):
    """
    초기화
    
    Args:
        username: Oracle 사용자명
        password: Oracle 비밀번호
        host: Oracle 호스트
        port: Oracle 포트
        service_name: Oracle 서비스명
        openai_api_base: OpenAI-like API URL
        openai_api_key: API 키
        model_name: 모델명
    """
    
    # Oracle 연결 설정
    self.oracle_manager = OracleConnectionManager(
        username, password, host, port, service_name
    )
    
    # Text2SQL 변환기 설정
    self.converter = Text2SQLConverter(
        self.oracle_manager,
        openai_api_base,
        openai_api_key,
        model_name
    )
    
    logger.info("Text2SQL Oracle 시스템 초기화 완료")

def query(self, natural_language: str, max_rows: int = 1000) -> Dict[str, Any]:
    """자연어로 데이터베이스 조회"""
    return self.converter.process_natural_query(
        natural_language, 
        execute=True, 
        max_rows=max_rows
    )

def get_sql_only(self, natural_language: str) -> str:
    """SQL만 생성 (실행하지 않음)"""
    result = self.converter.process_natural_query(
        natural_language, 
        execute=False
    )
    return result.get('sql_query', '')

def get_available_tables(self) -> Dict[str, str]:
    """사용 가능한 테이블 목록 조회"""
    return self.converter.available_tables

def get_table_schema(self, table_name: str, schema_name: Optional[str] = None) -> Dict[str, Any]:
    """특정 테이블의 스키마 정보 조회"""
    return self.oracle_manager.get_table_schema(table_name, schema_name)

def test_connection(self) -> bool:
    """연결 테스트"""
    return self.oracle_manager.test_connection()

def close(self):
    """연결 종료"""
    self.oracle_manager.close()
```

def main():
“”“사용 예시”””

```
# 설정 (실제 사용시 환경변수나 설정 파일 사용 권장)
config = {
    'username': 'your_username',
    'password': 'your_password', 
    'host': 'your_oracle_host',
    'port': 1521,
    'service_name': 'SMIP_DEV',
    'openai_api_base': 'http://localhost:8000/v1',  # 예시 로컬 API
    'openai_api_key': 'your-api-key',
    'model_name': 'gpt-3.5-turbo'
}

try:
    # Text2SQL 시스템 초기화
    print("Text2SQL Oracle 시스템 초기화 중...")
    text2sql = Text2SQLOracle(**config)
    
    # 사용 가능한 테이블 조회
    print("\n=== 사용 가능한 테이블 목록 ===")
    tables = text2sql.get_available_tables()
    print(f"총 테이블 수: {len(tables)}개")
    
    # 상위 10개 테이블 출력
    for i, (table_name, comment) in enumerate(list(tables.items())[:10]):
        print(f"{i+1}. {table_name}: {comment}")
    
    # 샘플 질의들
    sample_queries = [
        "모든 사용자 정보를 보여주세요",
        "최근 1개월간 거래 내역을 조회해주세요", 
        "부서별 직원 수를 계산해주세요",
        "급여가 가장 높은 상위 10명을 알려주세요",
        "제품별 매출 현황을 보여주세요"
    ]
    
    print("\n=== 샘플 질의 테스트 ===")
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{i}. 자연어 질의: {query}")
        
        # SQL만 생성
        try:
            sql = text2sql.get_sql_only(query)
            print(f"   생성된 SQL: {sql}")
            
            # 실제 실행 (주석 해제하여 사용)
            # result = text2sql.query(query, max_rows=5)
            # if result['success']:
            #     print(f"   결과: {result['row_count']}개 행")
            #     if result['data'] is not None:
            #         print(result['data'].head())
            # else:
            #     print(f"   오류: {result['error_message']}")
                
        except Exception as e:
            print(f"   오류: {str(e)}")
    
    # 연결 종료
    text2sql.close()
    
    print("\n실제 사용을 위해서는 Oracle DB 연결 정보와 OpenAI-like API 설정이 필요합니다.")
    
except Exception as e:
    print(f"초기화 실패: {str(e)}")
    print("Oracle 클라이언트와 oracledb 패키지가 설치되었는지 확인하세요.")
```

if **name** == “**main**”:
main()