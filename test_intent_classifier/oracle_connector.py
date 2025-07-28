import oracledb
import pandas as pd
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """SQL 쿼리 결과 데이터 클래스"""
    success: bool
    data: Optional[pd.DataFrame] = None
    sql_query: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    row_count: Optional[int] = None

class OracleConnectionManager:
    """Oracle 데이터베이스 연결 관리 클래스 (oracledb 사용)"""
    
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
    
    def close(self):
        """연결 풀 종료"""
        if self.pool:
            self.pool.close()
            logger.info("Oracle 연결 풀 종료됨")
    
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
    
    def get_table_relationships(self, table_names: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """테이블 간 관계 분석 (외래키 제약조건 기반)"""
        try:
            with self.pool.acquire() as connection:
                cursor = connection.cursor()
                
                relationships = {}
                
                for table_name in table_names:
                    if '.' in table_name:
                        schema_name, tab_name = table_name.split('.', 1)
                    else:
                        schema_name, tab_name = None, table_name
                    
                    # 외래키 제약조건 조회
                    fk_query = f"""
                        SELECT 
                            a.constraint_name,
                            a.column_name,
                            c_pk.table_name r_table_name,
                            c_pk.owner r_owner,
                            b.column_name r_column_name
                        FROM all_cons_columns a
                        JOIN all_constraints c ON a.owner = c.owner AND a.constraint_name = c.constraint_name
                        JOIN all_constraints c_pk ON c.r_owner = c_pk.owner AND c.r_constraint_name = c_pk.constraint_name
                        JOIN all_cons_columns b ON c_pk.owner = b.owner AND c_pk.constraint_name = b.constraint_name AND b.position = a.position
                        WHERE c.constraint_type = 'R'
                        AND a.table_name = '{tab_name.upper()}'
                    """
                    
                    if schema_name:
                        fk_query += f" AND a.owner = '{schema_name.upper()}'"
                    
                    cursor.execute(fk_query)
                    fk_results = cursor.fetchall()
                    
                    table_relationships = []
                    for constraint_name, column_name, r_table_name, r_owner, r_column_name in fk_results:
                        table_relationships.append({
                            'constraint_name': constraint_name,
                            'local_column': column_name,
                            'remote_table': f"{r_owner}.{r_table_name}",
                            'remote_column': r_column_name,
                            'join_condition': f"{table_name}.{column_name} = {r_owner}.{r_table_name}.{r_column_name}"
                        })
                    
                    if table_relationships:
                        relationships[table_name] = table_relationships
                
                cursor.close()
                
            logger.info(f"테이블 관계 분석 완료: {len(relationships)}개 테이블")
            return relationships
            
        except Exception as e:
            logger.error(f"테이블 관계 분석 실패: {str(e)}")
            return {}
    
    def find_common_columns(self, table_names: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """테이블 간 공통 컬럼 찾기 (조인 후보)"""
        try:
            table_columns = {}
            
            # 각 테이블의 컬럼 정보 수집
            for table_name in table_names:
                if '.' in table_name:
                    schema_name, tab_name = table_name.split('.', 1)
                else:
                    schema_name, tab_name = None, table_name
                
                schema_info = self.get_table_schema(tab_name, schema_name)
                if schema_info:
                    table_columns[table_name] = [col['name'] for col in schema_info.get('columns', [])]
            
            # 공통 컬럼 찾기
            common_columns = {}
            for i, table1 in enumerate(table_names):
                for table2 in table_names[i+1:]:
                    if table1 in table_columns and table2 in table_columns:
                        common_cols = []
                        
                        for col1 in table_columns[table1]:
                            for col2 in table_columns[table2]:
                                # 완전 일치하거나 ID 패턴 매칭
                                if (col1 == col2 or 
                                    (col1.endswith('_ID') and col2 == 'ID') or
                                    (col2.endswith('_ID') and col1 == 'ID') or
                                    (col1.endswith('_NO') and col2.endswith('_NO')) or
                                    (col1.endswith('_CODE') and col2.endswith('_CODE'))):
                                    
                                    common_cols.append({
                                        'table1_column': col1,
                                        'table2_column': col2,
                                        'join_condition': f"{table1}.{col1} = {table2}.{col2}"
                                    })
                        
                        if common_cols:
                            key = f"{table1}___{table2}"
                            common_columns[key] = common_cols
            
            logger.info(f"공통 컬럼 분석 완료: {len(common_columns)}개 관계")
            return common_columns
            
        except Exception as e:
            logger.error(f"공통 컬럼 분석 실패: {str(e)}")
            return {}
