# text2sql_engine.py
from oracle_connector import OracleConnectionManager # 수정된 import
import pandas as pd
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import re
from llama_index.llms.openai_like import OpenAILike

logger = logging.getLogger(__name__)


class Text2SQLConverter:
    """자연어를 SQL로 변환하는 클래스"""
    
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
        """자연어 질의에서 관련 테이블 추출 (조인 고려)"""
        query_lower = natural_query.lower()
        relevant_tables = []
        table_scores = {}
        
        # 1차: 키워드 기반 테이블 매칭
        for table_name, comment in self.available_tables.items():
            table_lower = table_name.lower()
            comment_lower = (comment or "").lower()
            score = 0
            
            # 질의 단어들과 테이블명/코멘트 매칭
            query_words = [word for word in query_lower.split() if len(word) > 2]
            for word in query_words:
                if word in table_lower:
                    score += 3  # 테이블명 직접 매칭은 높은 점수
                elif word in comment_lower:
                    score += 2  # 코멘트 매칭
                
                # 특정 키워드에 따른 테이블 유형 추론
                table_type_keywords = {
                    'user': ['사용자', '회원', '고객', '직원'],
                    'product': ['제품', '상품', '아이템'],
                    'order': ['주문', '거래', '구매'],
                    'department': ['부서', '조직'],
                    'employee': ['직원', '사원', '임직원'],
                    'sales': ['매출', '판매', '영업'],
                    'inventory': ['재고', '입출고'],
                    'customer': ['고객', '회원'],
                    'transaction': ['거래', '트랜잭션', '이체']
                }
                
                for table_type, keywords in table_type_keywords.items():
                    if any(keyword in query_lower for keyword in keywords):
                        if table_type in table_lower:
                            score += 2
            
            if score > 0:
                table_scores[table_name] = score
        
        # 점수 순으로 정렬
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        initial_tables = [table for table, score in sorted_tables[:max_tables]]
        
        # 2차: 관련 테이블들과 조인 가능한 테이블 추가
        if len(initial_tables) > 1:
            # 선택된 테이블들 간의 관계 확인
            relationships = self.oracle_manager.get_table_relationships(initial_tables)
            common_columns = self.oracle_manager.find_common_columns(initial_tables)
            
            # 관계가 있는 테이블들을 우선 선택
            connected_tables = set(initial_tables)
            
            # 외래키 관계로 연결된 테이블 추가
            for table, relations in relationships.items():
                for relation in relations:
                    remote_table = relation['remote_table']
                    if remote_table in self.available_tables:
                        connected_tables.add(remote_table)
            
            relevant_tables = list(connected_tables)[:max_tables]
        else:
            relevant_tables = initial_tables
        
        # 관련 테이블이 없으면 상위 몇 개 테이블 반환
        if not relevant_tables:
            relevant_tables = list(self.available_tables.keys())[:max_tables]
        
        logger.info(f"관련 테이블 선택: {relevant_tables}")
        return relevant_tables
    
    def get_table_schemas_context(self, table_names: List[str]) -> str:
        """테이블들의 스키마 정보와 관계를 컨텍스트로 구성"""
        context = "=== 사용 가능한 테이블 스키마 ===\n\n"
        
        # 기본 스키마 정보
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
        
        # 테이블 관계 정보 추가
        if len(table_names) > 1:
            context += "=== 테이블 간 관계 정보 ===\n\n"
            
            # 외래키 관계
            relationships = self.oracle_manager.get_table_relationships(table_names)
            if relationships:
                context += "**외래키 관계:**\n"
                for table, relations in relationships.items():
                    for relation in relations:
                        context += f"- {relation['join_condition']}\n"
                context += "\n"
            
            # 공통 컬럼 관계
            common_columns = self.oracle_manager.find_common_columns(table_names)
            if common_columns:
                context += "**공통 컬럼 기반 조인 후보:**\n"
                for table_pair, columns in common_columns.items():
                    table1, table2 = table_pair.split('___')
                    context += f"- {table1} ↔ {table2}:\n"
                    for col_info in columns:
                        context += f"  * {col_info['join_condition']}\n"
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
        """SQL 변환용 프롬프트 구성 (조인 지원)"""
        prompt = f"""
당신은 Oracle 데이터베이스 전문가입니다. 자연어 질의를 Oracle SQL로 변환해주세요.

{schema_context}

=== Oracle SQL 변환 규칙 ===

**기본 규칙:**
1. Oracle SQL 문법을 정확히 사용하세요
2. 테이블명은 스키마.테이블명 형태로 정확히 사용하세요  
3. 컬럼명과 테이블명은 대소문자를 정확히 맞춰주세요
4. SELECT 문만 생성하세요 (DML/DDL 금지)

**조인 관련 규칙:**
5. 여러 테이블에서 데이터를 가져와야 하는 경우 적절한 JOIN을 사용하세요
6. 테이블 간 관계를 파악하여 올바른 조인 조건을 설정하세요:
   - 주키(Primary Key)와 외래키(Foreign Key) 관계 활용
   - 공통 컬럼명 (ID, CODE, NO 등)을 통한 조인
   - 날짜나 코드 기반 조인 고려
7. 조인 유형 선택:
   - INNER JOIN: 양쪽 테이블에 모두 존재하는 데이터만
   - LEFT JOIN: 왼쪽 테이블의 모든 데이터 + 오른쪽 테이블의 매칭 데이터
   - RIGHT JOIN: 오른쪽 테이블의 모든 데이터 + 왼쪽 테이블의 매칭 데이터
   - FULL OUTER JOIN: 양쪽 테이블의 모든 데이터
8. 테이블 별칭(alias) 사용으로 가독성 향상 (예: SELECT u.name, d.dept_name FROM users u JOIN departments d ON u.dept_id = d.id)
9. 조인 조건은 ON 절에, 필터 조건은 WHERE 절에 분리하여 작성

**성능 및 품질 규칙:**
10. 한국어 검색 조건은 LIKE 연산자와 와일드카드(%)를 사용하세요
11. 날짜 조건은 TO_DATE 함수, DATE 리터럴, 또는 TRUNC 함수를 사용하세요
12. 성능을 위해 적절한 WHERE 조건을 사용하세요
13. 결과가 많을 것 같으면 ROWNUM으로 제한하거나 TOP-N 쿼리를 사용하세요
14. 적절한 ORDER BY를 추가하여 정렬된 결과를 제공하세요
15. 집계 함수 사용시 GROUP BY를 올바르게 사용하세요
16. 서브쿼리가 필요한 경우 EXISTS, IN, 스칼라 서브쿼리 등을 적절히 활용하세요

**조인 예시 패턴:**
- 사용자와 부서: users u JOIN departments d ON u.dept_id = d.id
- 주문과 고객: orders o JOIN customers c ON o.customer_id = c.id  
- 제품과 카테고리: products p JOIN categories cat ON p.category_id = cat.id
- 거래와 사용자: transactions t JOIN users u ON t.user_id = u.id
- 상위-하위 관계: parent p JOIN child c ON p.id = c.parent_id

**복잡한 쿼리 처리:**
- 다중 조인: A JOIN B ON ... JOIN C ON ... 
- 조건부 조인: LEFT JOIN과 WHERE 조건 조합
- 집계와 조인: GROUP BY와 JOIN 조합
- 서브쿼리와 조인: JOIN (SELECT ... FROM ...) sub ON ...

=== 자연어 질의 ===
{natural_query}

=== 분석 과정 ===
1. 필요한 정보가 어떤 테이블들에 있는지 파악
2. 테이블 간 관계와 조인 조건 확인  
3. 필터 조건과 정렬 조건 결정
4. 최적화된 Oracle SQL 작성

=== 생성할 Oracle SQL ===
(SQL 쿼리만 반환하고, 설명이나 코드 블록 마크다운은 사용하지 마세요)
"""
        return prompt
    
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
