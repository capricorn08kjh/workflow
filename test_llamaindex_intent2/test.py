import os
import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# LlamaIndex imports (실제 환경에서 사용)
# from llama_index.core import VectorStoreIndex, Document, Settings
# from llama_index.core.query_engine import RouterQueryEngine, NLSQLTableQueryEngine
# from llama_index.core.tools import QueryEngineTool
# from llama_index.llms.openai import OpenAI
# from llama_index.core.response_synthesizers import ResponseMode

# 데모를 위한 Mock 클래스들
class MockLLM:
    """실제 LLM 대신 사용할 Mock 클래스"""
    
    def generate_sql(self, query: str, schema: str) -> str:
        """자연어 질문을 SQL로 변환"""
        query_lower = query.lower()
        
        if "공정별" in query and "실적" in query:
            return """
            SELECT 공정명, SUM(생산량) as 총생산량, AVG(효율성) as 평균효율성
            FROM 공정실적 
            GROUP BY 공정명 
            ORDER BY 총생산량 DESC
            """
        elif "월간보고서" in query and "결산" in query:
            return """
            SELECT m.월, m.매출, m.비용, f.순이익, f.ROE
            FROM 월간보고서 m
            JOIN 결산실적 f ON m.월 = f.월
            WHERE m.월 = 4
            """
        elif "분기별" in query and "실적" in query:
            return """
            SELECT 분기, 매출, 영업이익, 순이익
            FROM 분기실적
            ORDER BY 분기
            """
        elif "부서별" in query and "예산" in query:
            return """
            SELECT 부서명, 배정예산, 사용예산, (배정예산-사용예산) as 잔여예산
            FROM 부서예산
            ORDER BY 잔여예산 DESC
            """
        else:
            return "SELECT * FROM 기본테이블 LIMIT 10"
    
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """질의 의도 분류"""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["실적", "매출", "분기", "월간", "결산", "예산", "공정"]):
            return {
                "intent": "structured_data_query",
                "confidence": 0.9,
                "parameters": {
                    "query_type": "database",
                    "entities": self._extract_entities(query)
                }
            }
        elif any(keyword in query_lower for keyword in ["회의록", "보고서", "문서", "언급"]):
            return {
                "intent": "document_search",
                "confidence": 0.85,
                "parameters": {
                    "query_type": "document",
                    "search_terms": self._extract_search_terms(query)
                }
            }
        else:
            return {
                "intent": "general_query",
                "confidence": 0.7,
                "parameters": {}
            }
    
    def _extract_entities(self, query: str) -> List[str]:
        """엔티티 추출"""
        entities = []
        if "분기" in query: entities.append("분기")
        if "공정" in query: entities.append("공정")
        if "부서" in query: entities.append("부서")
        if "월간" in query: entities.append("월간")
        return entities
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """검색어 추출"""
        terms = []
        # 간단한 인명 패턴 매칭
        if "김영철" in query: terms.append("김영철")
        if "회의록" in query: terms.append("회의록")
        if "보고서" in query: terms.append("보고서")
        return terms
    
    def generate_response(self, data: Any, query: str) -> str:
        """최종 응답 생성"""
        if isinstance(data, pd.DataFrame):
            if len(data) == 0:
                return "요청하신 데이터를 찾을 수 없습니다."
            
            summary = f"총 {len(data)}개의 결과를 찾았습니다.\n\n"
            if "공정" in query:
                summary += "공정별 실적 분석 결과입니다. 차트로 더 자세한 내용을 확인하실 수 있습니다."
            elif "분기" in query:
                summary += "분기별 실적 추이를 아래 표와 차트로 나타냈습니다."
            elif "보고서" in query:
                summary += "월간보고서와 결산실적을 통합하여 보여드립니다."
            else:
                summary += "요청하신 데이터 조회 결과입니다."
            
            return summary
        else:
            return str(data)

class QueryType(Enum):
    STRUCTURED_DATA = "structured_data"
    DOCUMENT_SEARCH = "document_search"
    GENERAL = "general"

@dataclass
class QueryResult:
    """질의 결과 데이터 클래스"""
    intent: str
    data: Any
    visualization_type: str
    response_text: str
    chart_path: Optional[str] = None

class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "demo.db"):
        self.db_path = db_path
        self.setup_demo_data()
    
    def setup_demo_data(self):
        """데모용 데이터 설정"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 공정실적 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS 공정실적 (
            id INTEGER PRIMARY KEY,
            공정명 TEXT,
            생산량 INTEGER,
            효율성 REAL,
            날짜 DATE
        )
        """)
        
        # 분기실적 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS 분기실적 (
            id INTEGER PRIMARY KEY,
            분기 TEXT,
            매출 INTEGER,
            영업이익 INTEGER,
            순이익 INTEGER
        )
        """)
        
        # 월간보고서 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS 월간보고서 (
            id INTEGER PRIMARY KEY,
            월 INTEGER,
            매출 INTEGER,
            비용 INTEGER
        )
        """)
        
        # 결산실적 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS 결산실적 (
            id INTEGER PRIMARY KEY,
            월 INTEGER,
            순이익 INTEGER,
            ROE REAL
        )
        """)
        
        # 부서예산 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS 부서예산 (
            id INTEGER PRIMARY KEY,
            부서명 TEXT,
            배정예산 INTEGER,
            사용예산 INTEGER
        )
        """)
        
        # 샘플 데이터 삽입
        sample_data = [
            # 공정실적
            ("INSERT OR REPLACE INTO 공정실적 VALUES (1, 'A공정', 1500, 85.5, '2024-01-15')", ()),
            ("INSERT OR REPLACE INTO 공정실적 VALUES (2, 'B공정', 1200, 78.2, '2024-01-15')", ()),
            ("INSERT OR REPLACE INTO 공정실적 VALUES (3, 'C공정', 1800, 92.1, '2024-01-15')", ()),
            
            # 분기실적
            ("INSERT OR REPLACE INTO 분기실적 VALUES (1, '2024Q1', 15000000, 2500000, 1800000)", ()),
            ("INSERT OR REPLACE INTO 분기실적 VALUES (2, '2024Q2', 18000000, 3200000, 2400000)", ()),
            ("INSERT OR REPLACE INTO 분기실적 VALUES (3, '2024Q3', 16500000, 2800000, 2100000)", ()),
            
            # 월간보고서
            ("INSERT OR REPLACE INTO 월간보고서 VALUES (1, 4, 8500000, 6200000)", ()),
            
            # 결산실적
            ("INSERT OR REPLACE INTO 결산실적 VALUES (1, 4, 2300000, 12.5)", ()),
            
            # 부서예산
            ("INSERT OR REPLACE INTO 부서예산 VALUES (1, '영업부', 5000000, 4200000)", ()),
            ("INSERT OR REPLACE INTO 부서예산 VALUES (2, '개발부', 8000000, 7500000)", ()),
            ("INSERT OR REPLACE INTO 부서예산 VALUES (3, '마케팅부', 3000000, 2800000)", ())
        ]
        
        for query, params in sample_data:
            cursor.execute(query, params)
        
        conn.commit()
        conn.close()
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """SQL 쿼리 실행"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql, conn)
            conn.close()
            return df
        except Exception as e:
            logging.error(f"Database query error: {e}")
            return pd.DataFrame()

class DocumentSearchManager:
    """문서 검색 관리 클래스"""
    
    def __init__(self):
        self.documents = self._setup_demo_documents()
    
    def _setup_demo_documents(self) -> List[Dict]:
        """데모용 문서 데이터"""
        return [
            {
                "id": "meeting_001",
                "title": "2024년 1분기 영업회의록",
                "content": "김영철 팀장이 새로운 마케팅 전략에 대해 발표했습니다. 목표 달성률은 95%로 예상됩니다.",
                "date": "2024-03-15",
                "type": "회의록"
            },
            {
                "id": "meeting_002", 
                "title": "개발팀 주간회의록",
                "content": "프로젝트 진행상황을 점검했습니다. 김영철씨가 제안한 아키텍처 개선안이 채택되었습니다.",
                "date": "2024-04-02",
                "type": "회의록"
            },
            {
                "id": "report_001",
                "title": "4월 월간보고서",
                "content": "4월 매출은 전월 대비 15% 증가했습니다. 주요 성과 지표들이 개선되었습니다.",
                "date": "2024-04-30",
                "type": "보고서"
            }
        ]
    
    def search_documents(self, search_terms: List[str]) -> List[Dict]:
        """문서 검색"""
        results = []
        for doc in self.documents:
            for term in search_terms:
                if term in doc["content"] or term in doc["title"]:
                    results.append(doc)
                    break
        return results

class ChartGenerator:
    """차트 생성 클래스"""
    
    def __init__(self, output_dir: str = "charts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
        sns.set_style("whitegrid")
    
    def create_chart(self, data: pd.DataFrame, chart_type: str, query: str) -> str:
        """차트 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        plt.figure(figsize=(10, 6))
        
        if chart_type == "bar" and "공정" in query:
            if '공정명' in data.columns and '총생산량' in data.columns:
                plt.bar(data['공정명'], data['총생산량'])
                plt.title('Process Performance by Production Volume')
                plt.xlabel('Process Name')
                plt.ylabel('Total Production')
                plt.xticks(rotation=45)
        
        elif chart_type == "line" and "분기" in query:
            if '분기' in data.columns and '매출' in data.columns:
                plt.plot(data['분기'], data['매출'], marker='o', label='Revenue')
                plt.plot(data['분기'], data['영업이익'], marker='s', label='Operating Profit')
                plt.title('Quarterly Performance Trend')
                plt.xlabel('Quarter')
                plt.ylabel('Amount (KRW)')
                plt.legend()
                plt.xticks(rotation=45)
        
        elif chart_type == "combined" and "보고서" in query:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 매출/비용 비교
            categories = ['Revenue', 'Cost']
            values = [data['매출'].iloc[0], data['비용'].iloc[0]]
            ax1.bar(categories, values, color=['green', 'red'])
            ax1.set_title('April Monthly Report')
            ax1.set_ylabel('Amount (KRW)')
            
            # ROE 표시
            ax2.bar(['ROE'], [data['ROE'].iloc[0]], color='blue')
            ax2.set_title('Financial Performance')
            ax2.set_ylabel('ROE (%)')
        
        else:
            # 기본 차트
            if len(data.columns) >= 2:
                data.plot(kind='bar', ax=plt.gca())
                plt.title('Data Visualization')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath

class LlamaIndexSystem:
    """메인 LlamaIndex 시스템 클래스"""
    
    def __init__(self):
        self.llm = MockLLM()
        self.db_manager = DatabaseManager()
        self.doc_manager = DocumentSearchManager()
        self.chart_generator = ChartGenerator()
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_query(self, user_query: str) -> QueryResult:
        """사용자 질의 처리 메인 함수"""
        
        # 1. 의도 분류 및 파라미터 추출
        intent_result = self.llm.classify_intent(user_query)
        self.logger.info(f"Intent classification: {intent_result}")
        
        if intent_result["intent"] == "structured_data_query":
            return self._handle_structured_query(user_query, intent_result)
        
        elif intent_result["intent"] == "document_search":
            return self._handle_document_query(user_query, intent_result)
        
        else:
            return self._handle_general_query(user_query)
    
    def _handle_structured_query(self, query: str, intent_result: Dict) -> QueryResult:
        """정형 데이터 질의 처리"""
        
        # 2. SQL 쿼리 생성
        schema = self._get_database_schema()
        sql_query = self.llm.generate_sql(query, schema)
        self.logger.info(f"Generated SQL: {sql_query}")
        
        # 3. 데이터베이스 조회 실행
        result_data = self.db_manager.execute_query(sql_query)
        
        if result_data.empty:
            return QueryResult(
                intent="structured_data_query",
                data=result_data,
                visualization_type="none",
                response_text="요청하신 데이터를 찾을 수 없습니다."
            )
        
        # 4. 시각화 준비
        chart_type = self._determine_chart_type(query)
        chart_path = None
        
        if chart_type != "table_only":
            chart_path = self.chart_generator.create_chart(result_data, chart_type, query)
        
        # 5. 최종 응답 생성
        response_text = self.llm.generate_response(result_data, query)
        
        return QueryResult(
            intent="structured_data_query",
            data=result_data,
            visualization_type=chart_type,
            response_text=response_text,
            chart_path=chart_path
        )
    
    def _handle_document_query(self, query: str, intent_result: Dict) -> QueryResult:
        """문서 검색 질의 처리"""
        
        search_terms = intent_result["parameters"].get("search_terms", [])
        documents = self.doc_manager.search_documents(search_terms)
        
        if not documents:
            response_text = "검색 조건에 맞는 문서를 찾을 수 없습니다."
        else:
            response_text = f"총 {len(documents)}개의 관련 문서를 찾았습니다:\n\n"
            for doc in documents:
                response_text += f"- {doc['title']} ({doc['date']})\n"
                response_text += f"  내용: {doc['content'][:100]}...\n\n"
        
        return QueryResult(
            intent="document_search",
            data=documents,
            visualization_type="none",
            response_text=response_text
        )
    
    def _handle_general_query(self, query: str) -> QueryResult:
        """일반 질의 처리"""
        response_text = "죄송합니다. 해당 질문에 대해서는 구체적인 데이터나 문서를 찾을 수 없습니다. 다른 방식으로 질문해 주시거나, 구체적인 데이터나 문서명을 지정해 주세요."
        
        return QueryResult(
            intent="general_query",
            data=None,
            visualization_type="none",
            response_text=response_text
        )
    
    def _get_database_schema(self) -> str:
        """데이터베이스 스키마 정보 반환"""
        return """
        Available tables:
        - 공정실적: 공정명, 생산량, 효율성, 날짜
        - 분기실적: 분기, 매출, 영업이익, 순이익
        - 월간보고서: 월, 매출, 비용
        - 결산실적: 월, 순이익, ROE
        - 부서예산: 부서명, 배정예산, 사용예산
        """
    
    def _determine_chart_type(self, query: str) -> str:
        """질의에 따른 차트 타입 결정"""
        if "공정" in query:
            return "bar"
        elif "분기" in query or "추이" in query:
            return "line"
        elif "보고서" in query and "결산" in query:
            return "combined"
        else:
            return "table_only"

# 데모 실행 함수
def demo_system():
    """시스템 데모 실행"""
    system = LlamaIndexSystem()
    
    # 예상 사용자 쿼리들
    sample_queries = [
        "공정별 실적을 찾아줘",
        "회의록 중 김영철씨가 언급된 것을 찾아줘", 
        "4월 월간보고서와 결산 실적을 같이 보여줘",
        "분기별 매출 추이를 보여줘",
        "부서별 예산 사용 현황을 알려줘"
    ]
    
    print("=== LlamaIndex 기반 LLM 시스템 데모 ===\n")
    
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. 사용자 질의: '{query}'")
        print("-" * 50)
        
        try:
            result = system.process_query(query)
            
            print(f"의도 분류: {result.intent}")
            print(f"응답: {result.response_text}")
            
            if isinstance(result.data, pd.DataFrame) and not result.data.empty:
                print("\n데이터 테이블:")
                print(result.data.to_string(index=False))
            
            if result.chart_path:
                print(f"\n차트 생성됨: {result.chart_path}")
            
            print("\n" + "="*80 + "\n")
            
        except Exception as e:
            print(f"오류 발생: {e}\n")

# 사용자 인터페이스 함수
def interactive_demo():
    """대화형 데모"""
    system = LlamaIndexSystem()
    
    print("=== 대화형 LlamaIndex 시스템 ===")
    print("질문을 입력하세요 ('exit' 입력시 종료):\n")
    
    while True:
        user_input = input("질문: ").strip()
        
        if user_input.lower() in ['exit', 'quit', '종료']:
            print("시스템을 종료합니다.")
            break
        
        if not user_input:
            continue
        
        try:
            result = system.process_query(user_input)
            
            print(f"\n답변: {result.response_text}")
            
            if isinstance(result.data, pd.DataFrame) and not result.data.empty:
                print("\n데이터:")
                print(result.data.to_string(index=False))
            
            if result.chart_path:
                print(f"\n📊 차트가 생성되었습니다: {result.chart_path}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # 데모 실행
    demo_system()
    
    # 대화형 모드 (선택적)
    run_interactive = input("\n대화형 모드를 실행하시겠습니까? (y/n): ").lower().strip()
    if run_interactive == 'y':
        interactive_demo()
