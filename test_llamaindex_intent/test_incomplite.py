import os
import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import tempfile
from pathlib import Path
import uuid
from datetime import datetime

# LlamaIndex imports
from llama_index.core import SQLDatabase, VectorStoreIndex, Document, ServiceContext
from llama_index.core.query_engine import NLSQLTableQueryEngine, RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Date
from sqlalchemy.orm import sessionmaker

# 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")

class QueryType(Enum):
    """질의 유형 분류"""
    STRUCTURED_DATA = "structured_data"  # 공정별 실적 등
    DOCUMENT_SEARCH = "document_search"  # 회의록, 보고서 검색
    HYBRID_QUERY = "hybrid_query"        # 복합 질의 (보고서 + 실적)
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"

class VisualizationType(Enum):
    """시각화 유형"""
    TABLE = "table"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    DOCUMENT_SUMMARY = "document_summary"
    COMBINED_VIEW = "combined_view"

@dataclass
class QueryResult:
    """질의 결과 데이터 클래스"""
    data: Optional[pd.DataFrame] = None
    documents: Optional[List[Dict]] = None
    query_type: Optional[QueryType] = None
    visualization_type: Optional[VisualizationType] = None
    sql_query: Optional[str] = None
    chart_path: Optional[str] = None
    summary: Optional[str] = None
    combined_results: Optional[Dict] = None

class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "enterprise_data.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.setup_sample_data()
        
    def setup_sample_data(self):
        """기업 환경에 맞는 샘플 데이터 생성"""
        
        # 1. 공정별 실적 데이터
        process_performance = {
            'process_name': ['조립공정', '도장공정', '검사공정', '포장공정', '출하공정'] * 4,
            'month': ['2024-01', '2024-01', '2024-01', '2024-01', '2024-01',
                     '2024-02', '2024-02', '2024-02', '2024-02', '2024-02',
                     '2024-03', '2024-03', '2024-03', '2024-03', '2024-03',
                     '2024-04', '2024-04', '2024-04', '2024-04', '2024-04'],
            'production_volume': [1200, 1150, 1180, 1200, 1190,
                                1250, 1200, 1220, 1240, 1230,
                                1300, 1280, 1290, 1300, 1295,
                                1350, 1320, 1340, 1350, 1345],
            'efficiency_rate': [92.5, 89.3, 94.1, 96.2, 93.8,
                              94.2, 91.5, 95.8, 97.1, 94.9,
                              95.8, 93.2, 96.5, 98.2, 96.1,
                              96.5, 94.8, 97.2, 98.8, 97.3],
            'defect_rate': [2.1, 3.2, 1.8, 1.2, 2.5,
                           1.8, 2.9, 1.5, 1.0, 2.1,
                           1.5, 2.3, 1.2, 0.8, 1.8,
                           1.2, 1.9, 0.9, 0.6, 1.5]
        }
        
        pd.DataFrame(process_performance).to_sql('process_performance', self.engine, if_exists='replace', index=False)
        
        # 2. 분기별 실적 데이터
        quarterly_data = {
            'quarter': ['2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q1'],
            'revenue': [1000, 1200, 1100, 1400, 1300],
            'operating_profit': [100, 150, 120, 180, 160],
            'company': ['우리회사'] * 5
        }
        
        pd.DataFrame(quarterly_data).to_sql('quarterly_performance', self.engine, if_exists='replace', index=False)
        
        # 3. 월간 실적 데이터 (4월 결산용)
        monthly_financial = {
            'month': ['2024-01', '2024-02', '2024-03', '2024-04'],
            'revenue': [420, 450, 430, 480],
            'cost': [320, 340, 325, 360],
            'operating_profit': [52, 58, 54, 64],
            'net_profit': [38, 42, 39, 46]
        }
        
        pd.DataFrame(monthly_financial).to_sql('monthly_financial', self.engine, if_exists='replace', index=False)
        
    def get_sql_database(self) -> SQLDatabase:
        """LlamaIndex SQLDatabase 객체 반환"""
        return SQLDatabase(self.engine)

class DocumentManager:
    """문서 관리 클래스"""
    
    def __init__(self):
        self.documents = []
        self.setup_sample_documents()
        
    def setup_sample_documents(self):
        """샘플 문서 데이터 생성"""
        
        # 회의록 샘플
        meeting_records = [
            {
                "id": "meeting_001",
                "title": "2024년 1분기 생산회의",
                "date": "2024-04-01",
                "type": "meeting_record",
                "content": """
                참석자: 김영철(생산팀장), 박미영(품질팀장), 이준호(기획팀장)
                
                주요 안건:
                1. 1분기 생산실적 검토
                   - 김영철 팀장: 조립공정 효율성이 95% 달성, 목표 대비 양호
                   - 도장공정에서 일부 품질 이슈 발생, 개선 필요
                
                2. 2분기 생산계획
                   - 신제품 라인 증설 예정
                   - 김영철 팀장 제안: 자동화 설비 도입으로 효율성 향상
                
                결론: 김영철 팀장 주도로 생산성 개선 TF 구성
                """
            },
            {
                "id": "meeting_002", 
                "title": "품질개선 회의",
                "date": "2024-04-15",
                "type": "meeting_record",
                "content": """
                참석자: 박미영(품질팀장), 김영철(생산팀장), 최수진(기술팀장)
                
                주요 논의사항:
                1. 불량률 감소 방안
                   - 김영철: 공정 표준화로 불량률 2% → 1.5%로 개선
                   - 검사공정 강화 필요
                
                2. 고객 클레임 대응
                   - 즉시 대응팀 구성
                   - 김영철 팀장이 현장 점검 주도
                
                향후 계획: 월 1회 품질 점검 체계 구축
                """
            },
            {
                "id": "meeting_003",
                "title": "경영진 월례회의", 
                "date": "2024-04-20",
                "type": "meeting_record",
                "content": """
                참석자: 대표이사, 김영철(생산담당), 이민수(영업담당), 박정희(재무담당)
                
                보고사항:
                1. 4월 실적 보고 (박정희 재무담당)
                   - 매출 480억원, 전월 대비 11.6% 증가
                   - 영업이익 64억원으로 양호
                
                2. 생산현황 (김영철 생산담당)
                   - 모든 공정에서 목표 달성
                   - 신규 설비 도입 효과로 생산성 향상
                
                결정사항: 김영철 담당 주도로 하반기 증설 계획 수립
                """
            }
        ]
        
        # 월간보고서 샘플
        monthly_reports = [
            {
                "id": "report_001",
                "title": "2024년 4월 월간보고서",
                "date": "2024-04-30", 
                "type": "monthly_report",
                "content": """
                # 2024년 4월 월간보고서
                
                ## 경영실적 요약
                - 매출액: 480억원 (전월 대비 +11.6%, 전년 동월 대비 +8.2%)
                - 영업이익: 64억원 (영업이익률 13.3%)
                - 순이익: 46억원
                
                ## 주요 성과
                1. 생산부문
                   - 전 공정 목표 달성률 98% 이상
                   - 불량률 1.2%로 전월 대비 개선
                   - 신규 자동화 라인 가동 시작
                
                2. 영업부문  
                   - 신규 고객 3사 확보
                   - 기존 고객 추가 수주 증가
                
                3. 기술부문
                   - R&D 투자 확대
                   - 특허 출원 2건 완료
                
                ## 향후 계획
                - 5월 생산능력 10% 증설 예정
                - 신제품 출시 준비
                - 해외 시장 진출 검토
                """
            }
        ]
        
        self.documents = meeting_records + monthly_reports
        
    def search_documents(self, query: str, doc_type: str = None) -> List[Dict]:
        """문서 검색"""
        results = []
        
        for doc in self.documents:
            # 문서 타입 필터
            if doc_type and doc['type'] != doc_type:
                continue
                
            # 키워드 검색
            if any(keyword in doc['content'] for keyword in query.split()):
                results.append(doc)
                
        return results
        
    def get_documents_by_person(self, person_name: str) -> List[Dict]:
        """특정 인물이 언급된 문서 검색"""
        results = []
        
        for doc in self.documents:
            if person_name in doc['content']:
                results.append(doc)
                
        return results

class ChartGenerator:
    """차트 생성 클래스"""
    
    def __init__(self, output_dir: str = "charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_chart(self, data: pd.DataFrame, chart_type: VisualizationType, 
                      title: str = "데이터 차트") -> str:
        """차트 생성 및 파일 경로 반환"""
        plt.figure(figsize=(12, 8))
        
        if chart_type == VisualizationType.LINE_CHART:
            if 'month' in data.columns and 'efficiency_rate' in data.columns:
                # 공정별 효율성 추이
                for process in data['process_name'].unique():
                    process_data = data[data['process_name'] == process]
                    plt.plot(process_data['month'], process_data['efficiency_rate'], 
                           marker='o', label=process)
                plt.xlabel('월')
                plt.ylabel('효율성 (%)')
                plt.legend()
                plt.xticks(rotation=45)
                
            elif 'quarter' in data.columns:
                plt.plot(data['quarter'], data['revenue'], marker='o', label='매출')
                if 'operating_profit' in data.columns:
                    plt.plot(data['quarter'], data['operating_profit'], marker='s', label='영업이익')
                plt.xlabel('분기')
                plt.ylabel('금액 (억원)')
                plt.legend()
                plt.xticks(rotation=45)
            
        elif chart_type == VisualizationType.BAR_CHART:
            if 'process_name' in data.columns and 'production_volume' in data.columns:
                # 최신 월 기준 공정별 생산량
                latest_month = data['month'].max()
                latest_data = data[data['month'] == latest_month]
                plt.bar(latest_data['process_name'], latest_data['production_volume'])
                plt.xlabel('공정명')
                plt.ylabel('생산량')
                plt.xticks(rotation=45)
                
            elif 'month' in data.columns and 'revenue' in data.columns:
                plt.bar(data['month'], data['revenue'])
                plt.xlabel('월')
                plt.ylabel('매출 (억원)')
                plt.xticks(rotation=45)
                
        elif chart_type == VisualizationType.PIE_CHART:
            if 'process_name' in data.columns and 'production_volume' in data.columns:
                latest_month = data['month'].max()
                latest_data = data[data['month'] == latest_month]
                plt.pie(latest_data['production_volume'], labels=latest_data['process_name'], 
                       autopct='%1.1f%%')
        
        plt.title(title)
        plt.tight_layout()
        
        # 파일 저장
        filename = f"chart_{uuid.uuid4().hex[:8]}_{chart_type.value}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)

class QueryClassifier:
    """질의 의도 분류 클래스"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def classify_query(self, query: str) -> Tuple[QueryType, Dict[str, Any]]:
        """질의 분류 및 파라미터 추출"""
        classification_prompt = f"""
        다음 질문을 분석하여 질의 유형을 분류하고 파라미터를 추출하세요.
        
        질문: "{query}"
        
        분류 기준:
        1. structured_data: 공정별 실적, 분기별 실적 등 데이터베이스 조회가 필요한 정형 데이터 질의
        2. document_search: 회의록, 보고서에서 특정 내용이나 인물을 검색하는 질의
        3. hybrid_query: 월간보고서와 실적 데이터를 함께 보여달라는 등의 복합 질의
        4. general_query: 일반적인 질문이나 대화
        
        응답 형식 (JSON):
        {{
            "query_type": "structured_data|document_search|hybrid_query|general_query",
            "parameters": {{
                "keywords": ["추출된 키워드들"],
                "person_name": "언급된 인물명 (있는 경우)",
                "time_period": "시간 범위",
                "data_type": "요청된 데이터 유형",
                "document_type": "문서 유형 (meeting_record, monthly_report 등)"
            }}
        }}
        """
        
        response = self.llm.complete(classification_prompt)
        
        try:
            result = json.loads(response.text)
            query_type = QueryType(result.get('query_type', 'general_query'))
            parameters = result.get('parameters', {})
            return query_type, parameters
        except Exception as e:
            print(f"Classification error: {e}")
            return QueryType.UNKNOWN, {}

class VisualizationDecider:
    """시각화 방식 결정 클래스"""
    
    def decide_visualization(self, data: pd.DataFrame, query: str, query_type: QueryType) -> VisualizationType:
        """데이터와 질의를 기반으로 적절한 시각화 방식 결정"""
        
        if query_type == QueryType.DOCUMENT_SEARCH:
            return VisualizationType.DOCUMENT_SUMMARY
        elif query_type == QueryType.HYBRID_QUERY:
            return VisualizationType.COMBINED_VIEW
            
        if data.empty:
            return VisualizationType.TABLE
            
        # 공정별 데이터인 경우
        if 'process_name' in data.columns:
            if '효율' in query or '추이' in query:
                return VisualizationType.LINE_CHART
            elif '비교' in query or '현황' in query:
                return VisualizationType.BAR_CHART
            elif '구성' in query or '비율' in query:
                return VisualizationType.PIE_CHART
                
        # 시계열 데이터인 경우
        if 'month' in data.columns or 'quarter' in data.columns:
            if '추이' in query or '변화' in query or '트렌드' in query:
                return VisualizationType.LINE_CHART
            else:
                return VisualizationType.BAR_CHART
                
        return VisualizationType.TABLE

class EnterpriseQuerySystem:
    """기업용 통합 질의 처리 시스템"""
    
    def __init__(self, openai_api_key: str):
        # OpenAI LLM 설정
        os.environ['OPENAI_API_KEY'] = openai_api_key
        self.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
        
        # 구성 요소 초기화
        self.db_manager = DatabaseManager()
        self.doc_manager = DocumentManager()
        self.chart_generator = ChartGenerator()
        self.query_classifier = QueryClassifier(self.llm)
        self.viz_decider = VisualizationDecider()
        
        # LlamaIndex 구성 요소 설정
        self.setup_query_engines()
        self.setup_tools()
        self.setup_agent()
        
    def setup_query_engines(self):
        """쿼리 엔진 설정"""
        sql_database = self.db_manager.get_sql_database()
        
        # SQL 쿼리 엔진 (정형 데이터용)
        self.sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=["process_performance", "quarterly_performance", "monthly_financial"],
            llm=self.llm,
            synthesize_response=True
        )
        
        # 문서 검색 엔진
        documents = [Document(text=doc['content'], metadata=doc) for doc in self.doc_manager.documents]
        self.doc_index = VectorStoreIndex.from_documents(documents)
        self.doc_query_engine = self.doc_index.as_query_engine(llm=self.llm)
        
        # 라우터 쿼리 엔진용 도구들
        sql_tool = QueryEngineTool(
            query_engine=self.sql_query_engine,
            metadata=ToolMetadata(
                name="structured_data_query",
                description="공정별 실적, 분기별 실적, 월간 재무 데이터 등 정형 데이터베이스 조회에 사용"
            )
        )
        
        doc_tool = QueryEngineTool(
            query_engine=self.doc_query_engine,
            metadata=ToolMetadata(
                name="document_search",
                description="회의록, 보고서 등 문서에서 특정 내용이나 인물 검색에 사용"
            )
        )
        
        self.router_query_engine = RouterQueryEngine(
            selector=self.llm,
            query_engine_tools=[sql_tool, doc_tool]
        )
    
    def setup_tools(self):
        """에이전트용 도구 설정"""
        
        def search_process_performance(process_name: str = None, month: str = None) -> str:
            """공정별 실적 검색 도구"""
            try:
                query = "SELECT * FROM process_performance"
                conditions = []
                
                if process_name:
                    conditions.append(f"process_name LIKE '%{process_name}%'")
                if month:
                    conditions.append(f"month = '{month}'")
                    
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                    
                df = pd.read_sql(query, self.db_manager.engine)
                return df.to_json(orient='records', force_ascii=False)
            except Exception as e:
                return f"공정 실적 조회 오류: {str(e)}"
        
        def search_person_in_documents(person_name: str) -> str:
            """문서에서 특정 인물 검색 도구"""
            try:
                docs = self.doc_manager.get_documents_by_person(person_name)
                results = []
                for doc in docs:
                    results.append({
                        'title': doc['title'],
                        'date': doc['date'],
                        'type': doc['type'],
                        'summary': doc['content'][:200] + "..."
                    })
                return json.dumps(results, ensure_ascii=False, indent=2)
            except Exception as e:
                return f"문서 검색 오류: {str(e)}"
        
        def get_monthly_report_and_financial(month: str) -> str:
            """월간보고서와 재무실적 통합 조회"""
            try:
                # 월간 보고서 검색
                reports = [doc for doc in self.doc_manager.documents 
                          if doc['type'] == 'monthly_report' and month in doc['date']]
                
                # 재무 데이터 검색
                financial_query = f"SELECT * FROM monthly_financial WHERE month LIKE '%{month}%'"
                financial_df = pd.read_sql(financial_query, self.db_manager.engine)
                
                result = {
                    'monthly_report': reports,
                    'financial_data': financial_df.to_dict('records') if not financial_df.empty else []
                }
                
                return json.dumps(result, ensure_ascii=False, indent=2)
            except Exception as e:
                return f"통합 조회 오류: {str(e)}"
        
        def generate_chart_tool(data_json: str, chart_type: str, title: str) -> str:
            """차트 생성 도구"""
            try:
                data = pd.read_json(data_json)
                viz_type = VisualizationType(chart_type)
                chart_path = self.chart_generator.generate_chart(data, viz_type, title)
                return f"차트가 생성되었습니다: {chart_path}"
            except Exception as e:
                return f"차트 생성 중 오류 발생: {str(e)}"
        
        # 도구 등록
        self.process_tool = FunctionTool.from_defaults(
            fn=search_process_performance,
            name="search_process_performance",
            description="공정별 실적 데이터를 검색합니다. process_name(공정명), month(월) 파라미터 사용 가능"
        )
        
        self.person_search_tool = FunctionTool.from_defaults(
            fn=search_person_in_documents,
            name="search_person_in_documents", 
            description="회의록이나 보고서에서 특정 인물이 언급된 문서를 검색합니다."
        )
        
        self.hybrid_query_tool = FunctionTool.from_defaults(
            fn=get_monthly_report_and_financial,
            name="get_monthly_report_and_financial",
            description="월간보고서와 해당 월의 재무실적 데이터를 함께 조회합니다."
        )
        
        self.chart_tool = FunctionTool.from_defaults(
            fn=generate_chart_tool,
            name="generate_chart",
            description="데이터를 받아 차트를 생성합니다."
        )
    
    def setup_agent(self):
        """ReAct 에이전트 설정"""
        tools = [
            self.process_tool,
            self.person_search_tool, 
            self.hybrid_query_tool,
            self.chart_tool
        ]
        
        system_prompt = """
        당신은 기업 데이터 분석 전문가입니다. 사용자의 질문을 분석하여 적절한 도구를 사용해 답변하세요.
        
        사용 가능한 도구:
        1. search_process_performance: 공정별 실적 데이터 검색
        2. search_person_in_documents: 회의록/보고서에서 특정 인물 검색  
        3. get_monthly_report_and_financial: 월간보고서와 재무실적 통합 조회
        4. generate_chart: 데이터 차트 생성
        
        질의 유형별 처리 방법:
        - "공정별 실적": search_process_performance 사용
        - "김영철이 언급된 회의록": search_person_in_documents 사용
        - "4월 월간보고서와 결산 실적": get_monthly_report_and_financial 사용
        
        차트가 필요한 경우 generate_chart도 함께 사용하세요.
        한국어로 친근하고 전문적으로 답변하세요.
        """
        
        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True
        )
    
    def process_query(self, user_query: str) -> QueryResult:
        """사용자 질의 전체 처리 파이프라인"""
        
        try:
            # 1. 의도 분류 및 파라미터 추출
            query_type, parameters = self.query_classifier.classify_query(user_query)
            
            # 2. 질의 유형별 처리
            if query_type == QueryType.STRUCTURED_DATA:
                return self._process_structured_query(user_query, parameters)
            elif query_type == QueryType.DOCUMENT_SEARCH:
                return self._process_document_query(user_query, parameters)
            elif query_type == QueryType.HYBRID_QUERY:
                return self._process_hybrid_query(user_query, parameters)
            else:
                # 일반 질의
                response = self.llm.complete(user_query)
                return QueryResult(
                    query_type=query_type,
                    visualization_type=VisualizationType.DOCUMENT_SUMMARY,
                    summary=response.text
                )
                
        except Exception as e:
            return QueryResult(
                query_type=QueryType.UNKNOWN,
                visualization_type=VisualizationType.TABLE,
                summary=f"처리 중 오류가 발생했습니다: {str(e)}"
            )
    
    def _process_structured_query(self, query: str, parameters: Dict) -> QueryResult:
        """정형 데이터 질의 처리"""
        
        # 공정별 실적 질의인지 확인
        if '공정' in query:
            response = self.agent.chat(query)
            
            # 데이터 추출 (실제로는 agent response에서 추출)
            df = pd.read_sql("SELECT * FROM process_performance", self.db_manager.engine
