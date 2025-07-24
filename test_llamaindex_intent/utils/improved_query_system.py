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
import logging

# LlamaIndex imports
from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.query_engine import NLSQLTableQueryEngine, RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike
from sqlalchemy import create_engine

# 로컬 모듈 imports
from data_generator import SampleDataGenerator
from vector_db_manager import VectorDBManager, DocumentQueryEngine

# 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """질의 유형 분류"""
    STRUCTURED_DATA = "structured_data"
    DOCUMENT_SEARCH = "document_search"
    HYBRID = "hybrid"
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"

class VisualizationType(Enum):
    """시각화 유형"""
    TABLE = "table"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"

@dataclass
class QueryResult:
    """질의 결과 데이터 클래스"""
    answer: str
    query_type: QueryType
    visualization_type: VisualizationType
    structured_data: Optional[pd.DataFrame] = None
    document_results: Optional[List[Dict[str, Any]]] = None
    sql_query: Optional[str] = None
    chart_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DatabaseManager:
    """데이터베이스 관리 클래스 (개선됨)"""
    
    def __init__(self, db_path: str = "company_data.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.ensure_data_exists()
        
    def ensure_data_exists(self):
        """데이터베이스 존재 확인 및 생성"""
        if not Path(self.db_path).exists():
            logger.info("샘플 데이터베이스 생성 중...")
            generator = SampleDataGenerator(self.db_path)
            generator.generate_all_data()
            generator.close()
        
    def get_sql_database(self) -> SQLDatabase:
        """LlamaIndex SQLDatabase 객체 반환"""
        return SQLDatabase(self.engine)
    
    def get_table_info(self) -> Dict[str, List[str]]:
        """테이블 정보 반환"""
        try:
            tables = {}
            with self.engine.connect() as conn:
                # 테이블 목록 조회
                result = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                table_names = [row[0] for row in result]
                
                for table_name in table_names:
                    # 각 테이블의 컬럼 정보 조회
                    result = conn.execute(f"PRAGMA table_info({table_name})")
                    columns = [row[1] for row in result]  # row[1]이 컬럼명
                    tables[table_name] = columns
            
            return tables
        except Exception as e:
            logger.error(f"테이블 정보 조회 실패: {str(e)}")
            return {}

class EnhancedChartGenerator:
    """향상된 차트 생성 클래스"""
    
    def __init__(self, output_dir: str = "charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_chart(self, 
                      data: pd.DataFrame, 
                      chart_type: VisualizationType,
                      title: str = "데이터 차트",
                      x_column: str = None,
                      y_column: str = None) -> str:
        """향상된 차트 생성"""
        plt.figure(figsize=(12, 8))
        
        try:
            if chart_type == VisualizationType.LINE_CHART:
                self._create_line_chart(data, x_column, y_column)
                
            elif chart_type == VisualizationType.BAR_CHART:
                self._create_bar_chart(data, x_column, y_column)
                
            elif chart_type == VisualizationType.PIE_CHART:
                self._create_pie_chart(data, x_column, y_column)
                
            elif chart_type == VisualizationType.SCATTER_PLOT:
                self._create_scatter_plot(data, x_column, y_column)
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 파일 저장
            filename = f"chart_{abs(hash(title))}_{chart_type.value}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"차트 생성 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            plt.close()
            logger.error(f"차트 생성 실패: {str(e)}")
            return None
    
    def _create_line_chart(self, data: pd.DataFrame, x_col: str, y_col: str):
        """라인 차트 생성"""
        if x_col and y_col and x_col in data.columns and y_col in data.columns:
            plt.plot(data[x_col], data[y_col], marker='o', linewidth=2, markersize=8)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.xticks(rotation=45)
        elif 'quarter' in data.columns:
            numeric_cols = data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                plt.plot(data['quarter'], data[col], marker='o', label=col, linewidth=2, markersize=6)
            plt.xlabel('분기', fontsize=12)
            plt.ylabel('값', fontsize=12)
            plt.legend()
            plt.xticks(rotation=45)
    
    def _create_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: str):
        """바 차트 생성"""
        if x_col and y_col and x_col in data.columns and y_col in data.columns:
            bars = plt.bar(data[x_col], data[y_col], alpha=0.8, color='steelblue')
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.xticks(rotation=45)
            # 막대 위에 값 표시
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom')
        else:
            # 기본 바차트 로직
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                x_data = data.iloc[:, 0] if len(data.columns) > 0 else range(len(data))
                y_data = data[numeric_cols[0]]
                bars = plt.bar(x_data, y_data, alpha=0.8, color='steelblue')
                plt.xticks(rotation=45)
    
    def _create_pie_chart(self, data: pd.DataFrame, x_col: str, y_col: str):
        """파이 차트 생성"""
        if x_col and y_col and x_col in data.columns and y_col in data.columns:
            plt.pie(data[y_col], labels=data[x_col], autopct='%1.1f%%', startangle=90)
        elif 'product' in data.columns and 'sales' in data.columns:
            plt.pie(data['sales'], labels=data['product'], autopct='%1.1f%%', startangle=90)
        elif 'market_share' in data.columns:
            labels = data.iloc[:, 0] if len(data.columns) > 0 else [f'항목{i+1}' for i in range(len(data))]
            plt.pie(data['market_share'], labels=labels, autopct='%1.1f%%', startangle=90)
    
    def _create_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str):
        """산점도 생성"""
        if x_col and y_col and x_col in data.columns and y_col in data.columns:
            plt.scatter(data[x_col], data[y_col], alpha=0.7, s=60)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)

class IntelligentQueryClassifier:
    """지능형 질의 분류기"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def classify_query(self, query: str, table_info: Dict[str, List[str]]) -> Tuple[QueryType, Dict[str, Any]]:
        """질의 분류 및 파라미터 추출"""
        
        # 테이블 정보를 문자열로 변환
        table_info_str = "\n".join([
            f"- {table}: {', '.join(columns)}" 
            for table, columns in table_info.items()
        ])
        
        classification_prompt = f"""
다음 질문을 분석하여 질의 유형을 분류하고 관련 파라미터를 추출하세요.

사용 가능한 데이터베이스 테이블:
{table_info_str}

질문:'{query}'

질의 유형 분류 기준:
1. structured_data: SQL 데이터베이스에서 정형 데이터를 조회해야 하는 질의
2. document_search: 문서/보고서/회의록 등에서 정보를 찾아야 하는 질의
3. hybrid: 정형 데이터와 문서 검색이 모두 필요한 질의
4. general_query: 일반적인 질문이나 대화

키워드 분석:
- 정형 데이터 관련: 실적, 매출, 분기, 제품별, 공정별, 효율, 생산량 등
- 문서 검색 관련: 회의록, 보고서, 문서, 언급, 찾아줘, 요약 등
- 하이브리드: "보고서와 실적을 같이", "문서와 데이터 비교" 등

응답 형식 (JSON):
{{
    "query_type": "분류 결과",
    "confidence": 0.9,
    "parameters": {{
        "tables": ["관련 테이블명들"],
        "columns": ["관련 컬럼명들"],
        "person_names": ["언급된 인물명들"],
        "date_keywords": ["날짜 관련 키워드들"],
        "document_types": ["찾아야 할 문서 유형들"],
        "analysis_type": "요구되는 분석 유형"
    }},
    "reasoning": "분류 근거 설명"
}}
"""
        
        try:
            response = self.llm.complete(classification_prompt)
            result = json.loads(response.text)
            
            query_type = QueryType(result.get('query_type', 'general_query'))
            parameters = result.get('parameters', {})
            
            logger.info(f"질의 분류 완료: {query_type.value} (신뢰도: {result.get('confidence', 0)})")
            return query_type, parameters
            
        except Exception as e:
            logger.error(f"질의 분류 실패: {str(e)}")
            # 키워드 기반 폴백 분류
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Tuple[QueryType, Dict[str, Any]]:
        """폴백 분류 로직"""
        query_lower = query.lower()
        
        # 문서 검색 키워드
        doc_keywords = ['회의록', '보고서', '문서', '언급', '찾아', '요약']
        # 정형 데이터 키워드  
        data_keywords = ['실적', '매출', '분기', '제품', '공정', '효율', '생산']
        
        doc_score = sum(1 for kw in doc_keywords if kw in query_lower)
        data_score = sum(1 for kw in data_keywords if kw in query_lower)
        
        if doc_score > 0 and data_score > 0:
            return QueryType.HYBRID, {}
        elif doc_score > 0:
            return QueryType.DOCUMENT_SEARCH, {}
        elif data_score > 0:
            return QueryType.STRUCTURED_DATA, {}
        else:
            return QueryType.GENERAL_QUERY, {}

class SmartVisualizationDecider:
    """스마트 시각화 결정기"""
    
    def decide_visualization(self, 
                           data: pd.DataFrame, 
                           query: str, 
                           query_params: Dict[str, Any]) -> Tuple[VisualizationType, Dict[str, str]]:
        """데이터와 질의를 기반으로 적절한 시각화 방식 결정"""
        
        if data.empty:
            return VisualizationType.TABLE, {}
        
        # 컬럼 타입 분석
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 시계열 데이터 확인
        time_cols = [col for col in data.columns if any(t in col.lower() for t in ['date', 'quarter', 'month', 'year'])]
        
        # 질의 키워드 분석
        query_lower = query.lower()
        
        viz_params = {}
        
        # 시각화 타입 결정 로직
        if any(keyword in query_lower for keyword in ['추이', '변화', '트렌드', '증감']):
            if time_cols and numeric_cols:
                viz_params = {'x_column': time_cols[0], 'y_column': numeric_cols[0]}
                return VisualizationType.LINE_CHART, viz_params
        
        elif any(keyword in query_lower for keyword in ['비율', '구성', '점유율', '비중']):
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                viz_params = {'x_column': categorical_cols[0], 'y_column': numeric_cols[0]}
                return VisualizationType.PIE_CHART, viz_params
        
        elif any(keyword in query_lower for keyword in ['비교', '대비', '차이']):
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                viz_params = {'x_column': categorical_cols[0], 'y_column': numeric_cols[0]}
                return VisualizationType.BAR_CHART, viz_params
        
        elif any(keyword in query_lower for keyword in ['상관관계', '관계', '연관']):
            if len(numeric_cols) >= 2:
                viz_params = {'x_column': numeric_cols[0], 'y_column': numeric_cols[1]}
                return VisualizationType.SCATTER_PLOT, viz_params
        
        # 데이터 특성 기반 결정
        if len(data) <= 20 and len(categorical_cols) > 0 and len(numeric_cols) > 0:
            viz_params = {'x_column': categorical_cols[0], 'y_column': numeric_cols[0]}
            return VisualizationType.BAR_CHART, viz_params
        
        elif time_cols and numeric_cols:
            viz_params = {'x_column': time_cols[0], 'y_column': numeric_cols[0]}
            return VisualizationType.LINE_CHART, viz_params
        
        # 기본값: 표
        return VisualizationType.TABLE, {}

class ImprovedQuerySystem:
    """개선된 통합 질의 처리 시스템"""
    
    def __init__(self, 
                 openai_api_base: str,
                 openai_api_key: str = "fake-key",
                 model_name: str = "gpt-3.5-turbo"):
        """
        초기화
        
        Args:
            openai_api_base: OpenAI-like API 베이스 URL
            openai_api_key: API 키
            model_name: 모델명
        """
        
        # OpenAI-like LLM 설정
        self.llm = OpenAILike(
            api_base=openai_api_base,
            api_key=openai_api_key,
            model=model_name,
            temperature=0.1,
            max_tokens=2048
        )
        
        # 구성 요소 초기화
        self.db_manager = DatabaseManager()
        self.vector_db_manager = VectorDBManager()
        self.chart_generator = EnhancedChartGenerator()
        self.query_classifier = IntelligentQueryClassifier(self.llm)
        self.viz_decider = SmartVisualizationDecider()
        
        # 쿼리 엔진 설정
        self.setup_query_engines()
        self.setup_tools()
        self.setup_agent()
        
        logger.info("ImprovedQuerySystem 초기화 완료")
    
    def setup_query_engines(self):
        """향상된 쿼리 엔진 설정"""
        
        # SQL 데이터베이스 쿼리 엔진
        sql_database = self.db_manager.get_sql_database()
        self.sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=["quarterly_performance", "process_performance", "product_sales", 
                   "employee_performance", "market_analysis"],
            llm=self.llm,
            synthesize_response=True,
            verbose=True
        )
        
        # 문서 검색 쿼리 엔진
        self.doc_query_engine = self.vector_db_manager.get_query_engine()
        
        # 쿼리 엔진 도구들
        sql_tool = QueryEngineTool(
            query_engine=self.sql_query_engine,
            metadata=ToolMetadata(
                name="sql_database",
                description=(
                    "정형 데이터베이스에서 데이터를 조회할 때 사용합니다. "
                    "분기별 실적, 공정별 성과, 제품별 매출, 직원 성과, 시장 분석 등의 "
                    "수치 데이터와 통계 정보를 조회할 수 있습니다."
                )
            )
        )
        
        doc_tool = QueryEngineTool(
            query_engine=self.doc_query_engine,
            metadata=ToolMetadata(
                name="document_search",
                description=(
                    "회의록, 보고서, 계획서 등의 문서에서 정보를 검색할 때 사용합니다. "
                    "특정 인물의 언급, 프로젝트 내용, 의사결정 과정 등을 찾을 수 있습니다."
                )
            )
        )
        
        # 라우터 쿼리 엔진
        self.router_query_engine = RouterQueryEngine(
            selector=self.llm,
            query_engine_tools=[sql_tool, doc_tool],
            verbose=True
        )
    
    def setup_tools(self):
        """에이전트용 고급 도구 설정"""
        
        def analyze_structured_data(sql_query: str, analysis_type: str = "basic") -> str:
            """정형 데이터 분석 도구"""
            try:
                data = pd.read_sql(sql_query, self.db_manager.engine)
                
                if data.empty:
                    return "조회된 데이터가 없습니다."
                
                analysis_result = {
                    "data_summary": {
                        "row_count": len(data),
                        "column_count": len(data.columns),
                        "columns": data.columns.tolist()
                    },
                    "data": data.to_dict('records')
                }
                
                # 기본 통계 분석
                if analysis_type == "statistical":
                    numeric_data = data.select_dtypes(include=['number'])
                    if not numeric_data.empty:
                        analysis_result["statistics"] = numeric_data.describe().to_dict()
                
                return json.dumps(analysis_result, ensure_ascii=False, default=str)
                
            except Exception as e:
                return f"데이터 분석 중 오류 발생: {str(e)}"
        
        def search_documents(query: str, max_results: int = 5) -> str:
            """문서 검색 도구"""
            try:
                results = self.doc_query_engine.query(query)
                
                # 검색 결과 정리
                search_results = {
                    "query": query,
                    "answer": str(results),
                    "source_documents": []
                }
                
                # 소스 문서 정보 추출 (메타데이터가 있는 경우)
                if hasattr(results, 'source_nodes'):
                    for node in results.source_nodes[:max_results]:
                        doc_info = {
                            "content": node.text[:500] + "..." if len(node.text) > 500 else node.text,
                            "metadata": node.metadata if hasattr(node, 'metadata') else {}
                        }
                        search_results["source_documents"].append(doc_info)
                
                return json.dumps(search_results, ensure_ascii=False)
                
            except Exception as e:
                return f"문서 검색 중 오류 발생: {str(e)}"
        
        def create_visualization(data_json: str, chart_type: str, title: str, 
                               x_column: str = None, y_column: str = None) -> str:
            """향상된 시각화 생성 도구"""
            try:
                data = pd.read_json(data_json)
                viz_type = VisualizationType(chart_type)
                
                chart_path = self.chart_generator.generate_chart(
                    data, viz_type, title, x_column, y_column
                )
                
                if chart_path:
                    return f"시각화가 성공적으로 생성되었습니다: {chart_path}"
                else:
                    return "시각화 생성에 실패했습니다."
                    
            except Exception as e:
                return f"시각화 생성 중 오류 발생: {str(e)}"
        
        def hybrid_analysis(structured_query: str, document_query: str) -> str:
            """하이브리드 분석 도구 - 정형 데이터와 문서를 함께 분석"""
            try:
                # 정형 데이터 조회
                structured_data = pd.read_sql(structured_query, self.db_manager.engine)
                
                # 문서 검색
                doc_results = self.doc_query_engine.query(document_query)
                
                # 결과 통합
                hybrid_result = {
                    "structured_data": {
                        "summary": f"{len(structured_data)}개의 레코드 조회됨",
                        "data": structured_data.to_dict('records') if not structured_data.empty else []
                    },
                    "document_results": {
                        "answer": str(doc_results),
                        "source_count": len(doc_results.source_nodes) if hasattr(doc_results, 'source_nodes') else 0
                    }
                }
                
                return json.dumps(hybrid_result, ensure_ascii=False, default=str)
                
            except Exception as e:
                return f"하이브리드 분석 중 오류 발생: {str(e)}"
        
        # 도구 등록
        self.structured_data_tool = FunctionTool.from_defaults(
            fn=analyze_structured_data,
            name="analyze_structured_data",
            description="SQL 쿼리를 실행하여 정형 데이터를 조회하고 분석합니다."
        )
        
        self.document_search_tool = FunctionTool.from_defaults(
            fn=search_documents,
            name="search_documents", 
            description="문서 데이터베이스에서 관련 정보를 검색합니다."
        )
        
        self.visualization_tool = FunctionTool.from_defaults(
            fn=create_visualization,
            name="create_visualization",
            description="데이터를 받아 차트나 그래프로 시각화합니다."
        )
        
        self.hybrid_tool = FunctionTool.from_defaults(
            fn=hybrid_analysis,
            name="hybrid_analysis",
            description="정형 데이터와 문서 검색을 함께 수행하여 종합적인 분석을 제공합니다."
        )
    
    def setup_agent(self):
        """향상된 ReAct 에이전트 설정"""
        
        tools = [
            self.structured_data_tool,
            self.document_search_tool, 
            self.visualization_tool,
            self.hybrid_tool
        ]
        
        system_prompt = """
당신은 회사 데이터 분석 전문가입니다. 사용자의 질문을 정확히 분석하여 적절한 도구를 사용해 답변하세요.

사용 가능한 도구:
1. analyze_structured_data: SQL 데이터베이스에서 정형 데이터 조회 및 분석
2. search_documents: 회의록, 보고서 등 문서에서 정보 검색
3. create_visualization: 데이터를 차트로 시각화
4. hybrid_analysis: 정형 데이터와 문서를 함께 분석

질의 처리 단계:
1. 사용자 질문의 의도와 유형 파악
2. 필요한 데이터 조회 (정형/비정형)
3. 데이터 분석 및 인사이트 도출
4. 적절한 시각화 생성 (필요한 경우)
5. 종합적인 답변 제공

답변 시 주의사항:
- 한국어로 친근하고 전문적으로 답변
- 데이터의 한계나 불확실성 명시
- 구체적인 수치와 근거 제시
- 시각화가 도움이 되는 경우 차트 생성

데이터베이스 테이블 정보:
- quarterly_performance: 분기별 실적 데이터 
- process_performance: 공정별 성과 데이터
- product_sales: 제품별 매출 데이터
- employee_performance: 직원 성과 데이터
- market_analysis: 시장 분석 데이터
"""
        
        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
            max_iterations=10
        )
        
        logger.info("ReAct 에이전트 설정 완료")
    
    def process_query(self, user_query: str) -> QueryResult:
        """사용자 질의 전체 처리 파이프라인"""
        
        logger.info(f"질의 처리 시작: {user_query}")
        
        try:
            # 1. 질의 분류 및 파라미터 추출
            table_info = self.db_manager.get_table_info()
            query_type, parameters = self.query_classifier.classify_query(user_query, table_info)
            
            logger.info(f"질의 분류 결과: {query_type.value}")
            
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
            
            # 3. 에이전트를 통한 복합 질의 처리
            agent_response = self.agent.chat(user_query)
            
            # 4. 응답에서 데이터 추출 시도
            structured_data = self._extract_structured_data_from_response(str(agent_response))
            document_results = self._extract_document_results_from_response(str(agent_response))
            
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
            
        except Exception as e:
            logger.error(f"질의 처리 중 오류 발생: {str(e)}")
            return QueryResult(
                answer=f"질의 처리 중 오류가 발생했습니다: {str(e)}",
                query_type=QueryType.UNKNOWN,
                visualization_type=VisualizationType.TABLE,
                metadata={"error": str(e)}
            )
    
    def _extract_structured_data_from_response(self, response: str) -> Optional[pd.DataFrame]:
        """응답에서 정형 데이터 추출"""
        try:
            # JSON 형태의 데이터를 찾아서 DataFrame으로 변환
            import re
            json_pattern = r'\{[^{}]*"data"[^{}]*\[[^\]]*\][^{}]*\}'
            matches = re.findall(json_pattern, response)
            
            for match in matches:
                try:
                    data_dict = json.loads(match)
                    if 'data' in data_dict and isinstance(data_dict['data'], list):
                        return pd.DataFrame(data_dict['data'])
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"정형 데이터 추출 실패: {str(e)}")
            return None
    
    def _extract_document_results_from_response(self, response: str) -> Optional[List[Dict[str, Any]]]:
        """응답에서 문서 검색 결과 추출"""
        try:
            # 문서 검색 결과 패턴을 찾아서 추출
            import re
            doc_pattern = r'"source_documents":\s*\[[^\]]*\]'
            matches = re.findall(doc_pattern, response)
            
            for match in matches:
                try:
                    # 간단한 파싱으로 문서 결과 추출
                    # 실제로는 더 정교한 파싱이 필요할 수 있음
                    return [{"content": "문서 검색 결과가 포함되어 있습니다."}]
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"문서 결과 추출 실패: {str(e)}")
            return None
    
    def format_response(self, result: QueryResult) -> Dict[str, Any]:
        """사용자 친화적인 응답 포맷팅"""
        
        formatted_response = {
            "answer": result.answer,
            "query_type": result.query_type.value,
            "visualization_type": result.visualization_type.value,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 정형 데이터 추가
        if result.structured_data is not None and not result.structured_data.empty:
            formatted_response["structured_data"] = {
                "summary": f"{len(result.structured_data)}개 레코드, {len(result.structured_data.columns)}개 컬럼",
                "data": result.structured_data.to_dict('records'),
                "columns": result.structured_data.columns.tolist()
            }
        
        # 문서 검색 결과 추가
        if result.document_results:
            formatted_response["document_results"] = result.document_results
        
        # 차트 경로 추가
        if result.chart_path:
            formatted_response["visualization"] = {
                "type": result.visualization_type.value,
                "chart_path": result.chart_path,
                "description": f"{result.visualization_type.value} 형태로 시각화되었습니다."
            }
        
        # SQL 쿼리 정보 추가 (디버깅용)
        if result.sql_query:
            formatted_response["sql_query"] = result.sql_query
        
        # 메타데이터 추가
        if result.metadata:
            formatted_response["metadata"] = result.metadata
            
        return formatted_response
    
    def get_sample_queries(self) -> List[str]:
        """시스템 테스트용 샘플 질의 목록"""
        return [
            # 정형 데이터 질의
            "공정별 실적을 찾아줘",
            "최근 4분기 매출 추이를 보여줘",
            "제품별 매출 비교 분석해줘",
            "직원 성과 상위 10명을 알려줘",
            "시장 점유율 현황을 차트로 보여줘",
            
            # 문서 검색 질의  
            "회의록 중 김영철씨가 언급된 것을 찾아줘",
            "최근 보고서에서 품질 개선 관련 내용을 요약해줘",
            "프로젝트 계획서에서 예산 관련 내용을 찾아줘",
            "이사회 회의록에서 신제품 개발 논의 내용을 보여줘",
            "월간 보고서에서 리스크 요인들을 정리해줘",
            
            # 하이브리드 질의
            "4월 월간보고서와 결산 실적을 같이 보여줘",
            "공정 효율성 데이터와 관련 회의록 내용을 비교해줘",
            "매출 실적과 시장 분석 보고서를 함께 분석해줘",
            "직원 성과 데이터와 인사 평가 회의록을 연결해줘",
            "제품 매출 데이터와 고객 피드백 보고서를 같이 보여줘",
            
            # 일반 질의
            "회사의 전반적인 현황을 설명해줘",
            "데이터 분석에 도움이 필요해",
            "어떤 종류의 질문을 할 수 있나요?",
        ]
    
    def health_check(self) -> Dict[str, str]:
        """시스템 상태 확인"""
        status = {}
        
        try:
            # 데이터베이스 연결 확인
            tables = self.db_manager.get_table_info()
            status["database"] = f"OK - {len(tables)}개 테이블"
        except Exception as e:
            status["database"] = f"ERROR - {str(e)}"
        
        try:
            # 벡터 DB 확인
            test_query = self.doc_query_engine.query("테스트")
            status["vector_db"] = "OK"
        except Exception as e:
            status["vector_db"] = f"ERROR - {str(e)}"
        
        try:
            # LLM 연결 확인
            test_response = self.llm.complete("안녕하세요")
            status["llm"] = "OK"
        except Exception as e:
            status["llm"] = f"ERROR - {str(e)}"
        
        try:
            # 차트 생성 디렉토리 확인
            self.chart_generator.output_dir.mkdir(exist_ok=True)
            status["chart_generator"] = "OK"
        except Exception as e:
            status["chart_generator"] = f"ERROR - {str(e)}"
        
        return status


def main():
    """시스템 사용 예시 및 테스트"""
    
    print("=== 개선된 통합 질의 처리 시스템 ===")
    print("시스템 구성 요소:")
    print("1. IntelligentQueryClassifier - 지능형 질의 분류")
    print("2. DatabaseManager - 데이터베이스 관리")  
    print("3. VectorDBManager - 문서 벡터 데이터베이스")
    print("4. EnhancedChartGenerator - 향상된 차트 생성")
    print("5. SmartVisualizationDecider - 스마트 시각화 결정")
    print("6. ReActAgent - 복합 질의 처리 에이전트")
    print("-" * 50)
    
    # 시스템 초기화 예시 (실제 사용시 적절한 API 설정 필요)
    print("\n시스템 초기화 방법:")
    print("""
# 예시 코드
system = ImprovedQuerySystem(
    openai_api_base="http://localhost:1234/v1",  # Local LLM API
    openai_api_key="fake-key",
    model_name="gpt-3.5-turbo"
)

# 상태 확인
health_status = system.health_check()
print("시스템 상태:", health_status)

# 질의 처리
result = system.process_query("공정별 실적을 차트로 보여줘")
formatted_result = system.format_response(result)
print("결과:", formatted_result)
""")
    
    print("\n지원되는 질의 유형:")
    sample_queries = [
        "공정별 실적을 찾아줘",
        "회의록 중 김영철씨가 언급된 것을 찾아줘", 
        "4월 월간보고서와 결산 실적을 같이 보여줘",
        "최근 4분기 매출 추이를 차트로 보여줘",
        "제품별 시장 점유율을 파이차트로 만들어줘"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")
    
    print("\n주요 개선사항:")
    print("- OpenAI-like API 지원으로 다양한 LLM 사용 가능")
    print("- 지능형 질의 분류로 정확한 처리 경로 결정")
    print("- 하이브리드 질의 지원 (정형 + 비정형 데이터)")
    print("- 스마트 시각화로 질의에 맞는 차트 자동 생성")
    print("- ReAct 에이전트로 복잡한 다단계 질의 처리")
    print("- 풍부한 오류 처리 및 로깅")
    
    print("\n필요한 추가 파일:")
    print("- data_generator.py: 샘플 데이터 생성")
    print("- vector_db_manager.py: ChromaDB 기반 문서 검색")


if __name__ == "__main__":
    main()
