import os
import json
import argparse
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
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
        if not Path(self.db_path).exists():
            logger.info("샘플 데이터베이스 생성 중...")
            generator = SampleDataGenerator(self.db_path)
            generator.generate_all_data()
            generator.close()

    def get_sql_database(self) -> SQLDatabase:
        return SQLDatabase(self.engine)

    def get_table_info(self) -> Dict[str, List[str]]:
        try:
            tables: Dict[str, List[str]] = {}
            with self.engine.connect() as conn:
                result = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                table_names = [row[0] for row in result]
                for table_name in table_names:
                    info = conn.execute(f"PRAGMA table_info({table_name})")
                    columns = [row[1] for row in info]
                    tables[table_name] = columns
            return tables
        except Exception as e:
            logger.error(f"테이블 정보 조회 실패: {e}")
            return {}

class EnhancedChartGenerator:
    """향상된 차트 생성 클래스"""
    def __init__(self, output_dir: str = "charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    def generate_chart(self, data: pd.DataFrame, chart_type: VisualizationType,
                       title: str = "데이터 차트", x_column: str = None, y_column: str = None) -> str:
        plt.figure(figsize=(12, 8))
        try:
            if chart_type == VisualizationType.LINE_CHART:
                if x_column and y_column:
                    plt.plot(data[x_column], data[y_column], marker='o', linewidth=2)
                else:
                    for col in data.select_dtypes(include='number'):
                        plt.plot(data[data.columns[0]], data[col], marker='o', label=col)
                    plt.legend()
            elif chart_type == VisualizationType.BAR_CHART:
                if x_column and y_column:
                    plt.bar(data[x_column], data[y_column])
                else:
                    data.plot(kind='bar', x=data.columns[0])
            elif chart_type == VisualizationType.PIE_CHART:
                if x_column and y_column:
                    plt.pie(data[y_column], labels=data[x_column], autopct='%1.1f%%')
            elif chart_type == VisualizationType.SCATTER_PLOT:
                if x_column and y_column:
                    plt.scatter(data[x_column], data[y_column])
            plt.title(title)
            plt.tight_layout()
            filename = f"chart_{abs(hash(title))}_{chart_type.value}.png"
            path = self.output_dir / filename
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"차트 생성 완료: {path}")
            return str(path)
        except Exception as e:
            plt.close()
            logger.error(f"차트 생성 실패: {e}")
            return None

class IntelligentQueryClassifier:
    """지능형 질의 분류기"""
    def __init__(self, llm):
        self.llm = llm
    def classify_query(self, query: str, table_info: Dict[str, List[str]]) -> Tuple[QueryType, Dict[str, Any]]:
        # 테이블 정보 텍스트 변환
        tables_str = '\n'.join([f"- {t}: {', '.join(cols)}" for t, cols in table_info.items()])
        prompt = f"""
데이터베이스 테이블:
{tables_str}
질문: {query}
정형/문서/하이브리드/일반 분류 후 JSON 반환
"""
        try:
            resp = self.llm.complete(prompt)
            res = json.loads(resp.text)
            qtype = QueryType(res.get('query_type', 'unknown'))
            return qtype, res.get('parameters', {})
        except:
            # 키워드 폴백
            low = query.lower()
            data_kw = ['실적','매출','분기','공정']
            doc_kw = ['회의록','보고서','문서','언급']
            dscore = sum(kw in low for kw in data_kw)
            kscore = sum(kw in low for kw in doc_kw)
            if dscore and kscore:
                return QueryType.HYBRID, {}
            if kscore:
                return QueryType.DOCUMENT_SEARCH, {}
            if dscore:
                return QueryType.STRUCTURED_DATA, {}
            return QueryType.GENERAL_QUERY, {}

class SmartVisualizationDecider:
    """스마트 시각화 결정기"""
    def decide_visualization(self, data: pd.DataFrame, query: str, params: Dict[str, Any]) -> Tuple[VisualizationType, Dict[str, str]]:
        if data is None or data.empty:
            return VisualizationType.TABLE, {}
        nums = data.select_dtypes(include='number').columns.tolist()
        cats = data.select_dtypes(include=['object','category']).columns.tolist()
        time_cols = [c for c in data.columns if any(k in c.lower() for k in ['date','quarter','month','year'])]
        low = query.lower()
        if any(w in low for w in ['추이','변화','트렌드']) and time_cols and nums:
            return VisualizationType.LINE_CHART, {'x_column': time_cols[0], 'y_column': nums[0]}
        if any(w in low for w in ['비교','대비','차이']) and cats and nums:
            return VisualizationType.BAR_CHART, {'x_column': cats[0], 'y_column': nums[0]}
        if any(w in low for w in ['비율','구성','점유율']) and cats and nums:
            return VisualizationType.PIE_CHART, {'x_column': cats[0], 'y_column': nums[0]}
        if any(w in low for w in ['상관관계','연관']) and len(nums)>=2:
            return VisualizationType.SCATTER_PLOT, {'x_column': nums[0], 'y_column': nums[1]}
        if time_cols and nums:
            return VisualizationType.LINE_CHART, {'x_column': time_cols[0], 'y_column': nums[0]}
        if cats and nums:
            return VisualizationType.BAR_CHART, {'x_column': cats[0], 'y_column': nums[0]}
        return VisualizationType.TABLE, {}

class ImprovedQuerySystem:
    """개선된 통합 질의 처리 시스템"""
    def __init__(self, openai_api_base: str, openai_api_key: str = "fake-key", model_name: str = "gpt-3.5-turbo"):
        self.llm = OpenAILike(api_base=openai_api_base, api_key=openai_api_key, model=model_name, temperature=0.1, max_tokens=2048)
        self.db_manager = DatabaseManager()
        self.vector_db_manager = VectorDBManager(openai_api_base=openai_api_base, openai_api_key=openai_api_key)
        self.chart_generator = EnhancedChartGenerator()
        self.query_classifier = IntelligentQueryClassifier(self.llm)
        self.viz_decider = SmartVisualizationDecider()
        self.setup_query_engines()
        self.setup_tools()
        self.setup_agent()
        logger.info("ImprovedQuerySystem 초기화 완료")

    def setup_query_engines(self):
        sql_db = self.db_manager.get_sql_database()
        self.sql_query_engine = NLSQLTableQueryEngine(sql_database=sql_db,
            tables=["performance","process_performance","product_sales","quality_issues","safety_accidents"],
            llm=self.llm, synthesize_response=True, verbose=True)
        self.doc_query_engine = self.vector_db_manager.get_query_engine()
        sql_tool = QueryEngineTool(query_engine=self.sql_query_engine, metadata=ToolMetadata(name="sql_database", description="정형 데이터 조회"))
        doc_tool = QueryEngineTool(query_engine=self.doc_query_engine, metadata=ToolMetadata(name="document_search", description="문서 검색"))
        self.router_query_engine = RouterQueryEngine(selector=self.llm, query_engine_tools=[sql_tool, doc_tool], verbose=True)

    def setup_tools(self):
        def analyze_structured_data(sql_query: str, analysis_type: str = "basic") -> str:
            return FunctionTool.from_defaults(fn=lambda q, a: json.dumps({"data": pd.read_sql(q, self.db_manager.engine).to_dict("records")}), name="dummy").fn(sql_query, analysis_type)
        def search_documents(query: str, max_results: int = 5) -> str:
            return self.doc_query_engine.query(query).to_json() if hasattr(self.doc_query_engine.query(query), 'to_json') else str(self.doc_query_engine.query(query))
        def create_visualization(data_json: str, chart_type: str, title: str, x_column: str = None, y_column: str = None) -> str:
            df = pd.read_json(data_json)
            viz_type = VisualizationType(chart_type)
            path = self.chart_generator.generate_chart(df, viz_type, title, x_column, y_column)
            return path or ""
        def hybrid_analysis(structured_query: str, document_query: str) -> str:
            df = pd.read_sql(structured_query, self.db_manager.engine)
            docs = self.doc_query_engine.query(document_query)
            return json.dumps({"structured_data": df.to_dict("records"), "documents": str(docs)})
        self.structured_data_tool = FunctionTool.from_defaults(fn=analyze_structured_data, name="analyze_structured_data", description="정형 데이터 분석")
        self.document_search_tool = FunctionTool.from_defaults(fn=search_documents, name="search_documents", description="문서 검색")
        self.visualization_tool = FunctionTool.from_defaults(fn=create_visualization, name="create_visualization", description="시각화 생성")
        self.hybrid_tool = FunctionTool.from_defaults(fn=hybrid_analysis, name="hybrid_analysis", description="하이브리드 분석")

    def setup_agent(self):
        tools = [self.structured_data_tool, self.document_search_tool, self.visualization_tool, self.hybrid_tool]
        prompt = """
You are a data expert. Use tools to answer user queries based on structured or document data.
Available tools:
- analyze_structured_data
- search_documents
- create_visualization
- hybrid_analysis
"""
        self.agent = ReActAgent.from_tools(tools=tools, llm=self.llm, system_prompt=prompt, verbose=False)

    def process_query(self, user_query: str) -> QueryResult:
        logger.info(f"처리 시작: {user_query}")
        try:
            table_info = self.db_manager.get_table_info()
            qtype, params = self.query_classifier.classify_query(user_query, table_info)
            logger.info(f"분류: {qtype}")
            if qtype == QueryType.STRUCTURED_DATA:
                resp = self.sql_query_engine.query(user_query)
                sql_q = getattr(resp, 'metadata', {}).get('sql_query')
                analysis = self.structured_data_tool.fn(sql_q, params.get('analysis_type', 'basic'))
                data = pd.DataFrame(json.loads(analysis).get('data', []))
                viz, vizp = self.viz_decider.decide_visualization(data, user_query, params)
                chart = None
                if viz != VisualizationType.TABLE and not data.empty:
                    chart = self.chart_generator.generate_chart(data, viz, user_query, vizp.get('x_column'), vizp.get('y_column'))
                return QueryResult(answer=analysis, query_type=qtype, visualization_type=viz, structured_data=data, sql_query=sql_q, chart_path=chart)
            elif qtype == QueryType.DOCUMENT_SEARCH:
                search = self.document_search_tool.fn(user_query)
                res = json.loads(search) if isinstance(search, str) else search
                return QueryResult(answer=res.get('answer',''), query_type=qtype, visualization_type=VisualizationType.TABLE, document_results=res.get('source_documents', []))
            elif qtype == QueryType.HYBRID:
                sq = params.get('structured_query', user_query)
                dq = params.get('document_query', user_query)
                hybrid = self.hybrid_tool.fn(sq, dq)
                return QueryResult(answer=hybrid, query_type=qtype, visualization_type=VisualizationType.TABLE)
            else:
                chat = self.agent.chat([ChatMessage(role=MessageRole.USER, content=user_query)])
                content = getattr(chat, 'content', str(chat))
                return QueryResult(answer=content, query_type=qtype, visualization_type=VisualizationType.TABLE)
        except Exception as e:
            logger.error(f"오류: {e}")
            return QueryResult(answer=f"오류 발생: {e}", query_type=QueryType.UNKNOWN, visualization_type=VisualizationType.TABLE)

def main():
    parser = argparse.ArgumentParser(description="Improved Query System CLI")
    parser.add_argument("--api_base", required=True, help="OpenAI-like API base URL")
    parser.add_argument("--api_key", default="fake-key", help="OpenAI API key")
    args = parser.parse_args()
    system = ImprovedQuerySystem(openai_api_base=args.api_base, openai_api_key=args.api_key)
    sample_queries = [
        "공정별 실적을 찾아줘",
        "회의록 중 김영철씨가 언급된 것을 찾아줘",
        "4월 월간보고서와 결산 실적을 같이 보여줘",
        "공정별 효율 변화를 차트로 보여줘",
        "최근 안전사고 현황을 분석해줘",
        "제품별 시장 점유율을 비교해줘"
    ]
    for q in sample_queries:
        print(f"\n=== 질의: {q} ===")
        result = system.process_query(q)
        print(f"답변: {result.answer}\n")
        if result.structured_data is not None:
            print("-- Structured Data --")
            print(result.structured_data)
        if result.document_results:
            print("-- Documents --")
            for doc in result.document_results:
                print(f"- {doc.get('metadata',{}).get('title','')} : {doc.get('content','')[:100]}...")
        if result.chart_path:
            print(f"차트 경로: {result.chart_path}")
        print("=\"*40)

if __name__ == "__main__":
    main()
