import os
import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import tempfile
from pathlib import Path

# LlamaIndex imports
from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.query_engine import NLSQLTableQueryEngine, RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Date
from sqlalchemy.orm import sessionmaker

# 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")

class QueryType(Enum):
    """질의 유형 분류"""
    STRUCTURED_DATA = "structured_data"
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"

class VisualizationType(Enum):
    """시각화 유형"""
    TABLE = "table"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"

@dataclass
class QueryResult:
    """질의 결과 데이터 클래스"""
    data: pd.DataFrame
    query_type: QueryType
    visualization_type: VisualizationType
    sql_query: Optional[str] = None
    chart_path: Optional[str] = None
    summary: Optional[str] = None

class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "sample_data.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.setup_sample_data()
        
    def setup_sample_data(self):
        """샘플 데이터 생성"""
        # 샘플 실적 데이터
        sample_data = {
            'quarter': ['2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q1', '2024Q2'],
            'revenue': [1000, 1200, 1100, 1400, 1300, 1500],
            'operating_profit': [100, 150, 120, 180, 160, 200],
            'company': ['우리회사'] * 6
        }
        
        df = pd.DataFrame(sample_data)
        df.to_sql('performance', self.engine, if_exists='replace', index=False)
        
        # 제품별 매출 데이터
        product_data = {
            'product': ['제품A', '제품B', '제품C', '제품D'],
            'sales': [300, 450, 200, 350],
            'market_share': [15.5, 23.2, 10.3, 18.1]
        }
        
        pd.DataFrame(product_data).to_sql('product_sales', self.engine, if_exists='replace', index=False)
        
    def get_sql_database(self) -> SQLDatabase:
        """LlamaIndex SQLDatabase 객체 반환"""
        return SQLDatabase(self.engine)

class ChartGenerator:
    """차트 생성 클래스"""
    
    def __init__(self, output_dir: str = "charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_chart(self, data: pd.DataFrame, chart_type: VisualizationType, 
                      title: str = "데이터 차트") -> str:
        """차트 생성 및 파일 경로 반환"""
        plt.figure(figsize=(10, 6))
        
        if chart_type == VisualizationType.LINE_CHART:
            if 'quarter' in data.columns:
                plt.plot(data['quarter'], data['revenue'], marker='o', label='매출')
                if 'operating_profit' in data.columns:
                    plt.plot(data['quarter'], data['operating_profit'], marker='s', label='영업이익')
                plt.xlabel('분기')
                plt.ylabel('금액 (억원)')
                plt.legend()
                plt.xticks(rotation=45)
            
        elif chart_type == VisualizationType.BAR_CHART:
            if 'product' in data.columns:
                plt.bar(data['product'], data['sales'])
                plt.xlabel('제품')
                plt.ylabel('매출')
                plt.xticks(rotation=45)
            elif 'quarter' in data.columns:
                plt.bar(data['quarter'], data['revenue'])
                plt.xlabel('분기')
                plt.ylabel('매출')
                plt.xticks(rotation=45)
                
        elif chart_type == VisualizationType.PIE_CHART:
            if 'product' in data.columns and 'market_share' in data.columns:
                plt.pie(data['market_share'], labels=data['product'], autopct='%1.1f%%')
            elif 'product' in data.columns and 'sales' in data.columns:
                plt.pie(data['sales'], labels=data['product'], autopct='%1.1f%%')
        
        plt.title(title)
        plt.tight_layout()
        
        # 파일 저장
        filename = f"chart_{hash(title)}_{chart_type.value}.png"
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
        1. structured_data: 데이터베이스 조회가 필요한 정형 데이터 질의
        2. general_query: 일반적인 질문이나 대화
        
        응답 형식 (JSON):
        {{
            "query_type": "structured_data" or "general_query",
            "parameters": {{
                "table": "추정되는 테이블명",
                "conditions": ["조건들"],
                "metrics": ["조회할 지표들"],
                "time_period": "시간 범위"
            }}
        }}
        """
        
        response = self.llm.complete(classification_prompt)
        
        try:
            result = json.loads(response.text)
            query_type = QueryType(result.get('query_type', 'general_query'))
            parameters = result.get('parameters', {})
            return query_type, parameters
        except:
            return QueryType.UNKNOWN, {}

class VisualizationDecider:
    """시각화 방식 결정 클래스"""
    
    def decide_visualization(self, data: pd.DataFrame, query: str) -> VisualizationType:
        """데이터와 질의를 기반으로 적절한 시각화 방식 결정"""
        
        # 시계열 데이터인 경우
        if 'quarter' in data.columns or 'date' in data.columns:
            if '추이' in query or '변화' in query or '트렌드' in query:
                return VisualizationType.LINE_CHART
            else:
                return VisualizationType.BAR_CHART
                
        # 비율이나 구성 관련 질의
        if '비율' in query or '구성' in query or '점유율' in query:
            return VisualizationType.PIE_CHART
            
        # 비교 관련 질의
        if '비교' in query or data.shape[0] <= 10:
            return VisualizationType.BAR_CHART
            
        # 기본값: 표
        return VisualizationType.TABLE

class LlamaIndexQuerySystem:
    """LlamaIndex 기반 질의 처리 시스템"""
    
    def __init__(self, openai_api_key: str):
        # OpenAI LLM 설정
        os.environ['OPENAI_API_KEY'] = openai_api_key
        self.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
        
        # 구성 요소 초기화
        self.db_manager = DatabaseManager()
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
        
        # SQL 쿼리 엔진
        self.sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=["performance", "product_sales"],
            llm=self.llm,
            synthesize_response=True
        )
        
        # 라우터 쿼리 엔진용 도구들
        sql_tool = QueryEngineTool(
            query_engine=self.sql_query_engine,
            metadata=ToolMetadata(
                name="sql_query",
                description="정형 데이터베이스에서 데이터를 조회할 때 사용합니다. 실적, 매출, 분기별 데이터 등의 질의에 적합합니다."
            )
        )
        
        # 라우터 쿼리 엔진
        self.router_query_engine = RouterQueryEngine(
            selector=self.llm,
            query_engine_tools=[sql_tool]
        )
    
    def setup_tools(self):
        """에이전트용 도구 설정"""
        
        def generate_chart_tool(data_json: str, chart_type: str, title: str) -> str:
            """차트 생성 도구"""
            try:
                data = pd.read_json(data_json)
                viz_type = VisualizationType(chart_type)
                chart_path = self.chart_generator.generate_chart(data, viz_type, title)
                return f"차트가 생성되었습니다: {chart_path}"
            except Exception as e:
                return f"차트 생성 중 오류 발생: {str(e)}"
        
        def execute_sql_tool(sql_query: str) -> str:
            """SQL 직접 실행 도구"""
            try:
                df = pd.read_sql(sql_query, self.db_manager.engine)
                return df.to_json(orient='records', force_ascii=False)
            except Exception as e:
                return f"SQL 실행 오류: {str(e)}"
        
        self.chart_tool = FunctionTool.from_defaults(
            fn=generate_chart_tool,
            name="generate_chart",
            description="데이터를 받아 차트를 생성합니다. data_json(JSON 문자열), chart_type(line_chart/bar_chart/pie_chart), title이 필요합니다."
        )
        
        self.sql_tool = FunctionTool.from_defaults(
            fn=execute_sql_tool,
            name="execute_sql",
            description="SQL 쿼리를 직접 실행하여 데이터를 조회합니다."
        )
    
    def setup_agent(self):
        """ReAct 에이전트 설정"""
        tools = [self.chart_tool, self.sql_tool]
        
        system_prompt = """
        당신은 데이터 분석 전문가입니다. 사용자의 질문을 분석하여 적절한 도구를 사용해 답변하세요.
        
        사용 가능한 도구:
        1. execute_sql: SQL 쿼리로 데이터베이스에서 데이터를 조회
        2. generate_chart: 조회된 데이터를 차트로 시각화
        
        작업 순서:
        1. 사용자 질문 분석
        2. 필요한 경우 SQL로 데이터 조회
        3. 데이터를 표나 차트로 시각화
        4. 결과 해석 및 요약 제공
        
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
        
        # 1. 의도 분류 및 파라미터 추출
        query_type, parameters = self.query_classifier.classify_query(user_query)
        
        if query_type != QueryType.STRUCTURED_DATA:
            # 일반 질의는 간단히 처리
            response = self.llm.complete(user_query)
            return QueryResult(
                data=pd.DataFrame(),
                query_type=query_type,
                visualization_type=VisualizationType.TABLE,
                summary=response.text
            )
        
        try:
            # 2. SQL 쿼리 생성 및 실행
            sql_response = self.sql_query_engine.query(user_query)
            
            # 데이터 추출 (실제 구현에서는 response에서 데이터를 추출하는 로직 필요)
            # 여기서는 샘플 데이터로 대체
            if "분기별" in user_query or "실적" in user_query:
                data = pd.read_sql("SELECT * FROM performance", self.db_manager.engine)
            elif "제품" in user_query:
                data = pd.read_sql("SELECT * FROM product_sales", self.db_manager.engine)
            else:
                data = pd.DataFrame()
            
            # 3. 시각화 방식 결정
            viz_type = self.viz_decider.decide_visualization(data, user_query)
            
            # 4. 차트 생성 (필요한 경우)
            chart_path = None
            if viz_type != VisualizationType.TABLE and not data.empty:
                chart_path = self.chart_generator.generate_chart(
                    data, viz_type, f"{user_query} 결과"
                )
            
            # 5. 최종 응답 구성
            summary = self._generate_summary(data, user_query, str(sql_response))
            
            return QueryResult(
                data=data,
                query_type=query_type,
                visualization_type=viz_type,
                sql_query=getattr(sql_response, 'metadata', {}).get('sql_query'),
                chart_path=chart_path,
                summary=summary
            )
            
        except Exception as e:
            return QueryResult(
                data=pd.DataFrame(),
                query_type=query_type,
                visualization_type=VisualizationType.TABLE,
                summary=f"처리 중 오류가 발생했습니다: {str(e)}"
            )
    
    def _generate_summary(self, data: pd.DataFrame, query: str, sql_response: str) -> str:
        """결과 요약 생성"""
        summary_prompt = f"""
        사용자 질문: "{query}"
        조회된 데이터 요약: {data.describe().to_string() if not data.empty else "데이터 없음"}
        SQL 응답: {sql_response}
        
        위 정보를 바탕으로 사용자에게 친근하고 이해하기 쉬운 요약을 작성하세요.
        데이터의 주요 인사이트나 패턴이 있다면 언급하세요.
        """
        
        summary_response = self.llm.complete(summary_prompt)
        return summary_response.text
    
    def format_response(self, result: QueryResult) -> Dict[str, Any]:
        """응답 포맷팅"""
        response = {
            "summary": result.summary,
            "query_type": result.query_type.value,
            "visualization_type": result.visualization_type.value
        }
        
        # 표 데이터 추가
        if not result.data.empty:
            if result.visualization_type == VisualizationType.TABLE:
                response["table"] = result.data.to_dict('records')
            else:
                response["data"] = result.data.to_dict('records')
        
        # 차트 경로 추가
        if result.chart_path:
            response["chart_path"] = result.chart_path
            
        # SQL 쿼리 추가 (디버깅용)
        if result.sql_query:
            response["sql_query"] = result.sql_query
            
        return response

# 사용 예시
def main():
    """시스템 사용 예시"""
    # 시스템 초기화 (실제 사용시 OpenAI API 키 필요)
    # system = LlamaIndexQuerySystem("your-openai-api-key")
    
    # 예시 질의들
    sample_queries = [
        "우리 회사 분기별 실적을 알려줘",
        "최근 4개 분기 매출 추이를 차트로 보여줘",
        "제품별 매출 비교 분석해줘",
        "영업이익 성장률은 어떻게 되나요?"
    ]
    
    print("LlamaIndex 기반 정형 데이터 질의 처리 시스템")
    print("=" * 50)
    
    for query in sample_queries:
        print(f"\n질의: {query}")
        # result = system.process_query(query)
        # formatted_result = system.format_response(result)
        # print(f"결과: {formatted_result}")
        print("-> 실제 실행을 위해서는 OpenAI API 키가 필요합니다.")
    
    print("\n시스템 구성 요소:")
    print("1. QueryClassifier: 질의 의도 분류")
    print("2. DatabaseManager: 데이터베이스 관리")
    print("3. ChartGenerator: 차트 생성")
    print("4. VisualizationDecider: 시각화 방식 결정")
    print("5. LlamaIndexQuerySystem: 전체 시스템 조정")

if __name__ == "__main__":
    main()
