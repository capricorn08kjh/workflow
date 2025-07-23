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
from llama_index.llms.openai_like import OpenAILike # OpenAILike 임포트
from sqlalchemy import create_engine

# 로컬 모듈 imports
from data_generator import SampleDataGenerator
from vector_db_manager import VectorDBManager, DocumentQueryEngine

# 설정
# 한글 폰트 설정 (시스템에 설치된 폰트 사용)
# 예를 들어, 'Malgun Gothic' (Windows), 'AppleGothic' (macOS), 'NanumGothic' (Linux) 등
# 사용자의 환경에 따라 적절한 폰트 이름으로 변경해야 합니다.
try:
    plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 사용자용
    plt.rcParams['axes.unicode_minus'] = False # 유니코드 마이너스 기호 문제 해결
except Exception:
    logging.warning("Malgun Gothic 폰트를 찾을 수 없습니다. 다른 폰트를 시도합니다.")
    try:
        plt.rcParams['font.family'] = 'AppleGothic' # macOS 사용자용
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        logging.warning("AppleGothic 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        plt.rcParams['font.family'] = 'DejaVu Sans' # 기본 폰트 (한글 깨질 수 있음)


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
    visualization_type: VisualizationType = VisualizationType.TABLE # 기본값 설정
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
            logger.info(f"샘플 데이터베이스 '{self.db_path}' 생성 완료.")
        else:
            logger.info(f"데이터베이스 '{self.db_path}'가 이미 존재합니다.")
        
    def get_sql_database(self) -> SQLDatabase:
        """LlamaIndex SQLDatabase 객체 반환"""
        return SQLDatabase(self.engine)
    
    def get_table_info(self) -> Dict[str, List[str]]:
        """데이터베이스의 모든 테이블 이름과 컬럼 정보 반환"""
        try:
            tables = {}
            with self.engine.connect() as conn:
                inspector = sqlite3.connect(self.db_path).execute("SELECT name FROM sqlite_master WHERE type='table';")
                table_names = [row[0] for row in inspector.fetchall()]
                
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
        elif 'quarter' in data.columns and not data.empty: # 'quarter' 컬럼이 있고 데이터가 비어있지 않은 경우
            numeric_cols = data.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                for col in numeric_cols:
                    plt.plot(data['quarter'], data[col], marker='o', label=col, linewidth=2, markersize=6)
                plt.xlabel('분기', fontsize=12)
                plt.ylabel('값', fontsize=12)
                plt.legend()
                plt.xticks(rotation=45)
            else:
                logger.warning("라인 차트 생성에 적합한 숫자형 컬럼이 없습니다.")
        else:
            logger.warning("라인 차트 생성에 필요한 x_column, y_column 또는 'quarter' 컬럼이 데이터에 없습니다.")
    
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
        elif not data.empty: # 데이터가 비어있지 않은 경우
            # 기본 바차트 로직: 첫 번째 범주형 컬럼과 첫 번째 숫자형 컬럼 사용
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            numeric_cols = data.select_dtypes(include=['number']).columns
            if not categorical_cols.empty and not numeric_cols.empty:
                x_data = data[categorical_cols[0]]
                y_data = data[numeric_cols[0]]
                bars = plt.bar(x_data, y_data, alpha=0.8, color='steelblue')
                plt.xlabel(categorical_cols[0], fontsize=12)
                plt.ylabel(numeric_cols[0], fontsize=12)
                plt.xticks(rotation=45)
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.0f}', ha='center', va='bottom')
            else:
                logger.warning("바 차트 생성에 적합한 범주형 및 숫자형 컬럼이 없습니다.")
        else:
            logger.warning("바 차트 생성에 필요한 x_column, y_column 또는 적합한 컬럼이 데이터에 없습니다.")
    
    def _create_pie_chart(self, data: pd.DataFrame, x_col: str, y_col: str):
        """파이 차트 생성"""
        if x_col and y_col and x_col in data.columns and y_col in data.columns:
            # 0이 아닌 값만 필터링하여 파이 차트 생성 (오류 방지)
            filtered_data = data[data[y_col] > 0]
            if not filtered_data.empty:
                plt.pie(filtered_data[y_col], labels=filtered_data[x_col], autopct='%1.1f%%', startangle=90)
            else:
                logger.warning("파이 차트 생성을 위한 유효한 데이터가 없습니다 (y_column 값이 모두 0 또는 음수).")
        elif 'product_name' in data.columns and 'sales' in data.columns and not data.empty:
            filtered_data = data[data['sales'] > 0]
            if not filtered_data.empty:
                plt.pie(filtered_data['sales'], labels=filtered_data['product_name'], autopct='%1.1f%%', startangle=90)
            else:
                logger.warning("파이 차트 생성을 위한 유효한 데이터가 없습니다 (sales 값이 모두 0 또는 음수).")
        elif 'market_segment' in data.columns and 'market_size_usd' in data.columns and not data.empty:
            filtered_data = data[data['market_size_usd'] > 0]
            if not filtered_data.empty:
                plt.pie(filtered_data['market_size_usd'], labels=filtered_data['market_segment'], autopct='%1.1f%%', startangle=90)
            else:
                logger.warning("파이 차트 생성을 위한 유효한 데이터가 없습니다 (market_size_usd 값이 모두 0 또는 음수).")
        else:
            logger.warning("파이 차트 생성에 필요한 x_column, y_column 또는 적합한 컬럼이 데이터에 없습니다.")
    
    def _create_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str):
        """산점도 생성"""
        if x_col and y_col and x_col in data.columns and y_col in data.columns:
            plt.scatter(data[x_col], data[y_col], alpha=0.7, s=60)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
        else:
            logger.warning("산점도 생성에 필요한 x_column, y_column이 데이터에 없습니다.")

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
데이터베이스 테이블 정보와 문서 검색 관련 키워드를 참고하세요.

사용 가능한 데이터베이스 테이블:
{table_info_str}

질문: "{query}"

질의 유형 분류 기준:
1. structured_data: SQL 데이터베이스에서 정형 데이터를 조회해야 하는 질의 (예: '분기별 실적', '제품별 매출', '공정별 효율', '직원 성과', '시장 규모')
2. document_search: 문서/보고서/회의록 등에서 정보를 찾아야 하는 질의 (예: '회의록', '보고서', '문서', '언급', '요약', '기획서', '동향')
3. hybrid: 정형 데이터와 문서 검색이 모두 필요한 질의 (예: '4월 월간보고서와 결산 실적을 같이 보여줘', '시장 동향 보고서와 제품별 매출을 비교해줘')
4. general_query: 위 유형에 해당하지 않는 일반적인 질문이나 대화

파라미터 추출:
- tables: 관련된 데이터베이스 테이블명 (list of str)
- columns: 관련된 데이터베이스 컬럼명 (list of str)
- person_names: 언급된 인물명 (list of str)
- date_keywords: 날짜 관련 키워드 (list of str)
- document_types: 찾아야 할 문서 유형 (list of str, 예: '회의록', '보고서', '기획서')
- analysis_type: 요구되는 분석 유형 (str, 예: '비교', '추이', '요약', '상관관계')
- keywords: 질문에서 중요한 키워드 (list of str)

응답 형식 (JSON):
{{
    "query_type": "분류 결과 (structured_data, document_search, hybrid, general_query 중 하나)",
    "confidence": 0.9,
    "parameters": {{
        "tables": [],
        "columns": [],
        "person_names": [],
        "date_keywords": [],
        "document_types": [],
        "analysis_type": null,
        "keywords": []
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
            logger.debug(f"분류 파라미터: {parameters}")
            return query_type, parameters
            
        except Exception as e:
            logger.error(f"질의 분류 실패: {str(e)}. 폴백 분류를 시도합니다.")
            # 키워드 기반 폴백 분류
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Tuple[QueryType, Dict[str, Any]]:
        """LLM 분류 실패 시 키워드 기반 폴백 분류 로직"""
        query_lower = query.lower()
        
        # 문서 검색 키워드
        doc_keywords = ['회의록', '보고서', '문서', '언급', '찾아', '요약', '기획서', '동향']
        # 정형 데이터 키워드  
        data_keywords = ['실적', '매출', '분기', '제품', '공정', '효율', '생산', '직원', '성과', '시장', '규모']
        # 하이브리드 키워드 조합 (간단한 예시)
        hybrid_keywords = [('보고서', '실적'), ('문서', '데이터'), ('동향', '매출')]
        
        doc_score = sum(1 for kw in doc_keywords if kw in query_lower)
        data_score = sum(1 for kw in data_keywords if kw in query_lower)
        
        is_hybrid = False
        for kw1, kw2 in hybrid_keywords:
            if kw1 in query_lower and kw2 in query_lower:
                is_hybrid = True
                break

        if is_hybrid:
            return QueryType.HYBRID, {}
        elif doc_score > 0 and data_score == 0: # 문서 키워드만 있을 때
            return QueryType.DOCUMENT_SEARCH, {}
        elif data_score > 0 and doc_score == 0: # 데이터 키워드만 있을 때
            return QueryType.STRUCTURED_DATA, {}
        elif data_score > 0 and doc_score > 0: # 둘 다 있을 때 (하이브리드 또는 복합 정형/문서)
             return QueryType.HYBRID, {} # 좀 더 정교한 분류 필요하지만 일단 하이브리드로
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
        
        # 1. 질의 키워드 기반 결정 (우선 순위 높음)
        if any(keyword in query_lower for keyword in ['추이', '변화', '트렌드', '증감']):
            if time_cols and numeric_cols:
                # 시간 컬럼이 있고 숫자 컬럼이 있다면 라인 차트
                viz_params = {'x_column': time_cols[0], 'y_column': numeric_cols[0]}
                return VisualizationType.LINE_CHART, viz_params
        
        elif any(keyword in query_lower for keyword in ['비율', '구성', '점유율', '비중', '파이']):
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                # 범주형 컬럼과 숫자형 컬럼이 있다면 파이 차트
                viz_params = {'x_column': categorical_cols[0], 'y_column': numeric_cols[0]}
                return VisualizationType.PIE_CHART, viz_params
        
        elif any(keyword in query_lower for keyword in ['비교', '대비', '차이', '막대']):
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                # 범주형 컬럼과 숫자형 컬럼이 있다면 바 차트
                viz_params = {'x_column': categorical_cols[0], 'y_column': numeric_cols[0]}
                return VisualizationType.BAR_CHART, viz_params
            elif len(numeric_cols) >= 2: # 숫자형 컬럼이 2개 이상이면 산점도도 고려
                 viz_params = {'x_column': numeric_cols[0], 'y_column': numeric_cols[1]}
                 return VisualizationType.SCATTER_PLOT, viz_params
        
        elif any(keyword in query_lower for keyword in ['상관관계', '관계', '연관', '산점도']):
            if len(numeric_cols) >= 2:
                viz_params = {'x_column': numeric_cols[0], 'y_column': numeric_cols[1]}
                return VisualizationType.SCATTER_PLOT, viz_params
        
        # 2. 데이터 특성 기반 결정 (질의 키워드가 명확하지 않을 때)
        # 데이터 행이 적으면 테이블이 적합할 수 있음
        if len(data) <= 10:
            return VisualizationType.TABLE, {}

        # 시간 컬럼이 있다면 라인 차트 (기본)
        if time_cols and numeric_cols:
            viz_params = {'x_column': time_cols[0], 'y_column': numeric_cols[0]}
            return VisualizationType.LINE_CHART, viz_params
        
        # 범주형 컬럼이 있고 숫자형 컬럼이 있다면 바 차트 (기본)
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            viz_params = {'x_column': categorical_cols[0], 'y_column': numeric_cols[0]}
            return VisualizationType.BAR_CHART, viz_params
            
        # 숫자형 컬럼이 2개 이상이면 산점도
        if len(numeric_cols) >= 2:
            viz_params = {'x_column': numeric_cols[0], 'y_column': numeric_cols[1]}
            return VisualizationType.SCATTER_PLOT, viz_params

        # 기본값: 표
        return VisualizationType.TABLE, {}

class ImprovedQuerySystem:
    """개선된 통합 질의 처리 시스템"""
    
    def __init__(self, 
                 llm_api_base: str = "http://localhost:11434/v1", # OpenAILike LLM의 기본 API 베이스 URL
                 llm_model_name: str = "llama3", # OpenAILike LLM의 기본 모델명
                 embed_api_base: str = "http://localhost:11434/v1", # OpenAILike Embedding의 기본 API 베이스 URL
                 embed_model_name: str = "nomic-embed-text", # OpenAILike Embedding의 기본 모델명
                 db_path: str = "company_data.db",
                 chroma_persist_dir: str = "./chroma_db"):
        """
        초기화
        
        Args:
            llm_api_base: LLM의 OpenAI-like API 베이스 URL
            llm_model_name: LLM 모델명
            embed_api_base: 임베딩 모델의 OpenAI-like API 베이스 URL
            embed_model_name: 임베딩 모델명
            db_path: SQLite 데이터베이스 파일 경로
            chroma_persist_dir: ChromaDB 저장 경로
        """
        
        # OpenAI-like LLM 설정
        self.llm = OpenAILike(
            api_base=llm_api_base,
            model=llm_model_name,
            temperature=0.1,
            max_tokens=2048,
            api_key="fake-key" # OpenAILike는 실제 키가 필요 없을 수 있으나, 명시적으로 설정
        )
        
        # 구성 요소 초기화
        self.db_manager = DatabaseManager(db_path=db_path)
        self.vector_db_manager = VectorDBManager(
            persist_dir=chroma_persist_dir,
            openai_api_base=embed_api_base,
            embed_model_name=embed_model_name
        )
        self.chart_generator = EnhancedChartGenerator()
        self.query_classifier = IntelligentQueryClassifier(self.llm)
        self.viz_decider = SmartVisualizationDecider()
        
        # 쿼리 엔진 및 도구 설정
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
        
        # 라우터 쿼리 엔진 (현재는 사용하지 않지만, 향후 확장성을 위해 유지)
        self.router_query_engine = RouterQueryEngine(
            selector=self.llm,
            query_engine_tools=[sql_tool, doc_tool],
            verbose=True
        )
    
    def setup_tools(self):
        """에이전트용 고급 도구 설정"""
        
        def analyze_structured_data(query: str) -> str:
            """
            정형 데이터 분석 도구
            사용자의 자연어 쿼리를 받아 SQL 쿼리로 변환하고 실행하여 데이터를 분석합니다.
            """
            logger.info(f"analyze_structured_data 도구 호출: {query}")
            try:
                # NLSQLTableQueryEngine을 사용하여 자연어 쿼리를 SQL로 변환하고 실행
                response = self.sql_query_engine.query(query)
                
                # SQL 쿼리 추출 (디버깅용)
                sql_query = None
                if hasattr(response, 'metadata') and 'sql_query' in response.metadata:
                    sql_query = response.metadata['sql_query']
                
                # 데이터프레임 추출 (NLSQLTableQueryEngine 응답에서 직접 데이터프레임을 얻기 어려울 수 있으므로,
                # 필요하다면 SQL 쿼리를 다시 실행하여 데이터프레임을 얻는 로직 추가)
                # 여기서는 간단히 응답 텍스트와 SQL 쿼리만 반환
                
                # 더 나은 데이터 추출을 위해 직접 SQL 실행 로직을 추가
                if sql_query:
                    try:
                        data = pd.read_sql(sql_query, self.db_manager.engine)
                        analysis_result = {
                            "answer": str(response),
                            "data_summary": {
                                "row_count": len(data),
                                "column_count": len(data.columns),
                                "columns": data.columns.tolist()
                            },
                            "data": data.to_dict('records'),
                            "sql_query": sql_query
                        }
                        return json.dumps(analysis_result, ensure_ascii=False, default=str)
                    except Exception as e_sql:
                        logger.error(f"SQL 쿼리 직접 실행 실패: {e_sql}")
                        return f"데이터 분석 응답: {str(response)}. SQL 실행 중 오류 발생: {str(e_sql)}"
                else:
                    return f"데이터 분석 응답: {str(response)}. SQL 쿼리를 추출할 수 없습니다."
                
            except Exception as e:
                logger.error(f"정형 데이터 분석 중 오류 발생: {str(e)}")
                return f"정형 데이터 분석 중 오류 발생: {str(e)}"
        
        def search_documents(query: str, max_results: int = 5) -> str:
            """문서 검색 도구"""
            logger.info(f"search_documents 도구 호출: {query}")
            try:
                results = self.doc_query_engine.query(query)
                
                # 검색 결과 정리
                search_results = {
                    "query": query,
                    "answer": str(results), # LLM이 생성한 응답 텍스트
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
                
                return json.dumps(search_results, ensure_ascii=False, default=str)
                
            except Exception as e:
                logger.error(f"문서 검색 중 오류 발생: {str(e)}")
                return f"문서 검색 중 오류 발생: {str(e)}"
        
        def create_visualization(data_json: str, chart_type: str, title: str, 
                               x_column: str = None, y_column: str = None) -> str:
            """향상된 시각화 생성 도구"""
            logger.info(f"create_visualization 도구 호출: {chart_type}, {title}")
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
                logger.error(f"시각화 생성 중 오류 발생: {str(e)}")
                return f"시각화 생성 중 오류 발생: {str(e)}"
        
        def hybrid_analysis(structured_query: str, document_query: str) -> str:
            """하이브리드 분석 도구 - 정형 데이터와 문서를 함께 분석"""
            logger.info(f"hybrid_analysis 도구 호출: 정형='{structured_query}', 문서='{document_query}'")
            try:
                # 정형 데이터 조회
                structured_data_response = self.sql_query_engine.query(structured_query)
                structured_data_text = str(structured_data_response)
                
                # SQL 쿼리 추출 및 데이터프레임 로드 시도
                sql_query_for_data = None
                if hasattr(structured_data_response, 'metadata') and 'sql_query' in structured_data_response.metadata:
                    sql_query_for_data = structured_data_response.metadata['sql_query']
                
                structured_data_df = pd.DataFrame()
                if sql_query_for_data:
                    try:
                        structured_data_df = pd.read_sql(sql_query_for_data, self.db_manager.engine)
                    except Exception as e_sql:
                        logger.warning(f"하이브리드 분석 중 SQL 데이터프레임 로드 실패: {e_sql}")

                # 문서 검색
                doc_results = self.doc_query_engine.query(document_query)
                doc_results_text = str(doc_results)
                
                # 결과 통합
                hybrid_result = {
                    "structured_data_summary": structured_data_text,
                    "structured_data_table": structured_data_df.to_dict('records') if not structured_data_df.empty else [],
                    "document_results_summary": doc_results_text,
                    "document_source_count": len(doc_results.source_nodes) if hasattr(doc_results, 'source_nodes') else 0
                }
                
                return json.dumps(hybrid_result, ensure_ascii=False, default=str)
                
            except Exception as e:
                logger.error(f"하이브리드 분석 중 오류 발생: {str(e)}")
                return f"하이브리드 분석 중 오류 발생: {str(e)}"
        
        # 도구 등록
        self.structured_data_tool = FunctionTool.from_defaults(
            fn=analyze_structured_data,
            name="analyze_structured_data",
            description="SQL 데이터베이스에서 정형 데이터를 조회하고 분석합니다. 사용자 자연어 쿼리를 입력으로 받습니다."
        )
        
        self.document_search_tool = FunctionTool.from_defaults(
            fn=search_documents,
            name="search_documents", 
            description="문서 데이터베이스에서 관련 정보를 검색합니다. 사용자 자연어 쿼리를 입력으로 받습니다."
        )
        
        self.visualization_tool = FunctionTool.from_defaults(
            fn=create_visualization,
            name="create_visualization",
            description=(
                "데이터(JSON 문자열), 차트 유형(line_chart, bar_chart, pie_chart, scatter_plot), 제목, "
                "선택적으로 x축 컬럼명, y축 컬럼명을 받아 차트나 그래프로 시각화합니다."
            )
        )
        
        self.hybrid_tool = FunctionTool.from_defaults(
            fn=hybrid_analysis,
            name="hybrid_analysis",
            description="정형 데이터 쿼리와 문서 검색 쿼리를 각각 받아 함께 수행하여 종합적인 분석을 제공합니다."
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
필요하다면 여러 도구를 조합하여 사용하세요.

사용 가능한 도구:
1. analyze_structured_data(query: str): SQL 데이터베이스에서 정형 데이터 조회 및 분석. 예: "분기별 매출을 알려줘"
2. search_documents(query: str): 회의록, 보고서 등 문서에서 정보 검색. 예: "김영철씨가 언급된 회의록을 찾아줘"
3. create_visualization(data_json: str, chart_type: str, title: str, x_column: Optional[str]=None, y_column: Optional[str]=None): 데이터를 차트로 시각화. 데이터는 JSON 문자열 형태여야 합니다.
4. hybrid_analysis(structured_query: str, document_query: str): 정형 데이터와 문서를 함께 분석. 예: "4월 월간보고서와 결산 실적을 같이 보여줘"

질의 처리 단계:
1. 사용자 질문의 의도와 유형 파악
2. 필요한 데이터 조회 (정형/비정형)
3. 데이터 분석 및 인사이트 도출
4. 적절한 시각화 생성 (필요한 경우)
5. 종합적인 답변 제공

답변 시 주의사항:
- 한국어로 친근하고 전문적으로 답변
- 데이터의 한계나 불확실성 명시 (예: "현재 데이터에는 ~ 정보가 부족합니다.")
- 구체적인 수치와 근거 제시
- 시각화가 도움이 되는 경우 차트 생성 (create_visualization 도구 사용)
- 시각화가 생성되면, 해당 차트 파일 경로를 사용자에게 알려주세요.
- 최종 답변은 사용자가 이해하기 쉽도록 명확하고 간결하게 요약하세요.

데이터베이스 테이블 정보:
- quarterly_performance: 분기별 실적 데이터 (quarter, revenue, operating_profit, net_profit, company)
- process_performance: 공정별 성과 데이터 (process_id, process_name, efficiency, defect_rate, throughput, date)
- product_sales: 제품별 매출 데이터 (product_id, product_name, sales, units_sold, category, market_share)
- employee_performance: 직원 성과 데이터 (employee_id, name, department, performance_score, sales_achieved, projects_completed)
- market_analysis: 시장 분석 데이터 (report_date, market_segment, market_size_usd, growth_rate_percent, key_competitors)
"""
        
        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
            max_iterations=15 # 반복 횟수 증가 (복잡한 질의 처리 위함)
        )
        
        logger.info("ReAct 에이전트 설정 완료")
    
    def process_query(self, user_query: str) -> QueryResult:
        """사용자 질의 전체 처리 파이프라인"""
        
        logger.info(f"질의 처리 시작: {user_query}")
        
        try:
            # 1. 질의 분류 및 파라미터 추출
            table_info = self.db_manager.get_table_info()
            query_type, parameters = self.query_classifier.classify_query(user_query, table_info)
            
            logger.info(f"질의 분류 결과: {query_type.value}, 파라미터: {parameters}")
            
            answer_text = ""
            structured_data_df = pd.DataFrame()
            document_results_list = []
            sql_query_used = None
            chart_file_path = None
            final_visualization_type = VisualizationType.TABLE

            # 2. 질의 유형별 처리 (에이전트에게 위임)
            # 에이전트가 도구를 사용하여 적절한 답변을 생성하도록 함
            agent_response = self.agent.query(user_query)
            answer_text = str(agent_response)
            
            # 에이전트 응답에서 데이터 및 차트 경로 추출 시도
            # 에이전트가 직접 JSON 형태의 결과를 반환하지 않을 수 있으므로,
            # 응답 텍스트를 파싱하거나, 도구 호출의 로그를 분석해야 할 수 있습니다.
            # 여기서는 간단히 에이전트의 최종 응답 텍스트를 사용하고,
            # create_visualization 도구 호출 시 반환되는 경로를 추적합니다.

            # TODO: 에이전트의 내부 실행 과정에서 생성된 데이터와 차트 경로를
            # QueryResult 객체에 더 정확하게 매핑하는 로직 추가 필요.
            # 현재는 agent_response에서 직접 추출하기 어려우므로,
            # create_visualization 도구의 반환 값을 활용하거나,
            # 에이전트의 내부 상태를 모니터링하는 방식이 필요합니다.
            
            # 임시로, 응답 텍스트에 차트 경로가 포함되어 있는지 확인
            if "시각화가 성공적으로 생성되었습니다:" in answer_text:
                chart_file_path_start = answer_text.find("시각화가 성공적으로 생성되었습니다:") + len("시각화가 성공적으로 생성되었습니다:")
                chart_file_path_end = answer_text.find(" ", chart_file_path_start) # 공백까지
                if chart_file_path_end == -1: # 공백이 없으면 끝까지
                    chart_file_path_end = len(answer_text)
                chart_file_path = answer_text[chart_file_path_start:chart_file_path_end].strip()
                final_visualization_type = VisualizationType.BAR_CHART # 차트가 생성되었으므로 임의로 설정

            # structured_data_df와 document_results_list는 에이전트 내부에서 처리되므로,
            # 최종 QueryResult에 포함시키기 위해서는 에이전트의 응답을 파싱하거나,
            # 도구의 반환 값을 직접 받아와야 합니다. 현재는 answer_text에 요약된 내용만 포함됩니다.
            # 더 정교한 구현을 위해서는 analyze_structured_data 및 search_documents 도구가
            # 반환하는 JSON을 파싱하여 QueryResult의 structured_data 및 document_results 필드를 채워야 합니다.
            
            # 예시: analyze_structured_data 도구의 JSON 응답을 파싱하여 structured_data_df 채우기
            # (이 부분은 에이전트의 응답 구조에 따라 달라질 수 있습니다.)
            try:
                if query_type == QueryType.STRUCTURED_DATA or query_type == QueryType.HYBRID:
                    # 에이전트가 analyze_structured_data를 호출하고 그 결과를 반환했다고 가정
                    # 실제로는 에이전트의 trace를 분석해야 더 정확함
                    if "data_summary" in answer_text and "data" in answer_text:
                        # 간단한 JSON 파싱 시도 (에이전트 응답이 깔끔한 JSON일 경우)
                        temp_json_start = answer_text.find("{")
                        temp_json_end = answer_text.rfind("}") + 1
                        if temp_json_start != -1 and temp_json_end != -1:
                            try:
                                parsed_agent_output = json.loads(answer_text[temp_json_start:temp_json_end])
                                if "data" in parsed_agent_output:
                                    structured_data_df = pd.DataFrame(parsed_agent_output["data"])
                                if "sql_query" in parsed_agent_output:
                                    sql_query_used = parsed_agent_output["sql_query"]
                                final_visualization_type = self.viz_decider.decide_visualization(
                                    structured_data_df, user_query, parameters
                                )[0] # 데이터가 있다면 시각화 결정
                            except json.JSONDecodeError:
                                logger.warning("에이전트 응답이 유효한 JSON이 아닙니다. 데이터프레임 추출 실패.")
            except Exception as e_parse:
                logger.error(f"에이전트 응답 파싱 중 오류 발생: {e_parse}")

            return QueryResult(
                answer=answer_text,
                query_type=query_type,
                visualization_type=final_visualization_type,
                structured_data=structured_data_df if not structured_data_df.empty else None,
                document_results=document_results_list if document_results_list else None,
                sql_query=sql_query_used,
                chart_path=chart_file_path,
                metadata=parameters # 분류 시 추출된 파라미터를 메타데이터로 저장
            )
            
        except Exception as e:
            logger.error(f"질의 처리 중 예상치 못한 오류 발생: {str(e)}")
            return QueryResult(
                answer=f"질의 처리 중 오류가 발생했습니다: {str(e)}",
                query_type=QueryType.UNKNOWN,
                visualization_type=VisualizationType.TABLE
            )
    
    def format_response(self, result: QueryResult) -> Dict[str, Any]:
        """응답 포맷팅"""
        response = {
            "answer": result.answer,
            "query_type": result.query_type.value,
            "visualization_type": result.visualization_type.value
        }
        
        # 표 데이터 추가
        if result.structured_data is not None and not result.structured_data.empty:
            response["structured_data"] = result.structured_data.to_dict('records')
        
        # 문서 검색 결과 추가
        if result.document_results:
            response["document_results"] = result.document_results
        
        # 차트 경로 추가
        if result.chart_path:
            response["chart_path"] = result.chart_path
            
        # SQL 쿼리 추가 (디버깅용)
        if result.sql_query:
            response["sql_query"] = result.sql_query
            
        # 추가 메타데이터
        if result.metadata:
            response["metadata"] = result.metadata
            
        return response

# 사용 예시
def main():
    """시스템 사용 예시"""
    # 시스템 초기화 (실제 사용시 Ollama 서버가 실행 중이어야 합니다)
    # LLM 모델 (예: llama3)과 임베딩 모델 (예: nomic-embed-text)이 Ollama에 다운로드되어 있어야 합니다.
    # 예: ollama run llama3, ollama run nomic-embed-text
    
    ollama_api_base = "http://localhost:11434/v1" # Ollama의 기본 API 베이스 URL
    llm_model = "llama3" # 사용할 LLM 모델명
    embed_model = "nomic-embed-text" # 사용할 임베딩 모델명

    print("ImprovedQuerySystem 초기화 중...")
    try:
        system = ImprovedQuerySystem(
            llm_api_base=ollama_api_base,
            llm_model_name=llm_model,
            embed_api_base=ollama_api_base,
            embed_model_name=embed_model
        )
        print("ImprovedQuerySystem 초기화 완료.")
    except Exception as e:
        print(f"시스템 초기화 실패: {e}")
        print("Ollama 서버가 실행 중인지, 그리고 'llama3' 및 'nomic-embed-text' 모델이 다운로드되어 있는지 확인하세요.")
        return

    sample_queries = [
        "우리 회사 분기별 실적을 알려줘",
        "최근 4개 분기 매출 추이를 라인 차트로 보여줘",
        "제품별 매출을 비교 분석해줘",
        "영업이익 성장률은 어떻게 되나요?",
        "공정별 효율성을 보여줘",
        "회의록 중 김영철씨가 언급된 것을 찾아줘",
        "4월 월간보고서의 주요 내용은 뭐야?",
        "2023년 결산 실적 보고서에 대해 알려줘",
        "신제품 개발 프로젝트 '알파' 기획서 내용을 요약해줘",
        "4월 월간보고서와 2023년 결산 실적을 같이 보여줘", # 하이브리드 질의
        "시장 동향 분석 보고서와 제품별 시장 점유율을 비교해줘", # 하이브리드 질의
        "직원 김철수의 최근 성과를 알려줘",
        "우리 회사의 시장 분석 보고서에 따르면 스마트폰 시장 규모는 어떻게 되나요?"
    ]
    
    print("\nLlamaIndex 기반 통합 질의 처리 시스템")
    print("=" * 50)
    
    for i, query in enumerate(sample_queries):
        print(f"\n--- 질의 {i+1}: {query} ---")
        result = system.process_query(query)
        formatted_result = system.format_response(result)
        
        print(f"최종 답변: {formatted_result.get('answer')}")
        print(f"질의 유형: {formatted_result.get('query_type')}")
        print(f"시각화 유형: {formatted_result.get('visualization_type')}")
        
        if formatted_result.get('structured_data'):
            print("\n정형 데이터 (일부):")
            print(pd.DataFrame(formatted_result['structured_data']).head())
        
        if formatted_result.get('document_results'):
            print("\n문서 검색 결과 (일부):")
            for doc in formatted_result['document_results'][:2]: # 최대 2개만 출력
                print(f"- 제목: {doc.get('metadata', {}).get('title', 'N/A')}")
                print(f"  내용: {doc.get('content', 'N/A')[:100]}...")
        
        if formatted_result.get('chart_path'):
            print(f"\n생성된 차트: {formatted_result['chart_path']}")
        
        if formatted_result.get('sql_query'):
            print(f"\n실행된 SQL 쿼리: {formatted_result['sql_query']}")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
                                                                          
