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

질문: "{query}"

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
        self.llm = OpenAI
