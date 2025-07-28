"""
향상된 쿼리 의도 분석 모듈
정형/비정형 데이터, 시각화 의도까지 분석하는 고도화된 분석기
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from llama_index.llms.openai_like import OpenAILike

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """쿼리 의도 분류 (확장)"""
    # 데이터 접근 관련
    STRUCTURED_DATA_RETRIEVAL = "structured_data_retrieval"      # 정형 데이터 조회
    UNSTRUCTURED_DATA_RETRIEVAL = "unstructured_data_retrieval"  # 비정형 데이터 조회
    HYBRID_DATA_RETRIEVAL = "hybrid_data_retrieval"              # 정형+비정형 데이터 조합
    
    # 분석 관련
    DATA_ANALYSIS = "data_analysis"                              # 데이터 분석/집계
    STATISTICAL_ANALYSIS = "statistical_analysis"               # 통계 분석
    
    # 시각화 관련
    CHART_GENERATION = "chart_generation"                        # 차트 생성
    TABLE_GENERATION = "table_generation"                        # 표 생성
    DASHBOARD_REQUEST = "dashboard_request"                      # 대시보드 요청
    
    # 기타
    SCHEMA_INQUIRY = "schema_inquiry"                           # 스키마/구조 문의
    DOCUMENT_SEARCH = "document_search"                         # 문서 검색
    GENERAL_QUESTION = "general_question"                       # 일반적인 질문
    GREETING = "greeting"                                       # 인사
    HELP_REQUEST = "help_request"                              # 도움말 요청
    UNCLEAR = "unclear"                                        # 의도 불분명

class DataSourceType(Enum):
    """데이터 소스 유형"""
    ORACLE_ONLY = "oracle_only"              # Oracle DB만
    CHROMADB_ONLY = "chromadb_only"          # ChromaDB만
    ORACLE_CHROMADB = "oracle_chromadb"      # Oracle + ChromaDB
    UNKNOWN = "unknown"                      # 알 수 없음

class VisualizationType(Enum):
    """시각화 유형"""
    LINE_CHART = "line_chart"                # 라인 차트
    BAR_CHART = "bar_chart"                  # 막대 차트
    PIE_CHART = "pie_chart"                  # 파이 차트
    SCATTER_PLOT = "scatter_plot"            # 산점도
    HISTOGRAM = "histogram"                  # 히스토그램
    HEATMAP = "heatmap"                      # 히트맵
    TABLE = "table"                          # 표
    DASHBOARD = "dashboard"                  # 대시보드
    NO_VISUALIZATION = "no_visualization"    # 시각화 없음

class QueryComplexity(Enum):
    """쿼리 복잡도 (확장)"""
    SIMPLE = "simple"                        # 단순 조회
    MODERATE = "moderate"                    # 중간 복잡도
    COMPLEX = "complex"                      # 복잡한 분석
    MULTI_SOURCE = "multi_source"            # 다중 소스 연동
    ADVANCED_ANALYTICS = "advanced_analytics" # 고급 분석

@dataclass
class EnhancedIntentAnalysisResult:
    """향상된 의도 분석 결과"""
    intent: QueryIntent
    data_source_type: DataSourceType
    visualization_type: VisualizationType
    complexity: QueryComplexity
    confidence: float
    
    # 데이터 소스별 필요성
    needs_oracle: bool
    needs_chromadb: bool
    needs_visualization: bool
    needs_clarification: bool
    
    # 추출된 정보
    extracted_entities: Dict[str, List[str]]
    suggested_questions: List[str]
    visualization_config: Dict[str, Any]
    
    # 추론 과정
    reasoning: str

class EnhancedQueryIntentAnalyzer:
    """향상된 쿼리 의도 분석기"""
    
    def __init__(self, 
                 llm: Optional[OpenAILike] = None,
                 oracle_tables: Optional[Dict[str, str]] = None,
                 chromadb_collections: Optional[List[str]] = None):
        """
        초기화
        
        Args:
            llm: LLM 인스턴스
            oracle_tables: Oracle 테이블 정보
            chromadb_collections: ChromaDB 컬렉션 정보
        """
        self.llm = llm
        self.oracle_tables = oracle_tables or {}
        self.chromadb_collections = chromadb_collections or []
        
        # 확장된 의도별 키워드 패턴
        self.intent_patterns = {
            QueryIntent.STRUCTURED_DATA_RETRIEVAL: [
                r'조회', r'찾아', r'가져와', r'보여주', r'리스트', r'목록',
                r'데이터', r'정보', r'현황', r'상태', r'내역',
                r'select', r'get', r'find', r'show', r'list', r'fetch'
            ],
            QueryIntent.UNSTRUCTURED_DATA_RETRIEVAL: [
                r'문서', r'텍스트', r'내용', r'원문', r'본문', r'문서에서',
                r'검색', r'찾기', r'유사', r'관련.*문서', r'문서.*내용',
                r'document', r'text', r'content', r'search', r'similar'
            ],
            QueryIntent.HYBRID_DATA_RETRIEVAL: [
                r'문서.*함께', r'원문.*포함', r'텍스트.*데이터',
                r'상세.*내용', r'문서.*연결', r'관련.*문서.*조회',
                r'with.*document', r'including.*text', r'detailed.*content'
            ],
            QueryIntent.CHART_GENERATION: [
                r'차트', r'그래프', r'시각화', r'그려', r'도표', r'플롯',
                r'라인.*차트', r'막대.*차트', r'파이.*차트', r'산점도',
                r'chart', r'graph', r'plot', r'visualize', r'line.*chart',
                r'bar.*chart', r'pie.*chart', r'scatter'
            ],
            QueryIntent.TABLE_GENERATION: [
                r'표', r'테이블', r'표로', r'정리해', r'요약.*표',
                r'table', r'tabular', r'summary.*table'
            ],
            QueryIntent.DATA_ANALYSIS: [
                r'분석', r'통계', r'집계', r'합계', r'평균', r'최대', r'최소',
                r'개수', r'비율', r'퍼센트', r'추이', r'트렌드', r'비교',
                r'analyze', r'analysis', r'statistics', r'aggregate', r'trend'
            ]
        }
        
        # 데이터 소스 힌트 패턴
        self.data_source_patterns = {
            DataSourceType.ORACLE_ONLY: [
                r'테이블', r'컬럼', r'조인', r'집계', r'그룹', r'정형',
                r'table', r'column', r'join', r'group', r'structured'
            ],
            DataSourceType.CHROMADB_ONLY: [
                r'문서', r'텍스트', r'검색', r'유사', r'벡터', r'임베딩',
                r'document', r'text', r'search', r'similar', r'vector', r'embedding'
            ],
            DataSourceType.ORACLE_CHROMADB: [
                r'문서.*함께', r'원문.*포함', r'상세.*내용', r'관련.*문서',
                r'with.*document', r'including.*text', r'detailed.*content'
            ]
        }
        
        # 시각화 유형 패턴
        self.visualization_patterns = {
            VisualizationType.LINE_CHART: [
                r'라인.*차트', r'선.*그래프', r'추이', r'트렌드', r'시계열',
                r'line.*chart', r'line.*graph', r'trend', r'time.*series'
            ],
            VisualizationType.BAR_CHART: [
                r'막대.*차트', r'바.*차트', r'비교', r'순위', r'랭킹',
                r'bar.*chart', r'column.*chart', r'ranking', r'comparison'
            ],
            VisualizationType.PIE_CHART: [
                r'파이.*차트', r'원.*그래프', r'비율', r'구성', r'분포',
                r'pie.*chart', r'ratio', r'proportion', r'distribution'
            ],
            VisualizationType.TABLE: [
                r'표', r'테이블', r'목록', r'리스트', r'정리',
                r'table', r'list', r'summary'
            ]
        }
    
    def analyze_intent(self, query: str) -> EnhancedIntentAnalysisResult:
        """
        향상된 쿼리 의도 분석
        
        Args:
            query: 자연어 쿼리
            
        Returns:
            EnhancedIntentAnalysisResult: 분석 결과
        """
        query_lower = query.lower().strip()
        
        # 1. 기본 의도 분류
        intent, intent_confidence = self._classify_enhanced_intent(query_lower)
        
        # 2. 데이터 소스 유형 결정
        data_source_type = self._determine_data_source_type(query_lower, intent)
        
        # 3. 시각화 유형 결정
        visualization_type = self._determine_visualization_type(query_lower, intent)
        
        # 4. 복잡도 분석
        complexity = self._analyze_enhanced_complexity(query_lower, data_source_type)
        
        # 5. 엔터티 추출
        entities = self._extract_enhanced_entities(query_lower)
        
        # 6. 필요성 판단
        needs_oracle = self._needs_oracle(data_source_type, entities)
        needs_chromadb = self._needs_chromadb(data_source_type, entities)
        needs_visualization = visualization_type != VisualizationType.NO_VISUALIZATION
        
        # 7. 명확화 필요성 및 제안 질문
        needs_clarification, suggested_questions = self._needs_enhanced_clarification(
            query_lower, intent, data_source_type, visualization_type, entities, intent_confidence
        )
        
        # 8. 시각화 설정
        visualization_config = self._generate_visualization_config(
            visualization_type, entities, intent
        )
        
        # 9. 추론 과정 설명
        reasoning = self._generate_enhanced_reasoning(
            query, intent, data_source_type, visualization_type, 
            complexity, entities, intent_confidence
        )
        
        return QueryIntent.UNCLEAR, 0.3 EnhancedIntentAnalysisResult(
            intent=intent,
            data_source_type=data_source_type,
            visualization_type=visualization_type,
            complexity=complexity,
            confidence=intent_confidence,
            needs_oracle=needs_oracle,
            needs_chromadb=needs_chromadb,
            needs_visualization=needs_visualization,
            needs_clarification=needs_clarification,
            extracted_entities=entities,
            suggested_questions=suggested_questions,
            visualization_config=visualization_config,
            reasoning=reasoning
        )
    
    def _classify_enhanced_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """향상된 의도 분류"""
        intent_scores = {}
        
        # 패턴 기반 점수 계산
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        # 복합 의도 감지
        if intent_scores.get(QueryIntent.STRUCTURED_DATA_RETRIEVAL, 0) > 0 and \
           intent_scores.get(QueryIntent.UNSTRUCTURED_DATA_RETRIEVAL, 0) > 0:
            intent_scores[QueryIntent.HYBRID_DATA_RETRIEVAL] = \
                intent_scores.get(QueryIntent.STRUCTURED_DATA_RETRIEVAL, 0) + \
                intent_scores.get(QueryIntent.UNSTRUCTURED_DATA_RETRIEVAL, 0)
        
        if not intent_scores:
            if self.llm:
                return self._llm_classify_enhanced_intent(query)
            else:
                return QueryIntent.UNCLEAR, 0.3
        
        # 가장 높은 점수의 의도 선택
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        total_score = sum(intent_scores.values())
        
        confidence = min(max_score / max(total_score, 1), 1.0)
        
        return best_intent, confidence
    
    def _determine_data_source_type(self, query: str, intent: QueryIntent) -> DataSourceType:
        """데이터 소스 유형 결정"""
        # 의도에 따른 기본 판단
        if intent == QueryIntent.UNSTRUCTURED_DATA_RETRIEVAL:
            return DataSourceType.CHROMADB_ONLY
        elif intent == QueryIntent.HYBRID_DATA_RETRIEVAL:
            return DataSourceType.ORACLE_CHROMADB
        elif intent == QueryIntent.DOCUMENT_SEARCH:
            return DataSourceType.CHROMADB_ONLY
        
        # 패턴 기반 판단
        source_scores = {}
        for source_type, patterns in self.data_source_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            
            if score > 0:
                source_scores[source_type] = score
        
        if not source_scores:
            return DataSourceType.ORACLE_ONLY  # 기본값
        
        return max(source_scores, key=source_scores.get)
    
    def _determine_visualization_type(self, query: str, intent: QueryIntent) -> VisualizationType:
        """시각화 유형 결정"""
        # 명시적 시각화 요청 확인
        for viz_type, patterns in self.visualization_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return viz_type
        
        # 의도에 따른 추론
        if intent == QueryIntent.CHART_GENERATION:
            # 데이터 특성에 따른 기본 차트 유형 결정
            if any(word in query for word in ['추이', '트렌드', '시계열', 'trend', 'time']):
                return VisualizationType.LINE_CHART
            elif any(word in query for word in ['비율', '구성', '분포', 'ratio', 'proportion']):
                return VisualizationType.PIE_CHART
            elif any(word in query for word in ['비교', '순위', 'comparison', 'ranking']):
                return VisualizationType.BAR_CHART
            else:
                return VisualizationType.BAR_CHART  # 기본값
        
        elif intent == QueryIntent.TABLE_GENERATION:
            return VisualizationType.TABLE
        
        elif intent in [QueryIntent.DATA_ANALYSIS, QueryIntent.STATISTICAL_ANALYSIS]:
            return VisualizationType.BAR_CHART  # 분석 결과는 주로 막대 차트
        
        return VisualizationType.NO_VISUALIZATION
    
    def _analyze_enhanced_complexity(self, query: str, data_source_type: DataSourceType) -> QueryComplexity:
        """향상된 복잡도 분석"""
        complexity_indicators = {
            QueryComplexity.ADVANCED_ANALYTICS: [
                r'머신러닝', r'예측', r'모델', r'클러스터', r'분류', r'회귀',
                r'machine.*learning', r'prediction', r'model', r'cluster', r'classification'
            ],
            QueryComplexity.MULTI_SOURCE: [
                r'문서.*함께', r'원문.*포함', r'조인.*문서', r'관련.*문서',
                r'with.*document', r'including.*text'
            ],
            QueryComplexity.COMPLEX: [
                r'조인', r'서브쿼리', r'중첩', r'계층', r'윈도우.*함수',
                r'join', r'subquery', r'nested', r'window.*function'
            ],
            QueryComplexity.MODERATE: [
                r'그룹', r'집계', r'정렬', r'필터', r'조건',
                r'group', r'aggregate', r'order', r'filter', r'condition'
            ]
        }
        
        # 데이터 소스에 따른 복잡도 조정
        if data_source_type == DataSourceType.ORACLE_CHROMADB:
            return QueryComplexity.MULTI_SOURCE
        
        # 패턴 기반 복잡도 결정
        for complexity, patterns in complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return complexity
        
        return QueryComplexity.SIMPLE
    
    def _extract_enhanced_entities(self, query: str) -> Dict[str, List[str]]:
        """향상된 엔터티 추출"""
        entities = {
            'oracle_tables': [],
            'chromadb_collections': [],
            'time_conditions': [],
            'numeric_conditions': [],
            'text_search_terms': [],
            'visualization_params': [],
            'aggregation_functions': []
        }
        
        # Oracle 테이블 매칭
        for table_name, comment in self.oracle_tables.items():
            table_lower = table_name.lower()
            comment_lower = (comment or "").lower()
            
            for word in query.split():
                if len(word) > 2:
                    if word in table_lower or word in comment_lower:
                        entities['oracle_tables'].append(table_name)
                        break
        
        # ChromaDB 컬렉션 매칭
        for collection in self.chromadb_collections:
            if collection.lower() in query:
                entities['chromadb_collections'].append(collection)
        
        # 텍스트 검색 키워드 추출
        text_search_patterns = [
            r'"([^"]+)"',  # 따옴표로 둘러싸인 텍스트
            r'검색.*?([가-힣\w]+)',  # '검색' 뒤의 키워드
            r'찾기.*?([가-힣\w]+)',  # '찾기' 뒤의 키워드
        ]
        
        for pattern in text_search_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities['text_search_terms'].extend(matches)
        
        # 집계 함수 추출
        aggregation_patterns = [
            r'합계', r'평균', r'최대', r'최소', r'개수', r'카운트',
            r'sum', r'average', r'avg', r'max', r'min', r'count'
        ]
        
        for pattern in aggregation_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                entities['aggregation_functions'].append(pattern)
        
        # 시각화 파라미터 추출
        viz_params = []
        if 'x축' in query or 'x-axis' in query.lower():
            viz_params.append('x_axis_specified')
        if 'y축' in query or 'y-axis' in query.lower():
            viz_params.append('y_axis_specified')
        if '색상' in query or 'color' in query.lower():
            viz_params.append('color_specified')
        
        entities['visualization_params'] = viz_params
        
        # 중복 제거
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _needs_oracle(self, data_source_type: DataSourceType, entities: Dict[str, List[str]]) -> bool:
        """Oracle DB 필요성 판단"""
        return data_source_type in [
            DataSourceType.ORACLE_ONLY, 
            DataSourceType.ORACLE_CHROMADB
        ] or bool(entities.get('oracle_tables'))
    
    def _needs_chromadb(self, data_source_type: DataSourceType, entities: Dict[str, List[str]]) -> bool:
        """ChromaDB 필요성 판단"""
        return data_source_type in [
            DataSourceType.CHROMADB_ONLY, 
            DataSourceType.ORACLE_CHROMADB
        ] or bool(entities.get('text_search_terms'))
    
    def _needs_enhanced_clarification(self, 
                                    query: str,
                                    intent: QueryIntent,
                                    data_source_type: DataSourceType,
                                    visualization_type: VisualizationType,
                                    entities: Dict[str, List[str]],
                                    confidence: float) -> Tuple[bool, List[str]]:
        """향상된 명확화 필요성 판단"""
        needs_clarification = False
        suggested_questions = []
        
        # 신뢰도가 낮은 경우
        if confidence < 0.6:
            needs_clarification = True
            suggested_questions.append("질문의 의도를 더 구체적으로 설명해주시겠어요?")
        
        # 데이터 소스가 불명확한 경우
        if data_source_type == DataSourceType.UNKNOWN:
            needs_clarification = True
            suggested_questions.append("정형 데이터(테이블)와 비정형 데이터(문서) 중 어떤 것을 조회하고 싶으신가요?")
        
        # Oracle 테이블이 명시되지 않은 경우
        if self._needs_oracle(data_source_type, entities) and not entities.get('oracle_tables'):
            needs_clarification = True
            suggested_questions.append("어떤 테이블의 데이터를 조회하고 싶으신가요?")
        
        # ChromaDB 검색어가 없는 경우
        if self._needs_chromadb(data_source_type, entities) and not entities.get('text_search_terms'):
            needs_clarification = True
            suggested_questions.append("문서에서 검색하고 싶은 키워드나 내용을 알려주시겠어요?")
        
        # 시각화 요청이지만 구체적이지 않은 경우
        if visualization_type == VisualizationType.NO_VISUALIZATION and intent == QueryIntent.CHART_GENERATION:
            needs_clarification = True
            suggested_questions.append("어떤 종류의 차트를 원하시나요? (라인, 막대, 파이 차트 등)")
        
        # 시각화 파라미터가 부족한 경우
        if visualization_type != VisualizationType.NO_VISUALIZATION:
            if not entities.get('visualization_params'):
                needs_clarification = True
                suggested_questions.append("차트의 X축과 Y축에 어떤 데이터를 표시하고 싶으신가요?")
        
        return needs_clarification, suggested_questions
    
    def _generate_visualization_config(self, 
                                     visualization_type: VisualizationType,
                                     entities: Dict[str, List[str]],
                                     intent: QueryIntent) -> Dict[str, Any]:
        """시각화 설정 생성"""
        config = {
            'type': visualization_type.value,
            'title': '',
            'x_axis': '',
            'y_axis': '',
            'color_scheme': 'default',
            'interactive': True,
            'show_legend': True
        }
        
        # 집계 함수에 따른 Y축 설정
        if entities.get('aggregation_functions'):
            agg_func = entities['aggregation_functions'][0]
            if agg_func in ['합계', 'sum']:
                config['y_axis'] = '합계'
            elif agg_func in ['평균', 'average', 'avg']:
                config['y_axis'] = '평균'
            elif agg_func in ['개수', 'count']:
                config['y_axis'] = '개수'
        
        # 시각화 유형별 기본 설정
        if visualization_type == VisualizationType.LINE_CHART:
            config.update({
                'show_markers': True,
                'line_style': 'solid',
                'fill_area': False
            })
        elif visualization_type == VisualizationType.PIE_CHART:
            config.update({
                'show_percentages': True,
                'explode_max': True,
                'start_angle': 90
            })
        elif visualization_type == VisualizationType.BAR_CHART:
            config.update({
                'orientation': 'vertical',
                'show_values': True,
                'sort_by': 'value'
            })
        
        return config
    
    def _generate_enhanced_reasoning(self, 
                                   query: str,
                                   intent: QueryIntent,
                                   data_source_type: DataSourceType,
                                   visualization_type: VisualizationType,
                                   complexity: QueryComplexity,
                                   entities: Dict[str, List[str]],
                                   confidence: float) -> str:
        """향상된 추론 과정 설명"""
        reasoning_parts = []
        
        reasoning_parts.append(f"쿼리 분석: '{query}'")
        reasoning_parts.append(f"의도: {intent.value} (신뢰도: {confidence:.2f})")
        reasoning_parts.append(f"데이터 소스: {data_source_type.value}")
        reasoning_parts.append(f"시각화: {visualization_type.value}")
        reasoning_parts.append(f"복잡도: {complexity.value}")
        
        if entities.get('oracle_tables'):
            reasoning_parts.append(f"Oracle 테이블: {', '.join(entities['oracle_tables'][:3])}")
        
        if entities.get('text_search_terms'):
            reasoning_parts.append(f"검색 키워드: {', '.join(entities['text_search_terms'][:3])}")
        
        if entities.get('aggregation_functions'):
            reasoning_parts.append(f"집계 함수: {', '.join(entities['aggregation_functions'])}")
        
        return " | ".join(reasoning_parts)
    
    def _llm_classify_enhanced_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """LLM을 사용한 향상된 의도 분류"""
        if not self.llm:
            return QueryIntent.UNCLEAR, 0.3
        
        prompt = f"""
다음 자연어 질의의 의도를 분석해주세요.

질의: "{query}"

의도 분류:
1. structured_data_retrieval: 정형 데이터(테이블) 조회
2. unstructured_data_retrieval: 비정형 데이터(문서) 조회/검색
3. hybrid_data_retrieval: 정형+비정형 데이터 결합 조회
4. data_analysis: 데이터 분석/집계/통계
5. chart_generation: 차트/그래프 생성 요청
6. table_generation: 표 형태로 정리 요청
7. document_search: 문서 검색
8. schema_inquiry: 데이터베이스/스키마 구조 문의
9. general_question: 일반적인 질문
10. greeting: 인사
11. help_request: 도움말 요청
12. unclear: 의도 불분명

응답 형식: intent_type,confidence_score
예시: structured_data_retrieval,0.85
"""
        
        try:
            response = self.llm.complete(prompt)
            result = response.text.strip()
            
            if ',' in result:
                intent_str, confidence_str = result.split(',')
                intent = QueryIntent(intent_str.strip())
                confidence = float(confidence_str.strip())
                return intent, confidence
            
        except Exception as e:
            logger.warning(f"LLM 의도 분류 실패: {e}")
        
        return