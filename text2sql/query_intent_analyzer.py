"""
쿼리 의도 분석 모듈
자연어 질의의 의도를 분석하고 Text2SQL이 필요한지 판단
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from llama_index.llms.openai_like import OpenAILike

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """쿼리 의도 분류"""
    DATA_RETRIEVAL = "data_retrieval"  # 데이터 조회
    DATA_ANALYSIS = "data_analysis"    # 데이터 분석/집계
    SCHEMA_INQUIRY = "schema_inquiry"   # 스키마/구조 문의
    GENERAL_QUESTION = "general_question"  # 일반적인 질문
    GREETING = "greeting"              # 인사
    HELP_REQUEST = "help_request"      # 도움말 요청
    UNCLEAR = "unclear"                # 의도 불분명

class QueryComplexity(Enum):
    """쿼리 복잡도"""
    SIMPLE = "simple"      # 단순 조회
    MODERATE = "moderate"  # 중간 복잡도 (조인, 집계 등)
    COMPLEX = "complex"    # 복잡한 분석

@dataclass
class IntentAnalysisResult:
    """의도 분석 결과"""
    intent: QueryIntent
    complexity: QueryComplexity
    confidence: float  # 0.0 ~ 1.0
    needs_sql: bool
    needs_clarification: bool
    suggested_questions: List[str]
    extracted_entities: Dict[str, List[str]]  # 테이블명, 컬럼명, 조건 등
    reasoning: str

class QueryIntentAnalyzer:
    """쿼리 의도 분석기"""
    
    def __init__(self, 
                 llm: Optional[OpenAILike] = None,
                 available_tables: Optional[Dict[str, str]] = None):
        """
        초기화
        
        Args:
            llm: LLM 인스턴스 (고급 분석용)
            available_tables: 사용 가능한 테이블 정보
        """
        self.llm = llm
        self.available_tables = available_tables or {}
        
        # 의도별 키워드 패턴
        self.intent_patterns = {
            QueryIntent.DATA_RETRIEVAL: [
                r'보여주세요', r'조회', r'찾아', r'가져와', r'출력', r'리스트',
                r'목록', r'데이터', r'정보', r'내역', r'현황', r'상태',
                r'show', r'select', r'get', r'find', r'list', r'display'
            ],
            QueryIntent.DATA_ANALYSIS: [
                r'분석', r'통계', r'집계', r'합계', r'평균', r'최대', r'최소',
                r'개수', r'카운트', r'그룹', r'순위', r'랭킹', r'비율', r'퍼센트',
                r'분포', r'추이', r'트렌드', r'비교',
                r'analyze', r'statistics', r'aggregate', r'sum', r'average',
                r'count', r'group', r'rank', r'percentage', r'trend'
            ],
            QueryIntent.SCHEMA_INQUIRY: [
                r'테이블', r'컬럼', r'스키마', r'구조', r'어떤.*있', r'무엇.*있',
                r'어떤.*테이블', r'무슨.*컬럼', r'데이터베이스.*구조',
                r'table', r'column', r'schema', r'structure', r'what.*table',
                r'which.*column', r'database.*structure'
            ],
            QueryIntent.GREETING: [
                r'안녕', r'하이', r'헬로', r'반가', r'처음',
                r'hello', r'hi', r'hey', r'good.*morning', r'good.*afternoon'
            ],
            QueryIntent.HELP_REQUEST: [
                r'도움', r'도와', r'헬프', r'사용법', r'어떻게', r'방법',
                r'help', r'assist', r'how.*to', r'what.*can', r'usage'
            ]
        }
        
        # 복잡도 판단 키워드
        self.complexity_patterns = {
            QueryComplexity.COMPLEX: [
                r'조인', r'join', r'연결', r'관계', r'상관관계', r'여러.*테이블',
                r'서브쿼리', r'subquery', r'중첩', r'계층', r'재귀',
                r'윈도우.*함수', r'window.*function', r'파티션', r'partition'
            ],
            QueryComplexity.MODERATE: [
                r'그룹', r'group', r'집계', r'aggregate', r'정렬', r'order',
                r'필터', r'filter', r'조건', r'condition', r'범위', r'range',
                r'합계', r'sum', r'평균', r'average', r'개수', r'count'
            ]
        }
        
        # 엔터티 추출 패턴
        self.entity_patterns = {
            'table_hints': [
                r'사용자', r'회원', r'고객', r'user', r'customer', r'member',
                r'직원', r'사원', r'employee', r'staff', r'worker',
                r'부서', r'조직', r'department', r'division', r'team',
                r'제품', r'상품', r'product', r'item', r'goods',
                r'주문', r'order', r'purchase', r'transaction',
                r'매출', r'판매', r'sales', r'revenue',
                r'재고', r'inventory', r'stock'
            ],
            'time_conditions': [
                r'\d{4}년', r'\d{1,2}월', r'\d{1,2}일',
                r'오늘', r'어제', r'yesterday', r'today',
                r'최근', r'recent', r'이번', r'지난', r'last',
                r'월간', r'월별', r'monthly', r'연간', r'yearly'
            ],
            'numeric_conditions': [
                r'\d+이상', r'\d+이하', r'\d+보다', r'상위\s*\d+',
                r'하위\s*\d+', r'최대\s*\d+', r'최소\s*\d+',
                r'>=\s*\d+', r'<=\s*\d+', r'>\s*\d+', r'<\s*\d+',
                r'top\s*\d+', r'bottom\s*\d+'
            ]
        }
    
    def analyze_intent(self, query: str) -> IntentAnalysisResult:
        """
        쿼리 의도 분석
        
        Args:
            query: 자연어 쿼리
            
        Returns:
            IntentAnalysisResult: 분석 결과
        """
        query_lower = query.lower().strip()
        
        # 1. 기본 의도 분류
        intent, intent_confidence = self._classify_intent(query_lower)
        
        # 2. 복잡도 분석
        complexity = self._analyze_complexity(query_lower)
        
        # 3. 엔터티 추출
        entities = self._extract_entities(query_lower)
        
        # 4. SQL 필요성 판단
        needs_sql = self._needs_sql_conversion(intent, entities)
        
        # 5. 명확화 필요성 판단
        needs_clarification, suggested_questions = self._needs_clarification(
            query_lower, intent, entities, intent_confidence
        )
        
        # 6. 추론 과정 설명
        reasoning = self._generate_reasoning(
            query, intent, complexity, entities, intent_confidence
        )
        
        return IntentAnalysisResult(
            intent=intent,
            complexity=complexity,
            confidence=intent_confidence,
            needs_sql=needs_sql,
            needs_clarification=needs_clarification,
            suggested_questions=suggested_questions,
            extracted_entities=entities,
            reasoning=reasoning
        )
    
    def _classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """의도 분류 및 신뢰도 계산"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            # 키워드 매칭이 안되면 LLM으로 분석
            if self.llm:
                return self._llm_classify_intent(query)
            else:
                return QueryIntent.UNCLEAR, 0.3
        
        # 가장 높은 점수의 의도 선택
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        total_score = sum(intent_scores.values())
        
        confidence = min(max_score / max(total_score, 1), 1.0)
        
        return best_intent, confidence
    
    def _analyze_complexity(self, query: str) -> QueryComplexity:
        """쿼리 복잡도 분석"""
        for complexity, patterns in self.complexity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return complexity
        
        # 기본값은 단순 쿼리
        return QueryComplexity.SIMPLE
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """엔터티 추출 (테이블명, 조건 등)"""
        entities = {
            'table_hints': [],
            'time_conditions': [],
            'numeric_conditions': [],
            'columns': [],
            'specific_values': []
        }
        
        # 패턴 기반 엔터티 추출
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                entities[entity_type].extend(matches)
        
        # 사용 가능한 테이블과 매칭
        if self.available_tables:
            for table_name, comment in self.available_tables.items():
                table_lower = table_name.lower()
                comment_lower = (comment or "").lower()
                
                # 테이블명이나 코멘트가 쿼리에 포함되어 있는지 확인
                for word in query.split():
                    if len(word) > 2:
                        if word in table_lower or word in comment_lower:
                            entities['table_hints'].append(table_name)
        
        # 중복 제거
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _needs_sql_conversion(self, intent: QueryIntent, entities: Dict[str, List[str]]) -> bool:
        """SQL 변환 필요성 판단"""
        sql_required_intents = [
            QueryIntent.DATA_RETRIEVAL,
            QueryIntent.DATA_ANALYSIS
        ]
        
        # 의도가 데이터 조회/분석이고 테이블 힌트가 있으면 SQL 필요
        if intent in sql_required_intents:
            return True
        
        # 스키마 문의의 경우 테이블 정보 조회만 필요
        if intent == QueryIntent.SCHEMA_INQUIRY:
            return False
        
        return False
    
    def _needs_clarification(self, 
                           query: str, 
                           intent: QueryIntent, 
                           entities: Dict[str, List[str]], 
                           confidence: float) -> Tuple[bool, List[str]]:
        """명확화 필요성 판단 및 질문 제안"""
        suggested_questions = []
        needs_clarification = False
        
        # 신뢰도가 낮으면 명확화 필요
        if confidence < 0.6:
            needs_clarification = True
            suggested_questions.append("질문의 의도를 좀 더 구체적으로 설명해주시겠어요?")
        
        # SQL이 필요한데 테이블 힌트가 없으면 명확화 필요
        if intent in [QueryIntent.DATA_RETRIEVAL, QueryIntent.DATA_ANALYSIS]:
            if not entities.get('table_hints'):
                needs_clarification = True
                suggested_questions.append("어떤 테이블의 데이터를 조회하고 싶으신가요?")
        
        # 모호한 조건들 체크
        ambiguous_terms = ['최근', '많은', '높은', '좋은', '큰']
        for term in ambiguous_terms:
            if term in query:
                needs_clarification = True
                if term == '최근':
                    suggested_questions.append("'최근'은 구체적으로 언제부터를 말씀하시는 건가요? (예: 최근 1주일, 1개월)")
                elif term in ['많은', '높은', '좋은', '큰']:
                    suggested_questions.append(f"'{term}'의 기준을 구체적으로 알려주시겠어요?")
        
        # 집계 관련 질문에서 그룹화 기준이 없으면
        if intent == QueryIntent.DATA_ANALYSIS:
            analysis_terms = ['평균', '합계', '개수', '통계']
            if any(term in query for term in analysis_terms):
                group_terms = ['별로', '별', '그룹', '분류']
                if not any(term in query for term in group_terms):
                    needs_clarification = True
                    suggested_questions.append("어떤 기준으로 그룹화해서 분석하고 싶으신가요?")
        
        return needs_clarification, suggested_questions
    
    def _generate_reasoning(self, 
                          query: str, 
                          intent: QueryIntent, 
                          complexity: QueryComplexity,
                          entities: Dict[str, List[str]], 
                          confidence: float) -> str:
        """분석 추론 과정 설명"""
        reasoning_parts = []
        
        reasoning_parts.append(f"쿼리 분석: '{query}'")
        reasoning_parts.append(f"감지된 의도: {intent.value} (신뢰도: {confidence:.2f})")
        reasoning_parts.append(f"복잡도: {complexity.value}")
        
        if entities.get('table_hints'):
            reasoning_parts.append(f"관련 테이블 후보: {', '.join(entities['table_hints'][:3])}")
        
        if entities.get('time_conditions'):
            reasoning_parts.append(f"시간 조건: {', '.join(entities['time_conditions'])}")
        
        if entities.get('numeric_conditions'):
            reasoning_parts.append(f"숫자 조건: {', '.join(entities['numeric_conditions'])}")
        
        return " | ".join(reasoning_parts)
    
    def _llm_classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """LLM을 사용한 고급 의도 분류"""
        if not self.llm:
            return QueryIntent.UNCLEAR, 0.3
        
        prompt = f"""
다음 자연어 질의의 의도를 분석해주세요.

질의: "{query}"

의도 분류:
1. data_retrieval: 데이터 조회/검색
2. data_analysis: 데이터 분석/집계/통계
3. schema_inquiry: 데이터베이스 구조/스키마 문의
4. general_question: 일반적인 질문
5. greeting: 인사
6. help_request: 도움말 요청
7. unclear: 의도 불분명

응답 형식: intent_type,confidence_score
예시: data_retrieval,0.85
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
        
        return QueryIntent.UNCLEAR, 0.3
    
    def get_available_table_summary(self) -> str:
        """사용 가능한 테이블 요약"""
        if not self.available_tables:
            return "사용 가능한 테이블 정보가 없습니다."
        
        summary = f"총 {len(self.available_tables)}개의 테이블이 사용 가능합니다.\n\n"
        summary += "주요 테이블:\n"
        
        # 상위 10개 테이블만 표시
        for i, (table_name, comment) in enumerate(list(self.available_tables.items())[:10]):
            summary += f"- {table_name}: {comment or '설명 없음'}\n"
        
        if len(self.available_tables) > 10:
            summary += f"... 외 {len(self.available_tables) - 10}개 테이블"
        
        return summary