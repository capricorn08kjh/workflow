"""
명확화 관리 모듈
사용자와의 대화를 통해 쿼리를 명확화하고 필요한 정보를 수집
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from llama_index.llms.openai_like import OpenAILike

logger = logging.getLogger(__name__)

class ClarificationStatus(Enum):
    """명확화 상태"""
    PENDING = "pending"      # 명확화 대기 중
    IN_PROGRESS = "in_progress"  # 명확화 진행 중
    COMPLETED = "completed"  # 명확화 완료
    FAILED = "failed"        # 명확화 실패

class QuestionType(Enum):
    """질문 유형"""
    TABLE_SELECTION = "table_selection"    # 테이블 선택
    COLUMN_SELECTION = "column_selection"  # 컬럼 선택
    CONDITION_CLARIFICATION = "condition_clarification"  # 조건 명확화
    TIME_RANGE = "time_range"             # 시간 범위
    AGGREGATION_TYPE = "aggregation_type" # 집계 방식
    SORT_ORDER = "sort_order"             # 정렬 방식
    GENERAL_INFO = "general_info"         # 일반 정보

@dataclass
class ClarificationQuestion:
    """명확화 질문"""
    question_id: str
    question_type: QuestionType
    question_text: str
    options: Optional[List[str]] = None  # 선택지
    is_required: bool = True
    answered: bool = False
    answer: Optional[str] = None
    validation_pattern: Optional[str] = None  # 답변 검증 패턴

@dataclass
class ClarificationSession:
    """명확화 세션"""
    session_id: str
    original_query: str
    status: ClarificationStatus = ClarificationStatus.PENDING
    questions: List[ClarificationQuestion] = field(default_factory=list)
    answers: Dict[str, str] = field(default_factory=dict)
    refined_query: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    max_rounds: int = 5
    current_round: int = 0

class ClarificationManager:
    """명확화 관리자"""
    
    def __init__(self, 
                 llm: Optional[OpenAILike] = None,
                 available_tables: Optional[Dict[str, str]] = None,
                 table_schemas: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        초기화
        
        Args:
            llm: LLM 인스턴스
            available_tables: 사용 가능한 테이블 정보
            table_schemas: 테이블 스키마 정보
        """
        self.llm = llm
        self.available_tables = available_tables or {}
        self.table_schemas = table_schemas or {}
        self.active_sessions: Dict[str, ClarificationSession] = {}
    
    def start_clarification(self, 
                           session_id: str,
                           original_query: str,
                           analysis_result: Any) -> ClarificationSession:
        """
        명확화 세션 시작
        
        Args:
            session_id: 세션 ID
            original_query: 원본 쿼리
            analysis_result: 의도 분석 결과
            
        Returns:
            ClarificationSession: 명확화 세션
        """
        session = ClarificationSession(
            session_id=session_id,
            original_query=original_query,
            status=ClarificationStatus.IN_PROGRESS
        )
        
        # 분석 결과를 바탕으로 질문 생성
        questions = self._generate_questions(original_query, analysis_result)
        session.questions = questions
        
        self.active_sessions[session_id] = session
        
        logger.info(f"명확화 세션 시작: {session_id}, 질문 {len(questions)}개")
        return session
    
    def get_next_question(self, session_id: str) -> Optional[ClarificationQuestion]:
        """
        다음 질문 가져오기
        
        Args:
            session_id: 세션 ID
            
        Returns:
            ClarificationQuestion: 다음 질문 또는 None
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        # 답변되지 않은 첫 번째 필수 질문 찾기
        for question in session.questions:
            if not question.answered and question.is_required:
                return question
        
        # 필수 질문이 모두 답변되면 선택적 질문 찾기
        for question in session.questions:
            if not question.answered:
                return question
        
        # 모든 질문이 답변되면 세션 완료
        session.status = ClarificationStatus.COMPLETED
        return None
    
    def answer_question(self, 
                       session_id: str, 
                       question_id: str, 
                       answer: str) -> Tuple[bool, str]:
        """
        질문에 답변
        
        Args:
            session_id: 세션 ID
            question_id: 질문 ID
            answer: 답변
            
        Returns:
            Tuple[bool, str]: (성공 여부, 메시지)
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return False, "세션을 찾을 수 없습니다."
        
        # 질문 찾기
        question = None
        for q in session.questions:
            if q.question_id == question_id:
                question = q
                break
        
        if not question:
            return False, "질문을 찾을 수 없습니다."
        
        # 답변 검증
        is_valid, validation_message = self._validate_answer(question, answer)
        if not is_valid:
            return False, validation_message
        
        # 답변 저장
        question.answered = True
        question.answer = answer
        session.answers[question_id] = answer
        session.updated_at = datetime.now()
        session.current_round += 1
        
        logger.info(f"질문 답변 완료: {question_id} = {answer}")
        return True, "답변이 저장되었습니다."
    
    def is_clarification_complete(self, session_id: str) -> bool:
        """
        명확화 완료 여부 확인
        
        Args:
            session_id: 세션 ID
            
        Returns:
            bool: 완료 여부
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        # 모든 필수 질문이 답변되었는지 확인
        for question in session.questions:
            if question.is_required and not question.answered:
                return False
        
        return True
    
    def generate_refined_query(self, session_id: str) -> Optional[str]:
        """
        명확화된 정보를 바탕으로 정제된 쿼리 생성
        
        Args:
            session_id: 세션 ID
            
        Returns:
            Optional[str]: 정제된 쿼리
        """
        session = self.active_sessions.get(session_id)
        if not session or not self.is_clarification_complete(session_id):
            return None
        
        if self.llm:
            refined_query = self._llm_generate_refined_query(session)
        else:
            refined_query = self._rule_based_refined_query(session)
        
        session.refined_query = refined_query
        session.status = ClarificationStatus.COMPLETED
        
        logger.info(f"정제된 쿼리 생성 완료: {session_id}")
        return refined_query
    
    def _generate_questions(self, 
                           original_query: str, 
                           analysis_result: Any) -> List[ClarificationQuestion]:
        """분석 결과를 바탕으로 질문 생성"""
        questions = []
        
        # 테이블 선택 질문
        if not analysis_result.extracted_entities.get('table_hints'):
            table_options = self._get_relevant_table_options(original_query)
            if table_options:
                questions.append(ClarificationQuestion(
                    question_id="table_selection",
                    question_type=QuestionType.TABLE_SELECTION,
                    question_text="어떤 테이블의 데이터를 조회하고 싶으신가요?",
                    options=table_options,
                    is_required=True
                ))
        
        # 시간 범위 질문 (모호한 시간 표현이 있을 때)
        time_terms = ['최근', '지난', '이번']
        if any(term in original_query for term in time_terms):
            questions.append(ClarificationQuestion(
                question_id="time_range",
                question_type=QuestionType.TIME_RANGE,
                question_text="구체적인 시간 범위를 알려주세요. (예: 2024-01-01부터 2024-12-31까지, 최근 30일)",
                is_required=True,
                validation_pattern=r'\d{4}-\d{2}-\d{2}|\d+일|\d+개월|\d+년'
            ))
        
        # 집계 방식 질문 (분석 의도가 있을 때)
        if analysis_result.intent.value == 'data_analysis':
            questions.append(ClarificationQuestion(
                question_id="aggregation_type",
                question_type=QuestionType.AGGREGATION_TYPE,
                question_text="어떤 방식으로 집계하고 싶으신가요?",
                options=["합계", "평균", "개수", "최대값", "최소값", "그룹별 통계"],
                is_required=False
            ))
        
        # 정렬 방식 질문
        sort_terms = ['순위', '랭킹', '높은', '낮은', '많은', '적은']
        if any(term in original_query for term in sort_terms):
            questions.append(ClarificationQuestion(
                question_id="sort_order",
                question_type=QuestionType.SORT_ORDER,
                question_text="결과를 어떻게 정렬하고 싶으신가요?",
                options=["오름차순", "내림차순", "정렬 안함"],
                is_required=False
            ))
        
        # 조건 명확화 질문 (모호한 조건이 있을 때)
        ambiguous_terms = ['많은', '높은', '좋은', '큰', '작은']
        for term in ambiguous_terms:
            if term in original_query:
                questions.append(ClarificationQuestion(
                    question_id=f"condition_{term}",
                    question_type=QuestionType.CONDITION_CLARIFICATION,
                    question_text=f"'{term}'의 구체적인 기준을 알려주세요. (예: 100개 이상, 50% 이상)",
                    is_required=True,
                    validation_pattern=r'\d+|\d+%|\d+원|\d+개'
                ))
        
        return questions
    
    def _get_relevant_table_options(self, query: str) -> List[str]:
        """쿼리와 관련된 테이블 옵션 생성"""
        if not self.available_tables:
            return []
        
        query_lower = query.lower()
        relevant_tables = []
        
        # 키워드 기반 테이블 매칭
        for table_name, comment in self.available_tables.items():
            table_lower = table_name.lower()
            comment_lower = (comment or "").lower()
            
            # 테이블명이나 코멘트에서 키워드 매칭
            for word in query_lower.split():
                if len(word) > 2 and (word in table_lower or word in comment_lower):
                    display_name = f"{table_name} ({comment or '설명 없음'})"
                    if display_name not in relevant_tables:
                        relevant_tables.append(display_name)
                    break
        
        # 관련 테이블이 없으면 상위 10개 반환
        if not relevant_tables:
            for i, (table_name, comment) in enumerate(list(self.available_tables.items())[:10]):
                relevant_tables.append(f"{table_name} ({comment or '설명 없음'})")
        
        return relevant_tables[:10]  # 최대 10개까지
    
    def _validate_answer(self, question: ClarificationQuestion, answer: str) -> Tuple[bool, str]:
        """답변 검증"""
        if not answer.strip():
            return False, "답변을 입력해주세요."
        
        # 선택지가 있는 경우 선택지 내에서만 선택 가능
        if question.options:
            # 부분 매칭 허용
            answer_lower = answer.lower()
            for option in question.options:
                if answer_lower in option.lower() or option.lower() in answer_lower:
                    return True, "유효한 답변입니다."
            
            return False, f"다음 중에서 선택해주세요: {', '.join(question.options)}"
        
        # 패턴 검증
        if question.validation_pattern:
            import re
            if not re.search(question.validation_pattern, answer):
                return False, "올바른 형식으로 입력해주세요."
        
        return True, "유효한 답변입니다."
    
    def _llm_generate_refined_query(self, session: ClarificationSession) -> str:
        """LLM을 사용한 정제된 쿼리 생성"""
        if not self.llm:
            return session.original_query
        
        # 답변 정보 구성
        answers_text = []
        for question in session.questions:
            if question.answered:
                answers_text.append(f"- {question.question_text}: {question.answer}")
        
        prompt = f"""
사용자의 원본 질의와 명확화 답변을 바탕으로 정제된 자연어 질의를 생성해주세요.

원본 질의: "{session.original_query}"

명확화 답변:
{chr(10).join(answers_text)}

정제된 질의는 다음 조건을 만족해야 합니다:
1. 모호한 표현을 구체적으로 변경
2. 테이블명과 조건을 명확히 포함
3. 자연스러운 한국어 문장
4. SQL 변환이 용이한 구조

정제된 질의:
"""
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM 정제 쿼리 생성 실패: {e}")
            return self._rule_based_refined_query(session)
    
    def _rule_based_refined_query(self, session: ClarificationSession) -> str:
        """규칙 기반 정제된 쿼리 생성"""
        refined_parts = []
        original = session.original_query
        
        # 원본 쿼리를 기준으로 시작
        refined_query = original
        
        # 답변을 바탕으로 쿼리 수정
        for question in session.questions:
            if not question.answered:
                continue
            
            if question.question_type == QuestionType.TABLE_SELECTION:
                # 테이블 정보 추가
                table_info = question.answer.split('(')[0].strip()
                refined_query = f"{table_info} 테이블에서 " + refined_query
            
            elif question.question_type == QuestionType.TIME_RANGE:
                # 시간 범위 명확화
                time_keywords = ['최근', '지난', '이번']
                for keyword in time_keywords:
                    if keyword in refined_query:
                        refined_query = refined_query.replace(keyword, question.answer)
                        break
            
            elif question.question_type == QuestionType.CONDITION_CLARIFICATION:
                # 조건 명확화
                ambiguous_terms = ['많은', '높은', '좋은', '큰', '작은']
                for term in ambiguous_terms:
                    if term in refined_query:
                        refined_query = refined_query.replace(term, f"{term}({question.answer})")
                        break
            
            elif question.question_type == QuestionType.SORT_ORDER:
                # 정렬 조건 추가
                if question.answer != "정렬 안함":
                    refined_query += f" {question.answer}으로 정렬"
        
        return refined_query
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 요약 정보 반환"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        answered_questions = [q for q in session.questions if q.answered]
        total_questions = len(session.questions)
        
        return {
            'session_id': session_id,
            'original_query': session.original_query,
            'status': session.status.value,
            'progress': f"{len(answered_questions)}/{total_questions}",
            'refined_query': session.refined_query,
            'answers': session.answers,
            'created_at': session.created_at.isoformat(),
            'updated_at': session.updated_at.isoformat()
        }
    
    def close_session(self, session_id: str) -> bool:
        """세션 종료"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"명확화 세션 종료: {session_id}")
            return True
        return False
    
    def get_active_sessions(self) -> List[str]:
        """활성 세션 목록 반환"""
        return list(self.active_sessions.keys())
    
    def format_question_for_user(self, question: ClarificationQuestion) -> str:
        """사용자에게 보여줄 질문 형식화"""
        formatted = f"❓ {question.question_text}\n"
        
        if question.options:
            formatted += "\n선택 옵션:\n"
            for i, option in enumerate(question.options, 1):
                formatted += f"  {i}. {option}\n"
            formatted += "\n번호나 옵션명을 입력해주세요."
        
        if question.validation_pattern:
            if question.question_type == QuestionType.TIME_RANGE:
                formatted += "\n💡 예시: 2024-01-01부터 2024-12-31까지, 최근 30일, 지난 3개월"
            elif question.question_type == QuestionType.CONDITION_CLARIFICATION:
                formatted += "\n💡 예시: 100개 이상, 50% 이상, 1000원 이상"
        
        return formatted
    
    def process_user_input(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """
        사용자 입력 처리
        
        Args:
            session_id: 세션 ID
            user_input: 사용자 입력
            
        Returns:
            Dict: 처리 결과
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return {
                'success': False,
                'message': '활성 세션이 없습니다.',
                'next_question': None,
                'is_complete': False
            }
        
        # 현재 질문 가져오기
        current_question = self.get_next_question(session_id)
        if not current_question:
            return {
                'success': True,
                'message': '모든 질문이 완료되었습니다.',
                'next_question': None,
                'is_complete': True
            }
        
        # 답변 처리
        success, message = self.answer_question(
            session_id, current_question.question_id, user_input
        )
        
        if not success:
            return {
                'success': False,
                'message': message,
                'next_question': current_question,
                'is_complete': False
            }
        
        # 다음 질문 확인
        next_question = self.get_next_question(session_id)
        is_complete = next_question is None
        
        return {
            'success': True,
            'message': message,
            'next_question': next_question,
            'is_complete': is_complete
        }