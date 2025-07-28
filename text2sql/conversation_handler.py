"""
대화 처리 모듈
사용자와의 전체적인 대화 흐름을 관리하고 조정
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from query_intent_analyzer import QueryIntentAnalyzer, QueryIntent, IntentAnalysisResult
from clarification_manager import ClarificationManager, ClarificationStatus
from text2sql_oracle import Text2SQLOracle

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """대화 상태"""
    IDLE = "idle"                    # 대기 상태
    ANALYZING = "analyzing"          # 의도 분석 중
    CLARIFYING = "clarifying"        # 명확화 진행 중
    PROCESSING = "processing"        # SQL 처리 중
    COMPLETED = "completed"          # 완료
    ERROR = "error"                  # 오류

@dataclass
class ConversationContext:
    """대화 컨텍스트"""
    session_id: str
    state: ConversationState
    original_query: Optional[str] = None
    analysis_result: Optional[IntentAnalysisResult] = None
    clarification_session_id: Optional[str] = None
    final_query: Optional[str] = None
    sql_result: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

class ConversationHandler:
    """대화 처리기"""
    
    def __init__(self, 
                 text2sql_system: Text2SQLOracle,
                 intent_analyzer: QueryIntentAnalyzer,
                 clarification_manager: ClarificationManager):
        """
        초기화
        
        Args:
            text2sql_system: Text2SQL 시스템
            intent_analyzer: 의도 분석기
            clarification_manager: 명확화 관리자
        """
        self.text2sql_system = text2sql_system
        self.intent_analyzer = intent_analyzer
        self.clarification_manager = clarification_manager
        
        # 활성 대화 컨텍스트
        self.active_contexts: Dict[str, ConversationContext] = {}
        
        logger.info("대화 처리기 초기화 완료")
    
    def start_conversation(self, user_id: str = None) -> str:
        """
        새로운 대화 세션 시작
        
        Args:
            user_id: 사용자 ID (선택적)
            
        Returns:
            str: 세션 ID
        """
        session_id = f"{user_id or 'user'}_{uuid.uuid4().hex[:8]}"
        
        context = ConversationContext(
            session_id=session_id,
            state=ConversationState.IDLE
        )
        
        self.active_contexts[session_id] = context
        
        logger.info(f"새 대화 세션 시작: {session_id}")
        return session_id
    
    def process_user_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        사용자 메시지 처리
        
        Args:
            session_id: 세션 ID
            message: 사용자 메시지
            
        Returns:
            Dict: 처리 결과
        """
        context = self.active_contexts.get(session_id)
        if not context:
            return {
                'success': False,
                'message': '세션을 찾을 수 없습니다. 새 대화를 시작해주세요.',
                'session_id': session_id,
                'state': 'error'
            }
        
        try:
            # 현재 상태에 따른 처리
            if context.state == ConversationState.IDLE:
                return self._handle_initial_query(context, message)
            
            elif context.state == ConversationState.CLARIFYING:
                return self._handle_clarification_response(context, message)
            
            elif context.state in [ConversationState.COMPLETED, ConversationState.ERROR]:
                # 새로운 질의로 처리
                context.state = ConversationState.IDLE
                return self._handle_initial_query(context, message)
            
            else:
                return {
                    'success': False,
                    'message': f'현재 상태({context.state.value})에서는 메시지를 처리할 수 없습니다.',
                    'session_id': session_id,
                    'state': context.state.value
                }
                
        except Exception as e:
            logger.error(f"메시지 처리 중 오류: {e}")
            context.state = ConversationState.ERROR
            return {
                'success': False,
                'message': f'처리 중 오류가 발생했습니다: {str(e)}',
                'session_id': session_id,
                'state': 'error'
            }
    
    def _handle_initial_query(self, context: ConversationContext, query: str) -> Dict[str, Any]:
        """초기 쿼리 처리"""
        context.original_query = query
        context.state = ConversationState.ANALYZING
        context.updated_at = datetime.now()
        
        logger.info(f"초기 쿼리 분석 시작: {query}")
        
        # 1. 의도 분석
        analysis_result = self.intent_analyzer.analyze_intent(query)
        context.analysis_result = analysis_result
        
        logger.info(f"의도 분석 결과: {analysis_result.intent.value}, 신뢰도: {analysis_result.confidence:.2f}")
        
        # 2. 인사말이나 도움말 요청 처리
        if analysis_result.intent in [QueryIntent.GREETING, QueryIntent.HELP_REQUEST]:
            return self._handle_non_sql_query(context, analysis_result)
        
        # 3. 스키마 문의 처리
        if analysis_result.intent == QueryIntent.SCHEMA_INQUIRY:
            return self._handle_schema_inquiry(context, analysis_result)
        
        # 4. 일반적인 질문 처리
        if analysis_result.intent == QueryIntent.GENERAL_QUESTION:
            return self._handle_general_question(context, analysis_result)
        
        # 5. SQL이 필요하지 않은 경우
        if not analysis_result.needs_sql:
            return self._handle_non_sql_query(context, analysis_result)
        
        # 6. 명확화가 필요한 경우
        if analysis_result.needs_clarification:
            return self._start_clarification(context, analysis_result)
        
        # 7. 바로 SQL 실행 가능한 경우
        return self._execute_sql_query(context, query)
    
    def _handle_clarification_response(self, context: ConversationContext, response: str) -> Dict[str, Any]:
        """명확화 응답 처리"""
        if not context.clarification_session_id:
            logger.error("명확화 세션 ID가 없습니다.")
            context.state = ConversationState.ERROR
            return {
                'success': False,
                'message': '명확화 세션 정보가 없습니다.',
                'session_id': context.session_id,
                'state': 'error'
            }
        
        # 명확화 관리자를 통해 응답 처리
        result = self.clarification_manager.process_user_input(
            context.clarification_session_id, response
        )
        
        if not result['success']:
            return {
                'success': False,
                'message': result['message'],
                'session_id': context.session_id,
                'state': 'clarifying',
                'question': self.clarification_manager.format_question_for_user(result['next_question']) if result['next_question'] else None
            }
        
        # 명확화가 완료된 경우
        if result['is_complete']:
            return self._complete_clarification(context)
        
        # 다음 질문이 있는 경우
        if result['next_question']:
            return {
                'success': True,
                'message': '답변이 저장되었습니다.',
                'session_id': context.session_id,
                'state': 'clarifying',
                'question': self.clarification_manager.format_question_for_user(result['next_question'])
            }
        
        return {
            'success': True,
            'message': '명확화가 완료되었습니다.',
            'session_id': context.session_id,
            'state': 'clarifying'
        }
    
    def _start_clarification(self, context: ConversationContext, analysis_result: IntentAnalysisResult) -> Dict[str, Any]:
        """명확화 시작"""
        clarification_session_id = f"clarif_{context.session_id}_{uuid.uuid4().hex[:6]}"
        context.clarification_session_id = clarification_session_id
        context.state = ConversationState.CLARIFYING
        
        # 명확화 세션 시작
        clarification_session = self.clarification_manager.start_clarification(
            clarification_session_id, context.original_query, analysis_result
        )
        
        # 첫 번째 질문 가져오기
        first_question = self.clarification_manager.get_next_question(clarification_session_id)
        
        if not first_question:
            logger.warning("명확화 질문이 생성되지 않았습니다.")
            return self._execute_sql_query(context, context.original_query)
        
        response_message = "질문을 더 정확히 이해하기 위해 몇 가지 질문을 드리겠습니다.\n\n"
        response_message += f"**분석 결과:** {analysis_result.reasoning}\n\n"
        
        if analysis_result.suggested_questions:
            response_message += "**추천 질문들:**\n"
            for q in analysis_result.suggested_questions:
                response_message += f"- {q}\n"
            response_message += "\n"
        
        return {
            'success': True,
            'message': response_message,
            'session_id': context.session_id,
            'state': 'clarifying',
            'question': self.clarification_manager.format_question_for_user(first_question),
            'analysis': {
                'intent': analysis_result.intent.value,
                'confidence': analysis_result.confidence,
                'complexity': analysis_result.complexity.value
            }
        }
    
    def _complete_clarification(self, context: ConversationContext) -> Dict[str, Any]:
        """명확화 완료 처리"""
        # 정제된 쿼리 생성
        refined_query = self.clarification_manager.generate_refined_query(
            context.clarification_session_id
        )
        
        if not refined_query:
            logger.error("정제된 쿼리 생성 실패")
            context.state = ConversationState.ERROR
            return {
                'success': False,
                'message': '정제된 쿼리 생성에 실패했습니다.',
                'session_id': context.session_id,
                'state': 'error'
            }
        
        context.final_query = refined_query
        
        # 명확화 세션 정리
        session_summary = self.clarification_manager.get_session_summary(
            context.clarification_session_id
        )
        
        response_message = "명확화가 완료되었습니다!\n\n"
        response_message += f"**정제된 질의:** {refined_query}\n\n"
        response_message += "이제 데이터를 조회하겠습니다..."
        
        # SQL 실행
        sql_result = self._execute_sql_query(context, refined_query)
        
        # 명확화 정보를 결과에 추가
        if sql_result.get('success'):
            sql_result['clarification_summary'] = session_summary
            sql_result['message'] = response_message + "\n\n" + sql_result.get('message', '')
        
        return sql_result
    
    def _execute_sql_query(self, context: ConversationContext, query: str) -> Dict[str, Any]:
        """SQL 쿼리 실행"""
        context.state = ConversationState.PROCESSING
        context.updated_at = datetime.now()
        
        logger.info(f"SQL 쿼리 실행 시작: {query}")
        
        try:
            # Text2SQL 시스템을 통해 쿼리 처리
            result = self.text2sql_system.query(query, max_rows=100)
            context.sql_result = result
            
            if result['success']:
                context.state = ConversationState.COMPLETED
                
                # 결과 포맷팅
                response_message = "✅ 쿼리가 성공적으로 실행되었습니다!\n\n"
                
                if result.get('sql_query'):
                    response_message += f"**생성된 SQL:**\n```sql\n{result['sql_query']}\n```\n\n"
                
                if result.get('data') is not None and not result['data'].empty:
                    response_message += f"**결과:** {result['row_count']}개 행이 조회되었습니다.\n"
                    response_message += f"**실행 시간:** {result.get('execution_time', 0):.2f}초\n\n"
                    
                    # 데이터 미리보기 (상위 5개 행)
                    preview_data = result['data'].head()
                    response_message += "**데이터 미리보기:**\n"
                    response_message += preview_data.to_string(index=False, max_cols=10)
                    
                    if result['row_count'] > 5:
                        response_message += f"\n... (총 {result['row_count']}개 행 중 5개 표시)"
                else:
                    response_message += "**결과:** 조건에 맞는 데이터가 없습니다."
                
                return {
                    'success': True,
                    'message': response_message,
                    'session_id': context.session_id,
                    'state': 'completed',
                    'data': result.get('data'),
                    'sql_query': result.get('sql_query'),
                    'row_count': result.get('row_count'),
                    'execution_time': result.get('execution_time'),
                    'relevant_tables': result.get('relevant_tables', [])
                }
            
            else:
                context.state = ConversationState.ERROR
                error_message = f"❌ 쿼리 실행 중 오류가 발생했습니다.\n\n**오류 내용:** {result.get('error_message', '알 수 없는 오류')}"
                
                if result.get('sql_query'):
                    error_message += f"\n\n**생성된 SQL:**\n```sql\n{result['sql_query']}\n```"
                
                error_message += "\n\n💡 질문을 다시 명확하게 표현해주시거나, 다른 방식으로 질문해보세요."
                
                return {
                    'success': False,
                    'message': error_message,
                    'session_id': context.session_id,
                    'state': 'error',
                    'error_details': result.get('error_message'),
                    'sql_query': result.get('sql_query')
                }
                
        except Exception as e:
            context.state = ConversationState.ERROR
            error_msg = str(e)
            logger.error(f"SQL 실행 중 예외 발생: {error_msg}")
            
            return {
                'success': False,
                'message': f'❌ 시스템 오류가 발생했습니다: {error_msg}',
                'session_id': context.session_id,
                'state': 'error'
            }
    
    def _handle_non_sql_query(self, context: ConversationContext, analysis_result: IntentAnalysisResult) -> Dict[str, Any]:
        """SQL이 필요하지 않은 쿼리 처리"""
        context.state = ConversationState.COMPLETED
        
        if analysis_result.intent == QueryIntent.GREETING:
            message = "안녕하세요! 저는 데이터베이스 질의 도우미입니다. 🤖\n\n"
            message += "자연어로 데이터 관련 질문을 해주시면, SQL로 변환하여 결과를 제공해드립니다.\n\n"
            message += "**사용 예시:**\n"
            message += "- '모든 사용자 정보를 보여주세요'\n"
            message += "- '부서별 직원 수를 알려주세요'\n"
            message += "- '최근 1개월 매출 현황을 분석해주세요'\n\n"
            message += "또한 '사용 가능한 테이블 목록을 보여주세요'라고 하시면 어떤 데이터가 있는지 확인할 수 있습니다."
            
        elif analysis_result.intent == QueryIntent.HELP_REQUEST:
            message = "📚 **데이터베이스 질의 도우미 사용법**\n\n"
            message += "**1. 기본 사용법:**\n"
            message += "- 자연어로 질문하세요: '사용자 목록을 보여주세요'\n"
            message += "- 구체적일수록 좋습니다: '2024년 1월 가입한 사용자들을 보여주세요'\n\n"
            message += "**2. 지원하는 질의 유형:**\n"
            message += "- 데이터 조회: '~를 보여주세요', '~를 찾아주세요'\n"
            message += "- 데이터 분석: '~를 집계해주세요', '~별 통계를 알려주세요'\n"
            message += "- 스키마 조회: '사용 가능한 테이블을 알려주세요'\n\n"
            message += "**3. 팁:**\n"
            message += "- 모호한 표현은 명확화 질문을 받을 수 있습니다\n"
            message += "- 테이블명을 모르면 '사용 가능한 테이블 목록'을 먼저 확인하세요\n"
            message += "- 복잡한 분석도 자연어로 설명하면 자동으로 SQL이 생성됩니다"
            
        else:
            message = "질문해주셔서 감사합니다. 데이터 관련 질의나 도움이 필요하시면 언제든 말씀해주세요!"
        
        return {
            'success': True,
            'message': message,
            'session_id': context.session_id,
            'state': 'completed',
            'intent': analysis_result.intent.value
        }
    
    def _handle_schema_inquiry(self, context: ConversationContext, analysis_result: IntentAnalysisResult) -> Dict[str, Any]:
        """스키마 문의 처리"""
        context.state = ConversationState.COMPLETED
        
        # 사용 가능한 테이블 목록 제공
        available_tables = self.text2sql_system.get_available_tables()
        
        message = f"📋 **사용 가능한 테이블 정보**\n\n"
        message += f"총 {len(available_tables)}개의 테이블을 사용할 수 있습니다.\n\n"
        
        # 상위 20개 테이블 표시
        for i, (table_name, comment) in enumerate(list(available_tables.items())[:20]):
            message += f"{i+1}. **{table_name}**\n"
            message += f"   - 설명: {comment or '설명 없음'}\n\n"
        
        if len(available_tables) > 20:
            message += f"... 외 {len(available_tables) - 20}개 테이블이 더 있습니다.\n\n"
        
        message += "💡 **사용 방법:**\n"
        message += "- 특정 테이블의 상세 정보: '사용자 테이블의 구조를 알려주세요'\n"
        message += "- 데이터 조회: '사용자 테이블에서 모든 데이터를 보여주세요'\n"
        message += "- 분석 요청: '부서별 직원 수를 계산해주세요'"
        
        return {
            'success': True,
            'message': message,
            'session_id': context.session_id,
            'state': 'completed',
            'intent': analysis_result.intent.value,
            'available_tables': available_tables
        }
    
    def _handle_general_question(self, context: ConversationContext, analysis_result: IntentAnalysisResult) -> Dict[str, Any]:
        """일반적인 질문 처리"""
        context.state = ConversationState.COMPLETED
        
        message = "죄송하지만 데이터베이스와 관련된 질의만 처리할 수 있습니다. 🤖\n\n"
        message += "**처리 가능한 질의 유형:**\n"
        message += "- 데이터 조회 및 검색\n"
        message += "- 데이터 분석 및 통계\n"
        message += "- 테이블 구조 및 스키마 정보\n\n"
        message += "데이터와 관련된 질문이 있으시면 언제든 말씀해주세요!"
        
        return {
            'success': True,
            'message': message,
            'session_id': context.session_id,
            'state': 'completed',
            'intent': analysis_result.intent.value
        }
    
    def get_conversation_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """대화 이력 조회"""
        context = self.active_contexts.get(session_id)
        if not context:
            return None
        
        history = {
            'session_id': session_id,
            'state': context.state.value,
            'original_query': context.original_query,
            'final_query': context.final_query,
            'created_at': context.created_at.isoformat() if context.created_at else None,
            'updated_at': context.updated_at.isoformat() if context.updated_at else None
        }
        
        if context.analysis_result:
            history['analysis'] = {
                'intent': context.analysis_result.intent.value,
                'confidence': context.analysis_result.confidence,
                'complexity': context.analysis_result.complexity.value,
                'needs_sql': context.analysis_result.needs_sql,
                'needs_clarification': context.analysis_result.needs_clarification
            }
        
        if context.clarification_session_id:
            clarification_summary = self.clarification_manager.get_session_summary(
                context.clarification_session_id
            )
            history['clarification'] = clarification_summary
        
        if context.sql_result:
            history['sql_result'] = {
                'success': context.sql_result.get('success'),
                'row_count': context.sql_result.get('row_count'),
                'execution_time': context.sql_result.get('execution_time'),
                'sql_query': context.sql_result.get('sql_query'),
                'error_message': context.sql_result.get('error_message')
            }
        
        return history
    
    def end_conversation(self, session_id: str) -> bool:
        """대화 세션 종료"""
        context = self.active_contexts.get(session_id)
        if not context:
            return False
        
        # 명확화 세션도 함께 정리
        if context.clarification_session_id:
            self.clarification_manager.close_session(context.clarification_session_id)
        
        # 컨텍스트 제거
        del self.active_contexts[session_id]
        
        logger.info(f"대화 세션 종료: {session_id}")
        return True
    
    def get_active_sessions(self) -> List[str]:
        """활성 세션 목록 반환"""
        return list(self.active_contexts.keys())
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 상태 조회"""
        context = self.active_contexts.get(session_id)
        if not context:
            return None
        
        status = {
            'session_id': session_id,
            'state': context.state.value,
            'has_query': context.original_query is not None,
            'has_analysis': context.analysis_result is not None,
            'is_clarifying': context.state == ConversationState.CLARIFYING,
            'is_completed': context.state == ConversationState.COMPLETED,
            'created_at': context.created_at.isoformat(),
            'updated_at': context.updated_at.isoformat()
        }
        
        if context.clarification_session_id:
            next_question = self.clarification_manager.get_next_question(
                context.clarification_session_id
            )
            status['has_pending_question'] = next_question is not None
            
            if next_question:
                status['next_question'] = self.clarification_manager.format_question_for_user(
                    next_question
                )
        
        return status
    
    def reset_session(self, session_id: str) -> bool:
        """세션 초기화 (새로운 질의를 위해)"""
        context = self.active_contexts.get(session_id)
        if not context:
            return False
        
        # 명확화 세션 정리
        if context.clarification_session_id:
            self.clarification_manager.close_session(context.clarification_session_id)
        
        # 컨텍스트 초기화
        context.state = ConversationState.IDLE
        context.original_query = None
        context.analysis_result = None
        context.clarification_session_id = None
        context.final_query = None
        context.sql_result = None
        context.updated_at = datetime.now()
        
        logger.info(f"세션 초기화: {session_id}")
        return True
    
    def provide_suggestion(self, session_id: str) -> Dict[str, Any]:
        """사용자에게 도움말이나 제안 제공"""
        context = self.active_contexts.get(session_id)
        
        suggestions = [
            "💡 **질의 예시들:**",
            "",
            "**기본 조회:**",
            "- '모든 사용자 정보를 보여주세요'",
            "- '2024년에 가입한 고객들을 찾아주세요'",
            "- '활성 상태인 직원 목록을 조회해주세요'",
            "",
            "**분석 질의:**",
            "- '부서별 직원 수를 계산해주세요'",
            "- '월별 매출 통계를 분석해주세요'",
            "- '상위 10개 제품의 판매량을 보여주세요'",
            "",
            "**복잡한 분석:**",
            "- '고객별 평균 주문 금액과 주문 횟수를 함께 조회해주세요'",
            "- '각 부서에서 가장 오래 근무한 직원 정보를 알려주세요'",
            "- '최근 3개월간 매출이 증가한 제품 카테고리를 분석해주세요'",
            "",
            "**스키마 조회:**",
            "- '사용 가능한 테이블 목록을 보여주세요'",
            "- '사용자 테이블의 구조를 알려주세요'"
        ]
        
        message = "\n".join(suggestions)
        
        if context and context.analysis_result:
            message += f"\n\n**이전 분석 결과:** {context.analysis_result.reasoning}"
            if context.analysis_result.suggested_questions:
                message += "\n\n**맞춤 제안:**"
                for q in context.analysis_result.suggested_questions:
                    message += f"\n- {q}"
        
        return {
            'success': True,
            'message': message,
            'session_id': session_id,
            'type': 'suggestion'
        }