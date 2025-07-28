"""
통합 Text2SQL 시스템
정형/비정형 데이터, 시각화, 대화형 인터페이스를 모두 통합한 완전한 시스템
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# LLM
from llama_index.llms.openai_like import OpenAILike

# 내부 모듈
from enhanced_query_intent_analyzer import (
    EnhancedQueryIntentAnalyzer, 
    EnhancedIntentAnalysisResult,
    QueryIntent,
    DataSourceType,
    VisualizationType
)
from clarification_manager import ClarificationManager
from data_source_manager import HybridDataSourceManager, DataSourceRouter
from visualization_generator import SmartChartGenerator, TableGenerator, DashboardGenerator
from conversation_handler import ConversationState

logger = logging.getLogger(__name__)

class ProcessingStage(Enum):
    """처리 단계"""
    INTENT_ANALYSIS = "intent_analysis"
    CLARIFICATION = "clarification"
    DATA_RETRIEVAL = "data_retrieval"
    VISUALIZATION = "visualization"
    RESPONSE_GENERATION = "response_generation"

@dataclass
class IntegratedQueryResult:
    """통합 쿼리 결과"""
    success: bool
    session_id: str
    query: str
    
    # 분석 결과
    intent_analysis: Optional[EnhancedIntentAnalysisResult] = None
    
    # 데이터 결과
    structured_data: Optional[Any] = None
    unstructured_data: Optional[List[Dict[str, Any]]] = None
    
    # 시각화 결과
    visualization: Optional[Any] = None
    
    # 메타데이터
    processing_stages: Dict[str, Any] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    
    # 응답 정보
    response_message: str = ""
    needs_clarification: bool = False
    clarification_questions: List[str] = None
    
    # 추가 정보 요청 가능 여부
    can_request_more_info: bool = True
    suggested_followup_questions: List[str] = None

class IntegratedText2SQLSystem:
    """통합 Text2SQL 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        시스템 초기화
        
        Args:
            config: 통합 설정
        """
        self.config = config
        
        # LLM 초기화
        self.llm = OpenAILike(
            model=config.get('model_name', 'gpt-3.5-turbo'),
            api_base=config['openai_api_base'],
            api_key=config.get('openai_api_key', 'fake-key'),
            temperature=config.get('temperature', 0.1),
            max_tokens=config.get('max_tokens', 2000)
        )
        
        # 하이브리드 데이터 소스 관리자 초기화
        self.data_manager = HybridDataSourceManager(
            oracle_config=config['oracle'],
            chromadb_config=config['chromadb'],
            llm=self.llm
        )
        
        # 데이터 소스 라우터
        self.data_router = DataSourceRouter(self.data_manager)
        
        # 의도 분석기 초기화
        oracle_tables = self.data_manager.get_oracle_tables()
        chromadb_collections = self.data_manager.get_chromadb_collections()
        
        self.intent_analyzer = EnhancedQueryIntentAnalyzer(
            llm=self.llm,
            oracle_tables=oracle_tables,
            chromadb_collections=chromadb_collections
        )
        
        # 명확화 관리자 초기화
        self.clarification_manager = ClarificationManager(
            llm=self.llm,
            available_tables=oracle_tables
        )
        
        # 시각화 생성기들 초기화
        self.chart_generator = SmartChartGenerator()
        self.table_generator = TableGenerator()
        self.dashboard_generator = DashboardGenerator(
            self.chart_generator, 
            self.table_generator
        )
        
        # 활성 세션 관리
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("통합 Text2SQL 시스템 초기화 완료")
    
    async def process_query(self, 
                          query: str, 
                          session_id: str = None,
                          user_context: Dict[str, Any] = None) -> IntegratedQueryResult:
        """
        통합 쿼리 처리
        
        Args:
            query: 자연어 쿼리
            session_id: 세션 ID
            user_context: 사용자 컨텍스트
            
        Returns:
            IntegratedQueryResult: 통합 처리 결과
        """
        start_time = datetime.now()
        
        # 세션 ID 생성 또는 사용
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # 세션 초기화
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                'created_at': datetime.now(),
                'query_history': [],
                'context': user_context or {}
            }
        
        processing_stages = {}
        
        try:
            # 1단계: 의도 분석
            logger.info(f"[{session_id}] 의도 분석 시작: {query}")
            processing_stages[ProcessingStage.INTENT_ANALYSIS.value] = {
                'start_time': datetime.now().isoformat()
            }
            
            intent_result = await self._analyze_intent(query, session_id)
            processing_stages[ProcessingStage.INTENT_ANALYSIS.value]['completed'] = True
            processing_stages[ProcessingStage.INTENT_ANALYSIS.value]['result'] = {
                'intent': intent_result.intent.value,
                'confidence': intent_result.confidence,
                'data_source_type': intent_result.data_source_type.value,
                'visualization_type': intent_result.visualization_type.value
            }
            
            # 2단계: 명확화 확인
            if intent_result.needs_clarification:
                return await self._handle_clarification_needed(
                    query, session_id, intent_result, processing_stages, start_time
                )
            
            # 3단계: 데이터 조회
            logger.info(f"[{session_id}] 데이터 조회 시작")
            processing_stages[ProcessingStage.DATA_RETRIEVAL.value] = {
                'start_time': datetime.now().isoformat()
            }
            
            data_result = await self._retrieve_data(query, intent_result, session_id)
            processing_stages[ProcessingStage.DATA_RETRIEVAL.value]['completed'] = True
            
            # 4단계: 시각화 생성 (필요한 경우)
            visualization_result = None
            if intent_result.needs_visualization:
                logger.info(f"[{session_id}] 시각화 생성 시작")
                processing_stages[ProcessingStage.VISUALIZATION.value] = {
                    'start_time': datetime.now().isoformat()
                }
                
                visualization_result = await self._generate_visualization(
                    data_result, intent_result, session_id
                )
                processing_stages[ProcessingStage.VISUALIZATION.value]['completed'] = True
            
            # 5단계: 응답 생성
            logger.info(f"[{session_id}] 응답 생성 시작")
            processing_stages[ProcessingStage.RESPONSE_GENERATION.value] = {
                'start_time': datetime.now().isoformat()
            }
            
            response_message, followup_questions = await self._generate_response(
                query, intent_result, data_result, visualization_result, session_id
            )
            processing_stages[ProcessingStage.RESPONSE_GENERATION.value]['completed'] = True
            
            # 세션 히스토리 업데이트
            self.active_sessions[session_id]['query_history'].append({
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'intent': intent_result.intent.value,
                'success': True
            })
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return IntegratedQueryResult(
                success=True,
                session_id=session_id,
                query=query,
                intent_analysis=intent_result,
                structured_data=data_result.get('structured'),
                unstructured_data=data_result.get('unstructured'),
                visualization=visualization_result,
                processing_stages=processing_stages,
                execution_time=execution_time,
                response_message=response_message,
                suggested_followup_questions=followup_questions
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"[{session_id}] 쿼리 처리 실패: {error_msg}")
            
            return IntegratedQueryResult(
                success=False,
                session_id=session_id,
                query=query,
                processing_stages=processing_stages,
                execution_time=execution_time,
                error_message=error_msg,
                response_message=f"처리 중 오류가 발생했습니다: {error_msg}"
            )
    
    async def _analyze_intent(self, query: str, session_id: str) -> EnhancedIntentAnalysisResult:
        """의도 분석"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.intent_analyzer.analyze_intent, 
            query
        )
    
    async def _handle_clarification_needed(self, 
                                         query: str, 
                                         session_id: str,
                                         intent_result: EnhancedIntentAnalysisResult,
                                         processing_stages: Dict[str, Any],
                                         start_time: datetime) -> IntegratedQueryResult:
        """명확화 필요 상황 처리"""
        processing_stages[ProcessingStage.CLARIFICATION.value] = {
            'start_time': datetime.now().isoformat(),
            'questions_generated': len(intent_result.suggested_questions)
        }
        
        # 명확화 세션 시작
        clarification_session_id = f"clarif_{session_id}_{uuid.uuid4().hex[:6]}"
        clarification_session = self.clarification_manager.start_clarification(
            clarification_session_id, query, intent_result
        )
        
        # 첫 번째 질문 가져오기
        first_question = self.clarification_manager.get_next_question(clarification_session_id)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 세션에 명확화 정보 저장
        self.active_sessions[session_id]['clarification_session_id'] = clarification_session_id
        
        return IntegratedQueryResult(
            success=True,
            session_id=session_id,
            query=query,
            intent_analysis=intent_result,
            processing_stages=processing_stages,
            execution_time=execution_time,
            response_message="질문을 더 정확히 이해하기 위해 몇 가지 질문을 드리겠습니다.",
            needs_clarification=True,
            clarification_questions=[self.clarification_manager.format_question_for_user(first_question)] if first_question else []
        )
    
    async def _retrieve_data(self, 
                           query: str, 
                           intent_result: EnhancedIntentAnalysisResult,
                           session_id: str) -> Dict[str, Any]:
        """데이터 조회"""
        result = await self.data_router.route_and_execute(intent_result, query)
        
        if isinstance(result, tuple):
            # 하이브리드 결과
            structured_result, unstructured_result = result
            return {
                'structured': structured_result,
                'unstructured': unstructured_result,
                'hybrid': True
            }
        else:
            # 단일 소스 결과
            if intent_result.data_source_type == DataSourceType.ORACLE_ONLY:
                return {'structured': result, 'hybrid': False}
            else:
                return {'unstructured': result, 'hybrid': False}
    
    async def _generate_visualization(self, 
                                    data_result: Dict[str, Any],
                                    intent_result: EnhancedIntentAnalysisResult,
                                    session_id: str) -> Any:
        """시각화 생성"""
        try:
            # 정형 데이터가 있는 경우 우선 사용
            if data_result.get('structured') and data_result['structured'].success:
                data_df = data_result['structured'].structured_data
                if data_df is not None and not data_df.empty:
                    viz_config = intent_result.visualization_config.copy()
                    viz_config['title'] = f"{intent_result.intent.value} 결과"
                    
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        self.chart_generator.generate_chart,
                        data_df,
                        viz_config
                    )
            
            # 비정형 데이터만 있는 경우 요약 테이블 생성
            elif data_result.get('unstructured') and data_result['unstructured'].success:
                unstructured_data = data_result['unstructured'].unstructured_data
                if unstructured_data:
                    # 문서 검색 결과를 테이블로 변환
                    import pandas as pd
                    df = pd.DataFrame([
                        {
                            '순위': i+1,
                            '내용 미리보기': doc.get('content', '')[:100] + '...',
                            '유사도 점수': f"{doc.get('score', 0):.3f}",
                            '문서 ID': doc.get('doc_id', f'doc_{i}')
                        }
                        for i, doc in enumerate(unstructured_data[:10])
                    ])
                    
                    table_config = {
                        'title': '문서 검색 결과',
                        'max_rows': 10,
                        'column_mapping': {
                            '순위': 'Rank',
                            '내용 미리보기': 'Content Preview',
                            '유사도 점수': 'Similarity Score',
                            '문서 ID': 'Document ID'
                        }
                    }
                    
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        self.table_generator.generate_table,
                        df,
                        table_config
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"시각화 생성 실패: {e}")
            return None
    
    async def _generate_response(self, 
                               query: str,
                               intent_result: EnhancedIntentAnalysisResult,
                               data_result: Dict[str, Any],
                               visualization_result: Any,
                               session_id: str) -> Tuple[str, List[str]]:
        """응답 메시지 생성"""
        response_parts = []
        followup_questions = []
        
        # 기본 성공 메시지
        if intent_result.intent == QueryIntent.GREETING:
            response_parts.append("안녕하세요! 저는 통합 데이터 분석 도우미입니다. 🤖")
            response_parts.append("정형 데이터(Oracle DB)와 비정형 데이터(문서)를 모두 활용하여 질문에 답변드립니다.")
            followup_questions = [
                "사용 가능한 테이블 목록을 보여주세요",
                "문서 컬렉션에는 어떤 것들이 있나요?",
                "차트를 그려서 데이터를 분석해주세요"
            ]
        
        elif intent_result.intent == QueryIntent.SCHEMA_INQUIRY:
            oracle_tables = self.data_manager.get_oracle_tables()
            chromadb_collections = self.data_manager.get_chromadb_collections()
            
            response_parts.append("📊 **사용 가능한 데이터 소스**")
            response_parts.append(f"**정형 데이터 (Oracle):** {len(oracle_tables)}개 테이블")
            for i, (table, desc) in enumerate(list(oracle_tables.items())[:10]):
                response_parts.append(f"  {i+1}. {table}: {desc or '설명 없음'}")
            
            response_parts.append(f"\n**비정형 데이터 (ChromaDB):** {len(chromadb_collections)}개 컬렉션")
            for i, collection in enumerate(chromadb_collections):
                response_parts.append(f"  {i+1}. {collection}")
        
        else:
            # 데이터 조회 결과 처리
            success_count = 0
            
            if data_result.get('structured') and data_result['structured'].success:
                structured_data = data_result['structured'].structured_data
                response_parts.append("✅ **정형 데이터 조회 성공**")
                if structured_data is not None:
                    response_parts.append(f"📊 {len(structured_data)}개 행의 데이터를 조회했습니다.")
                success_count += 1
            
            if data_result.get('unstructured') and data_result['unstructured'].success:
                unstructured_data = data_result['unstructured'].unstructured_data
                response_parts.append("✅ **비정형 데이터 검색 성공**")
                if unstructured_data:
                    response_parts.append(f"📄 {len(unstructured_data)}개의 관련 문서를 찾았습니다.")
                success_count += 1
            
            if success_count == 0:
                response_parts.append("❌ 데이터 조회에 실패했습니다.")
                if data_result.get('structured'):
                    response_parts.append(f"Oracle 오류: {data_result['structured'].error_message}")
                if data_result.get('unstructured'):
                    response_parts.append(f"ChromaDB 오류: {data_result['unstructured'].error_message}")
            
            # 시각화 결과
            if visualization_result and visualization_result.success:
                response_parts.append("📈 **시각화가 생성되었습니다.**")
            
            # 후속 질문 제안
            if success_count > 0:
                followup_questions = [
                    "이 데이터를 다른 형태의 차트로 보여주세요",
                    "더 자세한 분석을 해주세요",
                    "관련된 다른 데이터도 함께 보여주세요",
                    "이 결과를 표로 정리해주세요"
                ]
                
                # 하이브리드 데이터인 경우 추가 질문
                if data_result.get('hybrid'):
                    followup_questions.extend([
                        "원문 내용을 더 자세히 보여주세요",
                        "문서와 데이터를 연결하여 분석해주세요"
                    ])
        
        response_message = "\n".join(response_parts)
        
        return response_message, followup_questions
    
    async def handle_clarification_response(self, 
                                          session_id: str, 
                                          response: str) -> IntegratedQueryResult:
        """명확화 응답 처리"""
        if session_id not in self.active_sessions:
            return IntegratedQueryResult(
                success=False,
                session_id=session_id,
                query=response,
                error_message="세션을 찾을 수 없습니다."
            )
        
        session = self.active_sessions[session_id]
        clarification_session_id = session.get('clarification_session_id')
        
        if not clarification_session_id:
            return IntegratedQueryResult(
                success=False,
                session_id=session_id,
                query=response,
                error_message="명확화 세션이 활성화되어 있지 않습니다."
            )
        
        # 명확화 응답 처리
        result = self.clarification_manager.process_user_input(
            clarification_session_id, response
        )
        
        if not result['success']:
            return IntegratedQueryResult(
                success=False,
                session_id=session_id,
                query=response,
                error_message=result['message'],
                needs_clarification=True,
                clarification_questions=[
                    self.clarification_manager.format_question_for_user(result['next_question'])
                ] if result['next_question'] else []
            )
        
        # 명확화 완료 확인
        if result['is_complete']:
            # 정제된 쿼리로 다시 처리
            refined_query = self.clarification_manager.generate_refined_query(
                clarification_session_id
            )
            
            if refined_query:
                # 명확화 세션 정리
                del session['clarification_session_id']
                self.clarification_manager.close_session(clarification_session_id)
                
                # 정제된 쿼리로 재처리
                return await self.process_query(refined_query, session_id)
        
        # 다음 질문이 있는 경우
        next_question = result.get('next_question')
        if next_question:
            return IntegratedQueryResult(
                success=True,
                session_id=session_id,
                query=response,
                response_message="답변이 저장되었습니다.",
                needs_clarification=True,
                clarification_questions=[
                    self.clarification_manager.format_question_for_user(next_question)
                ]
            )
        
        return IntegratedQueryResult(
            success=True,
            session_id=session_id,
            query=response,
            response_message="명확화가 완료되었습니다."
        )
    
    async def request_additional_info(self, 
                                    session_id: str, 
                                    request_type: str,
                                    parameters: Dict[str, Any] = None) -> IntegratedQueryResult:
        """추가 정보 요청 처리"""
        if session_id not in self.active_sessions:
            return IntegratedQueryResult(
                success=False,
                session_id=session_id,
                query="",
                error_message="세션을 찾을 수 없습니다."
            )
        
        session = self.active_sessions[session_id]
        last_query_info = session['query_history'][-1] if session['query_history'] else None
        
        if not last_query_info:
            return IntegratedQueryResult(
                success=False,
                session_id=session_id,
                query="",
                error_message="이전 쿼리 정보를 찾을 수 없습니다."
            )
        
        # 요청 유형에 따른 처리
        if request_type == "more_details":
            # 더 자세한 정보 요청
            additional_query = f"{last_query_info['query']} - 더 자세한 분석과 원문 내용 포함"
            return await self.process_query(additional_query, session_id)
        
        elif request_type == "different_chart":
            # 다른 형태의 차트 요청
            chart_type = parameters.get('chart_type', 'bar')
            chart_query = f"{last_query_info['query']} - {chart_type} 차트로 표시"
            return await self.process_query(chart_query, session_id)
        
        elif request_type == "export_data":
            # 데이터 내보내기 요청
            # 실제 구현에서는 파일 생성 로직 추가
            return IntegratedQueryResult(
                success=True,
                session_id=session_id,
                query="export_request",
                response_message="데이터 내보내기 기능은 준비 중입니다."
            )
        
        else:
            return IntegratedQueryResult(
                success=False,
                session_id=session_id,
                query="",
                error_message=f"지원하지 않는 요청 유형입니다: {request_type}"
            )
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 정보 조회"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            'session_id': session_id,
            'created_at': session['created_at'].isoformat(),
            'query_count': len(session['query_history']),
            'recent_queries': session['query_history'][-5:],  # 최근 5개
            'has_clarification': 'clarification_session_id' in session,
            'context': session.get('context', {})
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        system_info = self.data_manager.get_system_info()
        
        return {
            'active_sessions': len(self.active_sessions),
            'data_sources': system_info,
            'capabilities': {
                'structured_data': system_info['oracle']['connected'],
                'unstructured_data': system_info['chromadb']['connected'],
                'visualization': True,
                'clarification': True,
                'hybrid_queries': system_info['hybrid_capabilities']['hybrid_queries']
            },
            'system_health': 'healthy' if all([
                system_info['oracle']['connected'],
                system_info['chromadb']['connected']
            ]) else 'degraded'
        }
    
    def close_session(self, session_id: str) -> bool:
        """세션 종료"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # 명확화 세션 정리
            if 'clarification_session_id' in session:
                self.clarification_manager.close_session(session['clarification_session_id'])
            
            del self.active_sessions[session_id]
            logger.info(f"세션 종료: {session_id}")
            return True
        return False
    
    def close(self):
        """시스템 종료"""
        # 모든 활성 세션 종료
        for session_id in list(self.active_sessions.keys()):
            self.close_session(session_id)
        
        # 데이터 소스 연결 종료
        self.data_manager.close()
        
        logger.info("통합 Text2SQL 시스템 종료 완료")

# 사용 예시 및 테스트용 함수들
async def demo_integrated_system():
    """통합 시스템 데모"""
    config = {
        'openai_api_base': 'http://localhost:8000/v1',
        'openai_api_key': 'your-api-key',
        'model_name': 'gpt-3.5-turbo',
        'oracle': {
            'username': 'your_username',
            'password': 'your_password',
            'host': 'localhost',
            'port': 1521,
            'service_name': 'SMIP_DEV',
            'api_base': 'http://localhost:8000/v1',
            'api_key': 'your-api-key',
            'model_name': 'gpt-3.5-turbo'
        },
        'chromadb': {
            'host': 'localhost',
            'port': 8000,
            'embedding_model': 'text-embedding-ada-002'
        }
    }
    
    try:
        system = IntegratedText2SQLSystem(config)
        
        # 테스트 쿼리들
        test_queries = [
            "안녕하세요!",
            "사용 가능한 데이터를 알려주세요",
            "사용자 테이블의 데이터를 막대 차트로 보여주세요",
            "고객 관련 문서를 검색해주세요",
            "매출 데이터와 관련 문서를 함께 분석해주세요"
        ]
        
        session_id = None
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"쿼리: {query}")
            print(f"{'='*50}")
            
            result = await system.process_query(query, session_id)
            session_id = result.session_id  # 세션 유지
            
            print(f"성공: {result.success}")
            print(f"실행 시간: {result.execution_time:.2f}초")
            print(f"응답: {result.response_message}")
            
            if result.needs_clarification:
                print("명확화 필요:")
                for q in result.clarification_questions:
                    print(f"  - {q}")
            
            if result.suggested_followup_questions:
                print("후속 질문 제안:")
                for q in result.suggested_followup_questions:
                    print(f"  - {q}")
        
        # 시스템 상태 확인
        print(f"\n{'='*50}")
        print("시스템 상태:")
        print(f"{'='*50}")
        status = system.get_system_status()
        print(f"활성 세션: {status['active_sessions']}")
        print(f"시스템 상태: {status['system_health']}")
        print(f"기능: {status['capabilities']}")
        
        # 세션 종료
        if session_id:
            system.close_session(session_id)
        
        system.close()
        
    except Exception as e:
        print(f"데모 실행 중 오류: {e}")

if __name__ == "__main__":
    asyncio.run(demo_integrated_system())