"""
향상된 Text2SQL 메인 시스템
의도 분석, 명확화, 대화 관리가 통합된 지능형 시스템
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from llama_index.llms.openai_like import OpenAILike
from text2sql_oracle import Text2SQLOracle
from query_intent_analyzer import QueryIntentAnalyzer
from clarification_manager import ClarificationManager
from conversation_handler import ConversationHandler

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedText2SQLSystem:
    """향상된 Text2SQL 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # LLM 초기화
        self.llm = OpenAILike(
            model=config.get('model_name', 'gpt-3.5-turbo'),
            api_base=config['openai_api_base'],
            api_key=config.get('openai_api_key', 'fake-key'),
            temperature=config.get('temperature', 0.1),
            max_tokens=config.get('max_tokens', 1500)
        )
        
        # Oracle Text2SQL 시스템 초기화
        self.text2sql_system = Text2SQLOracle(
            username=config['oracle_username'],
            password=config['oracle_password'],
            host=config['oracle_host'],
            port=config.get('oracle_port', 1521),
            service_name=config.get('oracle_service_name', 'SMIP_DEV'),
            openai_api_base=config['openai_api_base'],
            openai_api_key=config.get('openai_api_key', 'fake-key'),
            model_name=config.get('model_name', 'gpt-3.5-turbo')
        )
        
        # 사용 가능한 테이블 정보 로드
        self.available_tables = self.text2sql_system.get_available_tables()
        
        # 의도 분석기 초기화
        self.intent_analyzer = QueryIntentAnalyzer(
            llm=self.llm,
            available_tables=self.available_tables
        )
        
        # 명확화 관리자 초기화
        self.clarification_manager = ClarificationManager(
            llm=self.llm,
            available_tables=self.available_tables
        )
        
        # 대화 처리기 초기화
        self.conversation_handler = ConversationHandler(
            text2sql_system=self.text2sql_system,
            intent_analyzer=self.intent_analyzer,
            clarification_manager=self.clarification_manager
        )
        
        logger.info("향상된 Text2SQL 시스템 초기화 완료")
    
    def start_session(self, user_id: str = None) -> str:
        """
        새로운 대화 세션 시작
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            str: 세션 ID
        """
        session_id = self.conversation_handler.start_conversation(user_id)
        logger.info(f"새 세션 시작: {session_id}")
        return session_id
    
    def process_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        사용자 쿼리 처리
        
        Args:
            session_id: 세션 ID
            query: 자연어 쿼리
            
        Returns:
            Dict: 처리 결과
        """
        logger.info(f"쿼리 처리 시작 [{session_id}]: {query}")
        
        result = self.conversation_handler.process_user_message(session_id, query)
        
        # 결과 로깅
        if result.get('success'):
            logger.info(f"쿼리 처리 성공 [{session_id}]: {result.get('state')}")
        else:
            logger.warning(f"쿼리 처리 실패 [{session_id}]: {result.get('message')}")
        
        return result
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션 상태 조회
        
        Args:
            session_id: 세션 ID
            
        Returns:
            Dict: 세션 상태 정보
        """
        return self.conversation_handler.get_session_status(session_id)
    
    def get_conversation_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        대화 이력 조회
        
        Args:
            session_id: 세션 ID
            
        Returns:
            Dict: 대화 이력
        """
        return self.conversation_handler.get_conversation_history(session_id)
    
    def end_session(self, session_id: str) -> bool:
        """
        세션 종료
        
        Args:
            session_id: 세션 ID
            
        Returns:
            bool: 종료 성공 여부
        """
        success = self.conversation_handler.end_conversation(session_id)
        if success:
            logger.info(f"세션 종료: {session_id}")
        return success
    
    def reset_session(self, session_id: str) -> bool:
        """
        세션 초기화
        
        Args:
            session_id: 세션 ID
            
        Returns:
            bool: 초기화 성공 여부
        """
        return self.conversation_handler.reset_session(session_id)
    
    def get_suggestions(self, session_id: str) -> Dict[str, Any]:
        """
        사용자에게 제안 제공
        
        Args:
            session_id: 세션 ID
            
        Returns:
            Dict: 제안 정보
        """
        return self.conversation_handler.provide_suggestion(session_id)
    
    def get_available_tables(self) -> Dict[str, str]:
        """사용 가능한 테이블 목록 반환"""
        return self.available_tables
    
    def get_table_schema(self, table_name: str, schema_name: str = None) -> Dict[str, Any]:
        """
        테이블 스키마 정보 조회
        
        Args:
            table_name: 테이블명
            schema_name: 스키마명
            
        Returns:
            Dict: 스키마 정보
        """
        return self.text2sql_system.get_table_schema(table_name, schema_name)
    
    def test_connection(self) -> bool:
        """데이터베이스 연결 테스트"""
        return self.text2sql_system.test_connection()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 정보"""
        active_sessions = self.conversation_handler.get_active_sessions()
        
        return {
            'active_sessions': len(active_sessions),
            'session_ids': active_sessions,
            'available_tables_count': len(self.available_tables),
            'system_status': 'running',
            'database_connected': self.test_connection()
        }
    
    def close(self):
        """시스템 종료"""
        # 모든 활성 세션 종료
        active_sessions = self.conversation_handler.get_active_sessions()
        for session_id in active_sessions:
            self.end_session(session_id)
        
        # Text2SQL 시스템 종료
        self.text2sql_system.close()
        
        logger.info("향상된 Text2SQL 시스템 종료")

class InteractiveConsole:
    """대화형 콘솔 인터페이스"""
    
    def __init__(self, system: EnhancedText2SQLSystem):
        self.system = system
        self.current_session_id = None
    
    def start(self):
        """콘솔 시작"""
        print("🤖 향상된 Text2SQL 시스템에 오신 것을 환영합니다!")
        print("='=' * 30)
        
        # 새 세션 시작
        self.current_session_id = self.system.start_session("console_user")
        print(f"새 세션이 시작되었습니다: {self.current_session_id}")
        
        # 도움말 표시
        self._show_help()
        
        # 메인 루프
        try:
            while True:
                user_input = input("\n💬 질문을 입력하세요 (도움말: /help, 종료: /quit): ").strip()
                
                if not user_input:
                    continue
                
                # 명령어 처리
                if user_input.startswith('/'):
                    if not self._handle_command(user_input):
                        break
                    continue
                
                # 일반 쿼리 처리
                self._process_user_query(user_input)
                
        except KeyboardInterrupt:
            print("\n\n👋 이용해주셔서 감사합니다!")
        finally:
            if self.current_session_id:
                self.system.end_session(self.current_session_id)
    
    def _handle_command(self, command: str) -> bool:
        """명령어 처리"""
        cmd = command.lower().strip()
        
        if cmd == '/quit' or cmd == '/exit':
            print("👋 이용해주셔서 감사합니다!")
            return False
        
        elif cmd == '/help':
            self._show_help()
        
        elif cmd == '/status':
            self._show_status()
        
        elif cmd == '/history':
            self._show_history()
        
        elif cmd == '/tables':
            self._show_tables()
        
        elif cmd == '/reset':
            self._reset_session()
        
        elif cmd == '/suggest':
            self._show_suggestions()
        
        elif cmd == '/stats':
            self._show_system_stats()
        
        else:
            print(f"❌ 알 수 없는 명령어입니다: {command}")
            print("사용 가능한 명령어를 보려면 /help를 입력하세요.")
        
        return True
    
    def _process_user_query(self, query: str):
        """사용자 쿼리 처리"""
        print(f"\n🔄 처리 중: {query}")
        print("-" * 50)
        
        result = self.system.process_query(self.current_session_id, query)
        
        # 결과 출력
        print(f"\n📋 결과:")
        print(result.get('message', ''))
        
        # 추가 정보 출력
        if result.get('state') == 'clarifying' and result.get('question'):
            print(f"\n{result['question']}")
        
        elif result.get('success') and result.get('data') is not None:
            # 데이터가 있는 경우 추가 통계 표시
            if hasattr(result['data'], 'shape'):
                print(f"\n📊 데이터 정보: {result['data'].shape[0]}행 × {result['data'].shape[1]}열")
        
        print("-" * 50)
    
    def _show_help(self):
        """도움말 표시"""
        help_text = """
📚 사용법 및 명령어:

🔹 일반 질의:
   자연어로 질문하세요. 예:
   - "모든 사용자 정보를 보여주세요"
   - "부서별 직원 수를 계산해주세요"
   - "최근 1개월 매출 현황을 분석해주세요"

🔹 명령어:
   /help     - 이 도움말 표시
   /status   - 현재 세션 상태 확인
   /history  - 대화 이력 조회
   /tables   - 사용 가능한 테이블 목록
   /reset    - 세션 초기화 (새로운 질의 시작)
   /suggest  - 질의 예시 및 제안
   /stats    - 시스템 통계
   /quit     - 프로그램 종료

💡 팁: 모호한 질문은 명확화 과정을 거칩니다!
        """
        print(help_text)
    
    def _show_status(self):
        """세션 상태 표시"""
        status = self.system.get_session_status(self.current_session_id)
        if status:
            print(f"\n📊 세션 상태:")
            print(f"   ID: {status['session_id']}")
            print(f"   상태: {status['state']}")
            print(f"   쿼리 있음: {status['has_query']}")
            print(f"   분석 완료: {status['has_analysis']}")
            print(f"   명확화 중: {status['is_clarifying']}")
            print(f"   완료됨: {status['is_completed']}")
            
            if status.get('has_pending_question'):
                print(f"\n❓ 대기 중인 질문:")
                print(status.get('next_question', ''))
        else:
            print("❌ 세션 상태를 찾을 수 없습니다.")
    
    def _show_history(self):
        """대화 이력 표시"""
        history = self.system.get_conversation_history(self.current_session_id)
        if history:
            print(f"\n📚 대화 이력:")
            print(f"   원본 쿼리: {history.get('original_query', 'N/A')}")
            print(f"   최종 쿼리: {history.get('final_query', 'N/A')}")
            print(f"   상태: {history.get('state', 'N/A')}")
            
            if history.get('analysis'):
                analysis = history['analysis']
                print(f"\n🔍 분석 결과:")
                print(f"   의도: {analysis.get('intent', 'N/A')}")
                print(f"   신뢰도: {analysis.get('confidence', 0):.2f}")
                print(f"   복잡도: {analysis.get('complexity', 'N/A')}")
            
            if history.get('sql_result'):
                sql_result = history['sql_result']
                print(f"\n💾 SQL 실행 결과:")
                print(f"   성공: {sql_result.get('success', False)}")
                if sql_result.get('success'):
                    print(f"   행 수: {sql_result.get('row_count', 0)}")
                    print(f"   실행 시간: {sql_result.get('execution_time', 0):.2f}초")
                else:
                    print(f"   오류: {sql_result.get('error_message', 'N/A')}")
        else:
            print("❌ 대화 이력을 찾을 수 없습니다.")
    
    def _show_tables(self):
        """사용 가능한 테이블 목록 표시"""
        tables = self.system.get_available_tables()
        print(f"\n🗃️ 사용 가능한 테이블 ({len(tables)}개):")
        
        for i, (table_name, comment) in enumerate(list(tables.items())[:20]):
            print(f"   {i+1:2d}. {table_name:<30} | {comment or '설명 없음'}")
        
        if len(tables) > 20:
            print(f"   ... 외 {len(tables) - 20}개 테이블")
    
    def _reset_session(self):
        """세션 초기화"""
        success = self.system.reset_session(self.current_session_id)
        if success:
            print("✅ 세션이 초기화되었습니다. 새로운 질의를 시작할 수 있습니다.")
        else:
            print("❌ 세션 초기화에 실패했습니다.")
    
    def _show_suggestions(self):
        """제안 표시"""
        suggestions = self.system.get_suggestions(self.current_session_id)
        print(f"\n{suggestions.get('message', '')}")
    
    def _show_system_stats(self):
        """시스템 통계 표시"""
        stats = self.system.get_system_stats()
        print(f"\n📈 시스템 통계:")
        print(f"   활성 세션: {stats['active_sessions']}개")
        print(f"   사용 가능한 테이블: {stats['available_tables_count']}개")
        print(f"   시스템 상태: {stats['system_status']}")
        print(f"   DB 연결: {'✅ 연결됨' if stats['database_connected'] else '❌ 연결 실패'}")

def create_system_from_config(config_path: str = None, config_dict: Dict[str, Any] = None) -> EnhancedText2SQLSystem:
    """
    설정 파일 또는 딕셔너리로부터 시스템 생성
    
    Args:
        config_path: 설정 파일 경로 (JSON)
        config_dict: 설정 딕셔너리
        
    Returns:
        EnhancedText2SQLSystem: 초기화된 시스템
    """
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    elif config_dict:
        config = config_dict
    else:
        raise ValueError("config_path 또는 config_dict 중 하나는 제공되어야 합니다.")
    
    return EnhancedText2SQLSystem(config)

def main():
    """메인 함수 - 대화형 콘솔 실행"""
    
    # 예시 설정 (실제 사용시 환경변수나 설정 파일 사용 권장)
    config = {
        # Oracle 데이터베이스 설정
        'oracle_username': 'your_username',
        'oracle_password': 'your_password',
        'oracle_host': 'your_oracle_host',
        'oracle_port': 1521,
        'oracle_service_name': 'SMIP_DEV',
        
        # LLM 설정
        'openai_api_base': 'http://localhost:8000/v1',  # 로컬 API 서버
        'openai_api_key': 'your-api-key',
        'model_name': 'gpt-3.5-turbo',
        'temperature': 0.1,
        'max_tokens': 1500
    }
    
    try:
        print("🚀 향상된 Text2SQL 시스템 초기화 중...")
        
        # 시스템 초기화
        system = create_system_from_config(config_dict=config)
        
        # 연결 테스트
        if not system.test_connection():
            print("❌ 데이터베이스 연결에 실패했습니다.")
            print("설정을 확인하고 다시 시도해주세요.")
            return
        
        print("✅ 시스템 초기화 완료!")
        
        # 대화형 콘솔 시작
        console = InteractiveConsole(system)
        console.start()
        
    except Exception as e:
        logger.error(f"시스템 초기화 실패: {e}")
        print(f"❌ 시스템 초기화에 실패했습니다: {e}")
        print("\n설정을 확인해주세요:")
        print("1. Oracle 데이터베이스 연결 정보")
        print("2. OpenAI-like API 서버 설정")
        print("3. 필요한 Python 패키지 설치 (oracledb, llama-index 등)")
    
    finally:
        # 시스템 정리
        if 'system' in locals():
            system.close()

class WebAPIInterface:
    """
    웹 API 인터페이스 (FastAPI 등과 함께 사용)
    
    사용 예시:
    from fastapi import FastAPI
    from enhanced_text2sql_main import WebAPIInterface
    
    app = FastAPI()
    text2sql_api = WebAPIInterface(config)
    
    @app.post("/query")
    async def process_query(request: dict):
        return text2sql_api.process_query(
            request.get("session_id"), 
            request.get("query")
        )
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.system = EnhancedText2SQLSystem(config)
        logger.info("Web API 인터페이스 초기화 완료")
    
    def create_session(self, user_id: str = None) -> Dict[str, Any]:
        """새 세션 생성"""
        session_id = self.system.start_session(user_id)
        return {
            'success': True,
            'session_id': session_id,
            'message': '세션이 생성되었습니다.'
        }
    
    def process_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """쿼리 처리"""
        if not session_id:
            return {
                'success': False,
                'error': 'session_id가 필요합니다.'
            }
        
        if not query or not query.strip():
            return {
                'success': False,
                'error': '쿼리가 비어있습니다.'
            }
        
        return self.system.process_query(session_id, query.strip())
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """세션 정보 조회"""
        status = self.system.get_session_status(session_id)
        if not status:
            return {
                'success': False,
                'error': '세션을 찾을 수 없습니다.'
            }
        
        history = self.system.get_conversation_history(session_id)
        
        return {
            'success': True,
            'status': status,
            'history': history
        }
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """세션 종료"""
        success = self.system.end_session(session_id)
        return {
            'success': success,
            'message': '세션이 종료되었습니다.' if success else '세션을 찾을 수 없습니다.'
        }
    
    def get_available_tables(self) -> Dict[str, Any]:
        """사용 가능한 테이블 목록"""
        tables = self.system.get_available_tables()
        return {
            'success': True,
            'tables': tables,
            'count': len(tables)
        }
    
    def get_table_schema(self, table_name: str, schema_name: str = None) -> Dict[str, Any]:
        """테이블 스키마 정보"""
        try:
            schema = self.system.get_table_schema(table_name, schema_name)
            return {
                'success': True,
                'schema': schema
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        stats = self.system.get_system_stats()
        return {
            'success': True,
            'health': 'healthy' if stats['database_connected'] else 'unhealthy',
            'stats': stats
        }
    
    def close(self):
        """API 인터페이스 종료"""
        self.system.close()

if __name__ == "__main__":
    main()