# main.py
import logging
from config import ORACLE_CONFIG, LLM_CONFIG
from oracle_connector import OracleConnectionManager, QueryResult
from intent_classifier import IntentClassifier
from query_analyzer import QueryAnalyzer
from text2sql_engine import Text2SQLConverter

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Chatbot:
    """대화형 Text-to-SQL 챗봇의 메인 컨트롤러"""

    def __init__(self):
        try:
            logger.info("시스템 초기화를 시작합니다...")
            # 1. DB 연결 관리자 초기화
            self.db_manager = OracleConnectionManager(**ORACLE_CONFIG)
            
            # 2. 각 모듈 초기화
            self.intent_classifier = IntentClassifier(LLM_CONFIG)
            self.query_analyzer = QueryAnalyzer(LLM_CONFIG)
            self.text2sql_engine = Text2SQLConverter(self.db_manager, **LLM_CONFIG)
            
            # 3. 대화 상태 초기화
            self.conversation_state = "AWAITING_QUERY"
            self.conversation_context = {}
            
            logger.info("시스템 초기화가 완료되었습니다.")
        except Exception as e:
            logger.error(f"시스템 초기화 중 오류 발생: {e}")
            raise

    def _handle_greeting(self):
        print("AI: 안녕하세요! Oracle 데이터베이스에 대해 무엇이 궁금하신가요?")

    def _handle_unknown(self):
        print("AI: 죄송합니다. 데이터 조회와 관련된 질문만 답변할 수 있습니다. 다시 질문해주세요.")

    def _process_data_query(self, user_query: str):
        """데이터 조회 요청 처리 파이프라인"""
        try:
            # 1. 관련 테이블 및 스키마 정보 가져오기
            logger.info(f"관련 테이블 분석 중... (쿼리: {user_query})")
            relevant_tables = self.text2sql_engine.get_relevant_tables(user_query)
            if not relevant_tables:
                print("AI: 죄송합니다. 질문과 관련된 테이블을 찾지 못했습니다.")
                return
            schema_context = self.text2sql_engine.get_table_schemas_context(relevant_tables)

            # 2. 쿼리 명확성 분석
            logger.info("쿼리 명확성 분석 중...")
            analysis = self.query_analyzer.analyze(user_query, schema_context)

            if analysis["status"] == "clarification_needed":
                # 2-1. 정보가 불충분하면 되묻기
                print(f"AI: {analysis['question']}")
                self.conversation_state = "AWAITING_CLARIFICATION"
                self.conversation_context['original_query'] = user_query
                self.conversation_context['schema_context'] = schema_context
            elif analysis["status"] == "clear":
                # 2-2. 정보가 충분하면 SQL 생성 및 실행
                logger.info("SQL 생성 및 실행 중...")
                sql_query = self.text2sql_engine.convert_natural_language_to_sql(user_query)
                print(f"AI: 생성된 SQL 쿼리입니다.\n---\n{sql_query}\n---")
                
                result = self.db_manager.execute_query(sql_query)
                if result.success:
                    print(f"AI: 쿼리 실행 결과입니다. ({result.row_count}개 행)")
                    if result.data is not None and not result.data.empty:
                        print(result.data.to_string())
                    else:
                        print("결과 데이터가 없습니다.")
                else:
                    print(f"AI: 쿼리 실행 중 오류가 발생했습니다.\n오류: {result.error_message}")
                
                # 대화 상태 초기화
                self.reset_conversation()

        except Exception as e:
            logger.error(f"데이터 처리 중 오류 발생: {e}")
            print(f"AI: 요청을 처리하는 중에 오류가 발생했습니다.")
            self.reset_conversation()

    def reset_conversation(self):
        """대화 상태와 컨텍스트를 초기화합니다."""
        self.conversation_state = "AWAITING_QUERY"
        self.conversation_context = {}

    def start_chat(self):
        """대화형 루프를 시작합니다."""
        print("="*50)
        print("Oracle DB 대화형 쿼리 시스템에 오신 것을 환영합니다.")
        print("('quit' 또는 'exit'를 입력하여 종료)")
        print("="*50)

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    print("AI: 안녕히 가세요!")
                    self.db_manager.close()
                    break

                # 1. 의도 분류
                intent = self.intent_classifier.classify(user_input, self.conversation_state)
                logger.info(f"사용자 입력: '{user_input}', 분류된 의도: {intent}")

                # 2. 의도에 따른 처리 분기
                if intent == 'greeting':
                    self._handle_greeting()
                
                elif intent == 'clarification_response':
                    # 2-1. 사용자가 추가 정보를 입력한 경우
                    original_query = self.conversation_context.get('original_query', '')
                    # 대화의 맥락을 합쳐 새로운 쿼리 구성
                    combined_query = f"이전 질문 '{original_query}'에 대한 추가 정보입니다: '{user_input}'. 이 정보를 바탕으로 원래 질문에 다시 답해주세요."
                    self._process_data_query(combined_query)

                elif intent == 'data_query':
                    # 2-2. 새로운 데이터 조회 요청
                    self.reset_conversation() # 이전 대화가 있었다면 초기화
                    self._process_data_query(user_input)

                else: # 'unknown'
                    self._handle_unknown()

            except (KeyboardInterrupt, EOFError):
                print("\nAI: 강제 종료되었습니다. 안녕히 가세요!")
                self.db_manager.close()
                break
            except Exception as e:
                logger.error(f"대화 루프 중 예기치 않은 오류 발생: {e}")
                print("AI: 시스템에 문제가 발생했습니다. 세션을 다시 시작합니다.")
                self.reset_conversation()

if __name__ == "__main__":
    try:
        chatbot = Chatbot()
        chatbot.start_chat()
    except Exception as e:
        logger.critical(f"프로그램을 시작할 수 없습니다: {e}")
