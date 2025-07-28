"""
향상된 Text2SQL 시스템 사용 예시
다양한 시나리오와 사용법을 보여주는 데모
"""

import json
import time
from enhanced_text2sql_main import EnhancedText2SQLSystem, WebAPIInterface

def demo_basic_usage():
    """기본 사용법 데모"""
    print("=" * 60)
    print("🚀 기본 사용법 데모")
    print("=" * 60)
    
    # 설정
    config = {
        'oracle_username': 'your_username',
        'oracle_password': 'your_password',
        'oracle_host': 'localhost',
        'oracle_port': 1521,
        'oracle_service_name': 'SMIP_DEV',
        'openai_api_base': 'http://localhost:8000/v1',
        'openai_api_key': 'your-api-key',
        'model_name': 'gpt-3.5-turbo'
    }
    
    try:
        # 시스템 초기화
        system = EnhancedText2SQLSystem(config)
        
        # 새 세션 시작
        session_id = system.start_session("demo_user")
        print(f"✅ 세션 시작: {session_id}")
        
        # 테스트 쿼리들
        test_queries = [
            "안녕하세요!",  # 인사말
            "사용 가능한 테이블을 알려주세요",  # 스키마 조회
            "모든 사용자 정보를 보여주세요",  # 단순 조회
            "부서별 직원 수를 계산해주세요",  # 집계 분석
            "최근 매출 현황을 분석해주세요"  # 명확화 필요한 쿼리
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 테스트 {i}: {query} ---")
            
            result = system.process_query(session_id, query)
            
            print(f"상태: {result.get('state', 'unknown')}")
            print(f"성공: {result.get('success', False)}")
            print(f"메시지: {result.get('message', '')[:200]}...")
            
            # 명확화가 필요한 경우 시뮬레이션
            if result.get('state') == 'clarifying' and result.get('question'):
                print("\n🔄 명확화 질문 받음, 자동 응답 중...")
                # 간단한 자동 응답 (실제로는 사용자 입력)
                auto_response = "최근 30일"
                clarify_result = system.process_query(session_id, auto_response)
                print(f"명확화 결과: {clarify_result.get('success', False)}")
            
            time.sleep(1)  # 처리 간격
        
        # 세션 종료
        system.end_session(session_id)
        print(f"\n✅ 세션 종료: {session_id}")
        
    except Exception as e:
        print(f"❌ 데모 실행 중 오류: {e}")
    finally:
        if 'system' in locals():
            system.close()

def demo_clarification_flow():
    """명확화 과정 데모"""
    print("\n" + "=" * 60)
    print("🔍 명확화 과정 데모")
    print("=" * 60)
    
    config = {
        'oracle_username': 'demo_user',
        'oracle_password': 'demo_pass',
        'oracle_host': 'localhost',
        'oracle_port': 1521,
        'oracle_service_name': 'DEMO_DB',
        'openai_api_base': 'http://localhost:8000/v1',
        'openai_api_key': 'demo-key',
        'model_name': 'gpt-3.5-turbo'
    }
    
    try:
        system = EnhancedText2SQLSystem(config)
        session_id = system.start_session("clarification_demo")
        
        # 모호한 쿼리로 시작
        ambiguous_query = "높은 매출을 가진 고객들을 최근에 찾아주세요"
        print(f"모호한 쿼리: {ambiguous_query}")
        
        result = system.process_query(session_id, ambiguous_query)
        
        # 명확화 과정 시뮬레이션
        if result.get('state') == 'clarifying':
            print("\n🔄 명확화 프로세스 시작...")
            
            # 가상의 사용자 응답들
            mock_responses = [
                "고객 테이블",  # 테이블 선택
                "최근 30일",   # 시간 범위
                "100만원 이상", # 조건 명확화
                "내림차순"     # 정렬 방식
            ]
            
            for i, response in enumerate(mock_responses):
                print(f"\n사용자 응답 {i+1}: {response}")
                result = system.process_query(session_id, response)
                
                print(f"응답 처리 결과: {result.get('success', False)}")
                
                if result.get('state') == 'completed':
                    print("✅ 명확화 완료!")
                    break
                elif result.get('question'):
                    print(f"다음 질문: {result['question'][:100]}...")
        
        # 대화 이력 확인
        history = system.get_conversation_history(session_id)
        if history:
            print(f"\n📚 최종 결과:")
            print(f"원본 쿼리: {history.get('original_query')}")
            print(f"정제된 쿼리: {history.get('final_query')}")
            print(f"최종 상태: {history.get('state')}")
        
        system.end_session(session_id)
        
    except Exception as e:
        print(f"❌ 명확화 데모 실행 중 오류: {e}")
    finally:
        if 'system' in locals():
            system.close()

def demo_web_api():
    """웹 API 인터페이스 데모"""
    print("\n" + "=" * 60)
    print("🌐 웹 API 인터페이스 데모")
    print("=" * 60)
    
    config = {
        'oracle_username': 'api_user',
        'oracle_password': 'api_pass',
        'oracle_host': 'localhost',
        'oracle_port': 1521,
        'oracle_service_name': 'API_DB',
        'openai_api_base': 'http://localhost:8000/v1',
        'openai_api_key': 'api-key',
        'model_name': 'gpt-3.5-turbo'
    }
    
    try:
        # 웹 API 인터페이스 초기화
        api = WebAPIInterface(config)
        
        # 세션 생성
        session_result = api.create_session("web_user")
        print(f"세션 생성: {session_result}")
        
        if session_result['success']:
            session_id = session_result['session_id']
            
            # 쿼리 처리
            queries = [
                "사용 가능한 테이블 목록을 알려주세요",
                "직원 테이블에서 모든 데이터를 조회해주세요"
            ]
            
            for query in queries:
                print(f"\n🔄 처리: {query}")
                result = api.process_query(session_id, query)
                print(f"결과: {result.get('success', False)}")
                print(f"상태: {result.get('state', 'unknown')}")
            
            # 세션 정보 조회
            session_info = api.get_session_info(session_id)
            print(f"\n📊 세션 정보: {session_info.get('success', False)}")
            
            # 시스템 상태 확인
            health = api.get_system_health()
            print(f"시스템 상태: {health.get('health', 'unknown')}")
            
            # 세션 종료
            end_result = api.end_session(session_id)
            print(f"세션 종료: {end_result.get('success', False)}")
        
    except Exception as e:
        print(f"❌ 웹 API 데모 실행 중 오류: {e}")
    finally:
        if 'api' in locals():
            api.close()

def demo_error_handling():
    """오류 처리 데모"""
    print("\n" + "=" * 60)
    print("⚠️ 오류 처리 데모")
    print("=" * 60)
    
    # 잘못된 설정으로 시스템 초기화 시도
    invalid_config = {
        'oracle_username': 'invalid_user',
        'oracle_password': 'invalid_pass',
        'oracle_host': 'invalid_host',
        'oracle_port': 9999,
        'oracle_service_name': 'INVALID_DB',
        'openai_api_base': 'http://invalid:8000/v1',
        'openai_api_key': 'invalid-key',
        'model_name': 'invalid-model'
    }
    
    try:
        print("잘못된 설정으로 시스템 초기화 시도...")
        system = EnhancedText2SQLSystem(invalid_config)
        
        # 연결 테스트
        if not system.test_connection():
            print("❌ 예상대로 데이터베이스 연결 실패")
        
        # 세션 시작은 가능 (DB 연결과 무관)
        session_id = system.start_session("error_demo")
        print(f"✅ 세션 시작은 성공: {session_id}")
        
        # 쿼리 처리 시 오류 발생 예상
        result = system.process_query(session_id, "사용자 정보를 조회해주세요")
        
        if not result.get('success'):
            print(f"❌ 예상대로 쿼리 처리 실패: {result.get('message', '')}")
        
        system.end_session(session_id)
        
    except Exception as e:
        print(f"❌ 시스템 초기화 실패 (예상됨): {e}")
    finally:
        if 'system' in locals():
            system.close()

def demo_performance_test():
    """성능 테스트 데모"""
    print("\n" + "=" * 60)
    print("🚀 성능 테스트 데모")
    print("=" * 60)
    
    config = {
        'oracle_username': 'perf_user',
        'oracle_password': 'perf_pass',
        'oracle_host': 'localhost',
        'oracle_port': 1521,
        'oracle_service_name': 'PERF_DB',
        'openai_api_base': 'http://localhost:8000/v1',
        'openai_api_key': 'perf-key',
        'model_name': 'gpt-3.5-turbo'
    }
    
    try:
        system = EnhancedText2SQLSystem(config)
        
        # 다중 세션 성능 테스트
        session_count = 5
        sessions = []
        
        print(f"다중 세션 생성: {session_count}개")
        start_time = time.time()
        
        for i in range(session_count):
            session_id = system.start_session(f"perf_user_{i}")
            sessions.append(session_id)
        
        creation_time = time.time() - start_time
        print(f"세션 생성 시간: {creation_time:.2f}초")
        
        # 동시 쿼리 처리 시뮬레이션
        test_query = "안녕하세요"  # 간단한 쿼리
        
        print(f"동시 쿼리 처리: {session_count}개")
        start_time = time.time()
        
        results = []
        for session_id in sessions:
            result = system.process_query(session_id, test_query)
            results.append(result)
        
        processing_time = time.time() - start_time
        print(f"쿼리 처리 시간: {processing_time:.2f}초")
        print(f"평균 처리 시간: {processing_time/session_count:.2f}초/쿼리")
        
        # 성공률 계산
        success_count = sum(1 for r in results if r.get('success'))
        success_rate = (success_count / len(results)) * 100
        print(f"성공률: {success_rate:.1f}% ({success_count}/{len(results)})")
        
        # 세션 정리
        for session_id in sessions:
            system.end_session(session_id)
        
        # 시스템 통계
        stats = system.get_system_stats()
        print(f"\n📊 최종 시스템 통계:")
        print(f"활성 세션: {stats.get('active_sessions', 0)}개")
        print(f"사용 가능한 테이블: {stats.get('available_tables_count', 0)}개")
        
    except Exception as e:
        print(f"❌ 성능 테스트 실행 중 오류: {e}")
    finally:
        if 'system' in locals():
            system.close()

def main():
    """모든 데모 실행"""
    print("🎯 향상된 Text2SQL 시스템 데모")
    print("시스템이 실제로 작동하려면 올바른 Oracle DB와 LLM API 설정이 필요합니다.")
    print("현재는 데모 목적으로 실행되며, 연결 오류가 예상됩니다.\n")
    
    demos = [
        ("기본 사용법", demo_basic_usage),
        ("명확화 과정", demo_clarification_flow),
        ("웹 API 인터페이스", demo_web_api),
        ("오류 처리", demo_error_handling),
        ("성능 테스트", demo_performance_test)
    ]
    
    for name, demo_func in demos:
        try:
            print(f"\n{'=' * 20} {name} 데모 시작 {'=' * 20}")
            demo_func()
            print(f"{'=' * 20} {name} 데모 완료 {'=' * 20}")
        except Exception as e:
            print(f"❌ {name} 데모 실행 중 오류: {e}")
        
        time.sleep(2)  # 데모 간격
    
    print("\n🎉 모든 데모 완료!")
    print("\n실제 사용을 위해서는 다음을 확인하세요:")
    print("1. Oracle 데이터베이스 연결 정보 설정")
    print("2. OpenAI-like API 서버 실행 및 설정")
    print("3. 필요한 Python 패키지 설치")
    print("4. config_example.json 파일을 참고하여 설정 파일 작성")

if __name__ == "__main__":
    main()