"""
통합 Text2SQL 시스템 최종 데모
모든 기능을 종합적으로 테스트하고 시연하는 데모
"""

import asyncio
import json
import time
from typing import Dict, Any
from integrated_text2sql_system import IntegratedText2SQLSystem

class IntegratedSystemDemo:
    """통합 시스템 데모 클래스"""
    
    def __init__(self):
        self.config = self._load_demo_config()
        self.system = None
    
    def _load_demo_config(self) -> Dict[str, Any]:
        """데모용 설정 로드"""
        return {
            # LLM 설정
            'openai_api_base': 'http://localhost:8000/v1',
            'openai_api_key': 'your-api-key',
            'model_name': 'gpt-3.5-turbo',
            'temperature': 0.1,
            'max_tokens': 2000,
            
            # Oracle 설정
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
            
            # ChromaDB 설정
            'chromadb': {
                'host': 'localhost',
                'port': 8000,
                'embedding_model': 'text-embedding-ada-002'
            }
        }
    
    async def run_comprehensive_demo(self):
        """종합 데모 실행"""
        print("🚀 통합 Text2SQL 시스템 종합 데모")
        print("=" * 60)
        
        try:
            # 시스템 초기화
            print("📡 시스템 초기화 중...")
            self.system = IntegratedText2SQLSystem(self.config)
            
            # 시스템 상태 확인
            await self._check_system_status()
            
            # 다양한 시나리오 테스트
            await self._run_scenario_tests()
            
            # 대화형 데모
            # await self._run_interactive_demo()
            
        except Exception as e:
            print(f"❌ 데모 실행 중 오류: {e}")
        finally:
            if self.system:
                self.system.close()
    
    async def _check_system_status(self):
        """시스템 상태 확인"""
        print("\n📊 시스템 상태 확인")
        print("-" * 40)
        
        status = self.system.get_system_status()
        
        print(f"🔧 시스템 상태: {status['system_health']}")
        print(f"💾 Oracle 연결: {'✅' if status['data_sources']['oracle']['connected'] else '❌'}")
        print(f"📄 ChromaDB 연결: {'✅' if status['data_sources']['chromadb']['connected'] else '❌'}")
        print(f"📊 사용 가능한 테이블: {status['data_sources']['oracle']['table_count']}개")
        print(f"📚 문서 컬렉션: {status['data_sources']['chromadb']['collection_count']}개")
        
        if status['system_health'] != 'healthy':
            print("⚠️  일부 기능이 제한될 수 있습니다.")
    
    async def _run_scenario_tests(self):
        """시나리오별 테스트"""
        scenarios = [
            {
                'name': '인사 및 도움말',
                'queries': [
                    "안녕하세요!",
                    "도움말을 보여주세요",
                    "어떤 기능들이 있나요?"
                ]
            },
            {
                'name': '스키마 조회',
                'queries': [
                    "사용 가능한 테이블을 알려주세요",
                    "문서 컬렉션에는 어떤 것들이 있나요?",
                    "데이터 구조를 설명해주세요"
                ]
            },
            {
                'name': '정형 데이터 조회',
                'queries': [
                    "사용자 테이블의 모든 데이터를 보여주세요",
                    "부서별 직원 수를 계산해주세요",
                    "매출 상위 10개 제품을 조회해주세요"
                ]
            },
            {
                'name': '비정형 데이터 검색',
                'queries': [
                    "고객 서비스 관련 문서를 찾아주세요",
                    "제품 매뉴얼에서 설치 방법을 검색해주세요",
                    "회의록에서 예산 관련 내용을 찾아주세요"
                ]
            },
            {
                'name': '시각화 요청',
                'queries': [
                    "