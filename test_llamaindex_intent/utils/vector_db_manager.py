import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import VectorQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseSynthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentType:
    """문서 유형 상수"""
    MEETING_MINUTES = "meeting_minutes"
    MONTHLY_REPORT = "monthly_report"
    PROJECT_PLAN = "project_plan"
    PERFORMANCE_REVIEW = "performance_review"
    MARKET_ANALYSIS = "market_analysis"
    POLICY_DOCUMENT = "policy_document"
    TECHNICAL_REPORT = "technical_report"

class SampleDocumentGenerator:
    """샘플 문서 생성기"""
    
    def __init__(self):
        self.documents = []
        
    def generate_meeting_minutes(self) -> List[Document]:
        """회의록 문서 생성"""
        meeting_minutes = [
            {
                "title": "2024년 1분기 경영진 회의록",
                "date": "2024-03-15",
                "content": """
                2024년 1분기 경영진 회의

                참석자: 김영철 대표이사, 박민수 COO, 이수진 CFO, 정태호 CTO
                일시: 2024년 3월 15일 오후 2시

                안건 1: 1분기 실적 검토
                - 김영철 대표: "1분기 매출이 전년 동기 대비 15% 증가했습니다."
                - 이수진 CFO: "영업이익률도 12%로 목표치를 달성했습니다."
                - 공정별 실적을 보면 A공정이 가장 우수한 성과를 보였습니다.

                안건 2: 신제품 개발 현황
                - 정태호 CTO: "신제품 개발이 계획보다 2주 앞서 진행되고 있습니다."
                - 품질 테스트 결과 우수한 평가를 받았습니다.

                안건 3: 시장 확대 전략
                - 박민수 COO: "동남아 시장 진출을 위한 준비가 완료되어가고 있습니다."
                
                다음 회의: 2024년 4월 15일
                """,
                "participants": ["김영철", "박민수", "이수진", "정태호"],
                "keywords": ["실적", "매출", "영업이익", "공정", "신제품", "시장확대"]
            },
            {
                "title": "2024년 2분기 생산팀 회의록",
                "date": "2024-06-10", 
                "content": """
                2024년 2분기 생산팀 회의

                참석자: 최동훈 생산팀장, 김영철 품질관리자, 이민호 공정기술자, 송지은 안전관리자
                일시: 2024년 6월 10일 오전 10시

                안건 1: 공정별 생산 효율성 검토
                - 최동훈 팀장: "A공정의 효율성이 지난 분기 대비 8% 향상되었습니다."
                - 김영철 품질관리자: "품질 불량률도 0.5%로 크게 개선되었습니다."
                - B공정은 여전히 개선이 필요한 상황입니다.

                안건 2: 설비 업그레이드 계획
                - 이민호 공정기술자: "노후화된 설비 교체가 시급합니다."
                - 예산 승인 후 3분기 내 교체 예정입니다.

                안건 3: 안전 관리 현황
                - 송지은 안전관리자: "올해 안전사고 발생률이 작년 대비 40% 감소했습니다."
                
                액션 아이템:
                - B공정 효율성 개선 방안 수립 (담당: 이민호)
                - 설비 교체 예산안 작성 (담당: 최동훈)
                """,
                "participants": ["최동훈", "김영철", "이민호", "송지은"],
                "keywords": ["생산효율", "공정", "품질", "설비", "안전"]
            },
            {
                "title": "2024년 인사위원회 회의록",
                "date": "2024-05-20",
                "content": """
                2024년 인사위원회 회의

                참석자: 김영철 대표이사, 한정민 인사팀장, 이수진 CFO
                일시: 2024년 5월 20일 오후 3시

                안건 1: 상반기 직원 성과 평가
                - 한정민 팀장: "전체 직원 중 90%가 목표를 달성했습니다."
                - 김영철 대표: "우수 직원에 대한 인센티브 지급을 검토하겠습니다."
                - 특히 김영철 품질관리자의 성과가 뛰어났습니다.

                안건 2: 조직 개편 논의
                - 신규 사업부 신설을 위한 인력 충원 계획을 수립했습니다.
                - 경력직 채용을 통해 전문성을 강화하기로 했습니다.

                안건 3: 교육 훈련 계획
                - 전 직원 대상 디지털 전환 교육을 실시하기로 했습니다.
                - 외부 전문 기관과 협력하여 진행할 예정입니다.

                의결사항:
                - 우수 직원 포상 및 인센티브 지급 승인
                - 신규 채용 계획 승인
                """,
                "participants": ["김영철", "한정민", "이수진"],
                "keywords": ["성과평가", "인센티브", "조직개편", "교육훈련"]
            }
        ]
        
        documents = []
        for minutes in meeting_minutes:
            doc = Document(
                text=minutes["content"],
                metadata={
                    "doc_type": DocumentType.MEETING_MINUTES,
                    "title": minutes["title"],
                    "date": minutes["date"],
                    "participants": minutes["participants"],
                    "keywords": minutes["keywords"],
                    "source": "company_meetings"
                }
            )
            documents.append(doc)
            
        return documents
    
    def generate_monthly_reports(self) -> List[Document]:
        """월간 보고서 생성"""
        reports = [
            {
                "title": "2024년 4월 월간 경영보고서",
                "date": "2024-04-30",
                "content": """
                2024년 4월 월간 경영보고서

                1. 매출 현황
                - 4월 매출: 12억원 (전월 대비 5% 증가)
                - 누적 매출: 45억원 (전년 동기 대비 18% 증가)
                - 주요 증가 요인: 신제품 출시 효과 및 계절적 수요 증가

                2. 공정별 실적
                - A공정: 목표 달성률 110% (우수)
                - B공정: 목표 달성률 85% (개선 필요)
                - C공정: 목표 달성률 95% (양호)

                3. 품질 관리 현황
                - 불량률: 0.8% (목표: 1.0% 이하)
                - 고객 만족도: 4.2/5.0 (전월 대비 0.1점 상승)
                - 품질 개선 활동: ISO 인증 준비 진행 중

                4. 인력 현황
                - 총 직원 수: 156명
                - 신규 채용: 3명 (기술직 2명, 관리직 1명)
                - 퇴사자: 1명

                5. 재무 현황
                - 영업이익: 1.8억원 (영업이익률 15%)
                - 현금흐름: 양호
                - 투자 계획: R&D 투자 확대

                6. 리스크 요인
                - 원자재 가격 상승 압박
                - 인력 수급 어려움
                - 환율 변동 리스크

                작성자: 이수진 CFO
                """,
                "category": "경영보고서",
                "author": "이수진"
            },
            {
                "title": "2024년 5월 월간 생산보고서", 
                "date": "2024-05-31",
                "content": """
                2024년 5월 월간 생산보고서

                1. 생산 실적
                - 총 생산량: 15,000개 (계획 대비 102%)
                - 제품별 생산량:
                  * 제품A: 6,000개 (40%)
                  * 제품B: 5,500개 (37%)
                  * 제품C: 3,500개 (23%)

                2. 공정별 효율성
                - A공정: 효율성 92% (전월 90%)
                - B공정: 효율성 78% (전월 75%)
                - C공정: 효율성 88% (전월 87%)

                3. 품질 관리
                - 종합 불량률: 0.6% (목표치 달성)
                - 고객 클레임: 2건 (전월 3건)
                - 품질 개선 사항: 검사 프로세스 자동화 도입

                4. 설비 운영
                - 설비 가동률: 85%
                - 정기 보수: 예정대로 완료
                - 고장 건수: 2건 (경미한 수준)

                5. 안전 관리
                - 안전사고: 0건
                - 안전교육 실시: 전 직원 대상
                - 안전점검: 월 2회 정기 실시

                6. 개선 계획
                - B공정 효율성 향상을 위한 설비 개선
                - 품질 관리 시스템 업그레이드
                - 직원 기술 교육 강화

                작성자: 최동훈 생산팀장
                """,
                "category": "생산보고서",
                "author": "최동훈"
            },
            {
                "title": "2024년 6월 월간 마케팅보고서",
                "date": "2024-06-30", 
                "content": """
                2024년 6월 월간 마케팅보고서

                1. 시장 현황
                - 전체 시장 규모: 500억원 (전년 동기 대비 3% 성장)
                - 당사 시장점유율: 12% (전월 11.5%)
                - 주요 경쟁사 동향: 신제품 출시 러시

                2. 매출 분석
                - 6월 매출: 14억원 (전월 대비 17% 증가)
                - 채널별 매출:
                  * 온라인: 8억원 (57%)
                  * 오프라인: 6억원 (43%)
                - 지역별 매출: 수도권 60%, 지방 40%

                3. 고객 분석
                - 신규 고객: 250명
                - 고객 유지율: 85%
                - 평균 구매액: 45만원

                4. 프로모션 성과
                - 여름 시즌 프로모션 ROI: 320%
                - SNS 마케팅 도달률: 150만명
                - 브랜드 인지도: 38% (전월 35%)

                5. 경쟁 분석
                - 주요 경쟁사 A: 시장점유율 25%
                - 주요 경쟁사 B: 시장점유율 18%
                - 가격 경쟁 심화로 마진 압박 지속

                6. 향후 계획
                - 하반기 신제품 런칭 준비
                - 디지털 마케팅 비중 확대
                - 고객 관리 시스템(CRM) 도입

                작성자: 박민수 COO
                """,
                "category": "마케팅보고서", 
                "author": "박민수"
            }
        ]
        
        documents = []
        for report in reports:
            doc = Document(
                text=report["content"],
                metadata={
                    "doc_type": DocumentType.MONTHLY_REPORT,
                    "title": report["title"],
                    "date": report["date"],
                    "category": report["category"],
                    "author": report["author"],
                    "source": "monthly_reports"
                }
            )
            documents.append(doc)
            
        return documents
    
    def generate_project_plans(self) -> List[Document]:
        """프로젝트 계획서 생성"""
        plans = [
            {
                "title": "2024년 하반기 신제품 개발 프로젝트",
                "date": "2024-07-01",
                "content": """
                2024년 하반기 신제품 개발 프로젝트 계획서

                1. 프로젝트 개요
                - 프로젝트명: 차세대 스마트 제품 개발
                - 기간: 2024년 7월 ~ 12월 (6개월)
                - 예산: 5억원
                - 책임자: 정태호 CTO

                2. 목표
                - 시장 경쟁력 있는 신제품 개발
                - 기존 제품 대비 성능 30% 향상
                - 비용 효율성 20% 개선

                3. 세부 계획
                3.1 연구개발 단계 (7-8월)
                - 기술 조사 및 분석
                - 프로토타입 개발
                - 예산 배정: 2억원

                3.2 테스트 단계 (9-10월)
                - 성능 테스트
                - 안전성 검증
                - 예산 배정: 1.5억원

                3.3 상용화 준비 (11-12월)
                - 대량 생산 준비
                - 마케팅 전략 수립
                - 예산 배정: 1.5억원

                4. 참여 인력
                - 연구개발팀: 8명
                - 품질관리팀: 3명 (김영철 품질관리자 포함)
                - 생산팀: 5명
                - 마케팅팀: 3명

                5. 위험 요소
                - 기술적 어려움
                - 예산 초과 가능성
                - 일정 지연 리스크

                6. 성공 지표
                - 성능 목표 달성률
                - 예산 준수율
                - 일정 준수율

                승인자: 김영철 대표이사
                """,
                "project_manager": "정태호",
                "budget": 500000000
            },
            {
                "title": "공정 자동화 시스템 구축 프로젝트",
                "date": "2024-08-15",
                "content": """
                공정 자동화 시스템 구축 프로젝트 계획서

                1. 프로젝트 배경
                - 생산 효율성 향상 필요
                - 인력 부족 문제 해결
                - 품질 균일성 확보

                2. 프로젝트 범위
                - A공정 완전 자동화
                - B공정 부분 자동화
                - 통합 모니터링 시스템 구축

                3. 기대 효과
                - 생산성 40% 향상
                - 불량률 50% 감소
                - 인건비 30% 절감

                4. 추진 일정
                - 1단계 (9-10월): 시스템 설계
                - 2단계 (11-12월): 설비 설치
                - 3단계 (1-2월): 시운전 및 최적화

                5. 투자 비용
                - 총 투자액: 8억원
                - 설비비: 6억원
                - 시스템 구축비: 2억원

                6. 추진 조직
                - 프로젝트 총괄: 박민수 COO
                - 기술 책임자: 이민호 공정기술자
                - 품질 검토: 김영철 품질관리자

                작성자: 박민수 COO
                """,
                "project_manager": "박민수",
                "budget": 800000000
            }
        ]
        
        documents = []
        for plan in plans:
            doc = Document(
                text=plan["content"],
                metadata={
                    "doc_type": DocumentType.PROJECT_PLAN,
                    "title": plan["title"],
                    "date": plan["date"],
                    "project_manager": plan["project_manager"],
                    "budget": plan["budget"],
                    "source": "project_plans"
                }
            )
            documents.append(doc)
            
        return documents
    
    def generate_all_documents(self) -> List[Document]:
        """모든 샘플 문서 생성"""
        all_documents = []
        all_documents.extend(self.generate_meeting_minutes())
        all_documents.extend(self.generate_monthly_reports())
        all_documents.extend(self.generate_project_plans())
        
        logger.info(f"총 {len(all_documents)}개의 샘플 문서 생성 완료")
        return all_documents

class VectorDBManager:
    """ChromaDB 기반 벡터 데이터베이스 매니저"""
    
    def __init__(self, 
                 db_path: str = "./chroma_db",
                 collection_name: str = "company_documents",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        초기화
        
        Args:
            db_path: ChromaDB 저장 경로
            collection_name: 컬렉션 이름
            embedding_model: 임베딩 모델 (HuggingFace 또는 OpenAI)
        """
        
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # ChromaDB 클라이언트 설정
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 임베딩 모델 설정
        self.setup_embedding_model()
        
        # 컬렉션 생성 또는 로드
        self.setup_collection()
        
        # LlamaIndex 설정
        self.setup_llamaindex()
        
        logger.info(f"VectorDBManager 초기화 완료: {db_path}")
    
    def setup_embedding_model(self):
        """임베딩 모델 설정"""
        try:
            if self.embedding_model.startswith("text-embedding"):
                # OpenAI 임베딩 사용
                self.embed_model = OpenAIEmbedding(model=self.embedding_model)
                logger.info(f"OpenAI 임베딩 모델 로드: {self.embedding_model}")
            else:
                # HuggingFace 임베딩 사용 (기본값)
                self.embed_model = HuggingFaceEmbedding(
                    model_name=self.embedding_model,
                    trust_remote_code=True
                )
                logger.info(f"HuggingFace 임베딩 모델 로드: {self.embedding_model}")
                
        except Exception as e:
            logger.warning(f"임베딩 모델 로드 실패, 기본 모델 사용: {str(e)}")
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def setup_collection(self):
        """ChromaDB 컬렉션 설정"""
        try:
            # 기존 컬렉션 확인
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                self.chroma_collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"기존 컬렉션 로드: {self.collection_name}")
            else:
                self.chroma_collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Company documents collection"}
                )
                logger.info(f"새 컬렉션 생성: {self.collection_name}")
                
                # 샘플 문서 추가
                self.populate_sample_data()
                
        except Exception as e:
            logger.error(f"컬렉션 설정 실패: {str(e)}")
            raise
    
    def setup_llamaindex(self):
        """LlamaIndex 벡터 스토어 및 인덱스 설정"""
        try:
            # ChromaVectorStore 생성
            self.vector_store = ChromaVectorStore(
                chroma_collection=self.chroma_collection
            )
            
            # StorageContext 생성
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # 기존 인덱스가 있는지 확인
            if self.chroma_collection.count() > 0:
                # 기존 데이터로부터 인덱스 생성
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    embed_model=self.embed_model
                )
                logger.info("기존 벡터 스토어에서 인덱스 로드")
            else:
                # 빈 인덱스 생성
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=self.storage_context,
                    embed_model=self.embed_model
                )
                logger.info("새 벡터 인덱스 생성")
                
        except Exception as e:
            logger.error(f"LlamaIndex 설정 실패: {str(e)}")
            raise
    
    def populate_sample_data(self):
        """샘플 데이터 추가"""
        try:
            # 샘플 문서 생성
            doc_generator = SampleDocumentGenerator()
            documents = doc_generator.generate_all_documents()
            
            # 문서 추가
            self.add_documents(documents)
            
            logger.info(f"{len(documents)}개의 샘플 문서 추가 완료")
            
        except Exception as e:
            logger.error(f"샘플 데이터 추가 실패: {str(e)}")
    
    def add_documents(self, documents: List[Document]):
        """문서들을 벡터 데이터베이스에 추가"""
        try:
            # 문서 분할
            splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            
            nodes = splitter.get_nodes_from_documents(documents)
            
            # 인덱스에 노드 추가
            for node in nodes:
                self.index.insert(node)
            
            logger.info(f"{len(nodes)}개의 노드가 벡터 데이터베이스에 추가됨")
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {str(e)}")
            raise
    
    def search_documents(self, 
                        query: str, 
                        top_k: int = 5,
                        filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """문서 검색"""
        try:
            # 쿼리 엔진으로 검색
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="compact"
            )
            
            response = query_engine.query(query)
            
            # 결과 정리
            results = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    result = {
                        "content": node.text,
                        "score": getattr(node, 'score', 0.0),
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {str(e)}")
            return []
    
    def get_query_engine(self, **kwargs):
        """LlamaIndex 쿼리 엔진 반환"""
        return self.index.as_query_engine(
            similarity_top_k=kwargs.get('top_k', 5),
            response_mode=kwargs.get('response_mode', 'compact'),
            streaming=kwargs.get('streaming', False)
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보"""
        try:
            stats = {
                "collection_name": self.collection_name,
                "total_documents": self.chroma_collection.count(),
                "embedding_model": self.embedding_model,
                "db_path": str(self.db_path)
            }
            
            # 문서 타입별 통계
            if self.chroma_collection.count() > 0:
                # 메타데이터 기반 통계 (구현 필요 시)
                stats["document_types"] = {
                    "meeting_minutes": "회의록",
                    "monthly_reports": "월간보고서", 
                    "project_plans": "프로젝트계획서"
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"통계 정보 조회 실패: {str(e)}")
            return {}
    
    def delete_collection(self):
        """컬렉션 삭제"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info(f"컬렉션 삭제 완료: {self.collection_name}")
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패: {str(e)}")
    
    def backup_collection(self, backup_path: str):
        """컬렉션 백업"""
        try:
            # 간단한 백업 (실제로는 더 정교한 구현 필요)
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 컬렉션 정보 저장
            stats = self.get_statistics()
            with open(backup_dir / "collection_info.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"컬렉션 백업 완료: {backup_path}")
            
        except Exception as e:
            logger.error(f"컬렉션 백업 실패: {str(e)}")


class DocumentQueryEngine:
    """고급 문서 쿼리 엔진"""
    
    def __init__(self, vector_db_manager: VectorDBManager, llm=None):
        self.vector_db = vector_db_manager
        self.llm = llm
        self.setup_advanced_retrievers()
    
    def setup_advanced_retrievers(self):
        """고급 검색기 설정"""
        # 기본 벡터 검색기
        self.vector_retriever = VectorIndexRetriever(
            index=self.vector_db.index,
            similarity_top_k=10
        )
        
        # 응답 합성기
        self.response_synthesizer = ResponseSynthesizer.from_args(
            response_mode="compact",
            use_async=False
        )
        
        # 쿼리 엔진
        self.query_engine = RetrieverQueryEngine(
            retriever=self.vector_retriever,
            response_synthesizer=self.response_synthesizer
        )
    
    def search_by_person(self, person_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """특정 인물로 문서 검색"""
        query = f"{person_name}이 언급된 문서를 찾아줘"
        
        # 벡터 검색 수행
        results = self.vector_db.search_documents(query, top_k=top_k)
        
        # 인물명이 실제로 포함된 결과만 필터링
        filtered_results = []
        for result in results:
            if person_name in result["content"]:
                result["relevance_reason"] = f"{person_name}이 언급됨"
                filtered_results.append(result)
        
        return filtered_results
    
    def search_by_date_range(self, 
                           start_date: str, 
                           end_date: str, 
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """날짜 범위로 문서 검색"""
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # 전체 문서에서 날짜 범위에 해당하는 문서 필터링
            query = f"{start_date}부터 {end_date}까지의 문서"
            results = self.vector_db.search_documents(query, top_k=top_k * 2)
            
            filtered_results = []
            for result in results:
                if "metadata" in result and "date" in result["metadata"]:
                    doc_date = pd.to_datetime(result["metadata"]["date"])
                    if start_dt <= doc_date <= end_dt:
                        result["relevance_reason"] = f"날짜 범위 내 문서 ({result['metadata']['date']})"
                        filtered_results.append(result)
            
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"날짜 범위 검색 실패: {str(e)}")
            return []
    
    def search_by_document_type(self, 
                              doc_type: str, 
                              query: str = None, 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """문서 타입별 검색"""
        # 문서 타입 매핑
        type_mapping = {
            "회의록": DocumentType.MEETING_MINUTES,
            "보고서": DocumentType.MONTHLY_REPORT,
            "계획서": DocumentType.PROJECT_PLAN,
            "meeting": DocumentType.MEETING_MINUTES,
            "report": DocumentType.MONTHLY_REPORT,
            "plan": DocumentType.PROJECT_PLAN
        }
        
        target_type = type_mapping.get(doc_type.lower(), doc_type)
        
        if query:
            search_query = f"{doc_type} {query}"
        else:
            search_query = f"{doc_type} 문서"
        
        results = self.vector_db.search_documents(search_query, top_k=top_k * 2)
        
        # 문서 타입으로 필터링
        filtered_results = []
        for result in results:
            if ("metadata" in result and 
                result["metadata"].get("doc_type") == target_type):
                result["relevance_reason"] = f"{doc_type} 타입 문서"
                filtered_results.append(result)
        
        return filtered_results[:top_k]
    
    def semantic_search(self, 
                       query: str, 
                       top_k: int = 5,
                       include_metadata: bool = True) -> Dict[str, Any]:
        """의미 기반 고급 검색"""
        try:
            # LlamaIndex 쿼리 엔진 사용
            response = self.query_engine.query(query)
            
            search_result = {
                "query": query,
                "answer": str(response),
                "sources": []
            }
            
            # 소스 문서 정보 추가
            if hasattr(response, 'source_nodes'):
                for i, node in enumerate(response.source_nodes[:top_k]):
                    source_info = {
                        "rank": i + 1,
                        "content": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                        "score": getattr(node, 'score', 0.0)
                    }
                    
                    if include_metadata and hasattr(node, 'metadata'):
                        source_info["metadata"] = node.metadata
                    
                    search_result["sources"].append(source_info)
            
            return search_result
            
        except Exception as e:
            logger.error(f"의미 기반 검색 실패: {str(e)}")
            return {
                "query": query,
                "answer": f"검색 중 오류가 발생했습니다: {str(e)}",
                "sources": []
            }
    
    def multi_criteria_search(self, 
                            criteria: Dict[str, Any], 
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """다중 조건 검색"""
        results = []
        
        # 각 조건별로 검색 수행
        if "person" in criteria:
            person_results = self.search_by_person(criteria["person"], top_k)
            results.extend(person_results)
        
        if "date_range" in criteria:
            date_range = criteria["date_range"]
            date_results = self.search_by_date_range(
                date_range["start"], date_range["end"], top_k
            )
            results.extend(date_results)
        
        if "doc_type" in criteria:
            type_results = self.search_by_document_type(
                criteria["doc_type"], 
                criteria.get("query"), 
                top_k
            )
            results.extend(type_results)
        
        if "keywords" in criteria:
            for keyword in criteria["keywords"]:
                keyword_results = self.vector_db.search_documents(keyword, top_k//2)
                results.extend(keyword_results)
        
        # 중복 제거 및 점수 기반 정렬
        unique_results = {}
        for result in results:
            content_hash = hash(result["content"][:100])
            if content_hash not in unique_results:
                unique_results[content_hash] = result
            else:
                # 점수가 더 높은 결과로 업데이트
                if result.get("score", 0) > unique_results[content_hash].get("score", 0):
                    unique_results[content_hash] = result
        
        # 점수 기준 정렬
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x.get("score", 0),
            reverse=True
        )
        
        return sorted_results[:top_k]


def main():
    """VectorDBManager 사용 예시"""
    
    print("=== ChromaDB 기반 벡터 데이터베이스 관리자 ===")
    print("주요 기능:")
    print("1. ChromaDB를 이용한 벡터 데이터베이스 구축")
    print("2. HuggingFace/OpenAI 임베딩 모델 지원")
    print("3. LlamaIndex와의 완벽한 통합")
    print("4. 다양한 문서 타입 지원 (회의록, 보고서, 계획서)")
    print("5. 고급 검색 기능 (인물, 날짜, 타입별 검색)")
    print("-" * 50)
    
    try:
        # VectorDBManager 초기화
        print("\n1. VectorDBManager 초기화...")
        vector_db = VectorDBManager(
            db_path="./chroma_db",
            collection_name="company_documents",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 통계 정보 출력
        print("\n2. 데이터베이스 통계:")
        stats = vector_db.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # 검색 예시
        print("\n3. 검색 예시:")
        
        # 기본 검색
        print("\n3.1 기본 벡터 검색:")
        search_results = vector_db.search_documents("김영철 실적", top_k=3)
        for i, result in enumerate(search_results):
            print(f"   결과 {i+1}: {result['content'][:100]}...")
            print(f"   점수: {result.get('score', 0):.3f}")
            print(f"   메타데이터: {result.get('metadata', {})}")
            print()
        
        # 고급 문서 쿼리 엔진 사용
        print("\n3.2 고급 문서 쿼리 엔진:")
        doc_engine = DocumentQueryEngine(vector_db)
        
        # 인물별 검색
        print("\n   인물별 검색 (김영철):")
        person_results = doc_engine.search_by_person("김영철", top_k=2)
        for result in person_results:
            print(f"   - {result.get('relevance_reason', '관련성 높음')}")
            print(f"     {result['content'][:80]}...")
        
        # 문서 타입별 검색
        print("\n   문서 타입별 검색 (회의록):")
        type_results = doc_engine.search_by_document_type("회의록", top_k=2)
        for result in type_results:
            print(f"   - {result.get('relevance_reason', '타입 일치')}")
            print(f"     제목: {result.get('metadata', {}).get('title', '제목 없음')}")
        
        # 의미 기반 검색
        print("\n   의미 기반 검색:")
        semantic_result = doc_engine.semantic_search("공정 효율성 개선 방안")
        print(f"   질문: {semantic_result['query']}")
        print(f"   답변: {semantic_result['answer'][:200]}...")
        print(f"   소스 개수: {len(semantic_result['sources'])}")
        
        print("\n4. 사용 가능한 문서 타입:")
        doc_types = [
            "회의록 (meeting_minutes): 경영진, 생산팀, 인사위원회 회의록",
            "월간보고서 (monthly_report): 경영, 생산, 마케팅 보고서",
            "프로젝트계획서 (project_plan): 신제품 개발, 자동화 프로젝트"
        ]
        for doc_type in doc_types:
            print(f"   - {doc_type}")
        
        print("\n5. 주요 등장인물:")
        people = [
            "김영철: 대표이사, 품질관리자",
            "박민수: COO (최고운영책임자)",
            "이수진: CFO (최고재무책임자)",
            "정태호: CTO (최고기술책임자)",
            "최동훈: 생산팀장",
            "한정민: 인사팀장"
        ]
        for person in people:
            print(f"   - {person}")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        print("ChromaDB나 임베딩 모델 설치를 확인해주세요.")
        print("설치 명령어:")
        print("pip install chromadb")
        print("pip install sentence-transformers")


if __name__ == "__main__":
    main()
