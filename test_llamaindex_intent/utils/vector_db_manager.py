import chromadb
from chromadb.config import Settings
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

class VectorDBManager:
“”“ChromaDB를 사용한 벡터 데이터베이스 관리 클래스”””

```
def __init__(self, 
             chroma_db_path: str = "./chroma_db",
             collection_name: str = "company_documents",
             embedding_model: str = "text-embedding-ada-002",
             openai_api_base: Optional[str] = None,
             openai_api_key: str = "fake-key"):
    """
    초기화
    
    Args:
        chroma_db_path: ChromaDB 저장 경로
        collection_name: 컬렉션 이름
        embedding_model: 임베딩 모델명
        openai_api_base: OpenAI-like API 베이스 URL
        openai_api_key: API 키
    """
    self.chroma_db_path = chroma_db_path
    self.collection_name = collection_name
    
    # ChromaDB 클라이언트 설정
    self.chroma_client = chromadb.PersistentClient(
        path=chroma_db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # 임베딩 모델 설정 (OpenAI-like API 사용)
    if openai_api_base:
        self.embedding_model = OpenAILikeEmbedding(
            model=embedding_model,
            api_base=openai_api_base,
            api_key=openai_api_key
        )
    else:
        # 기본 OpenAI 임베딩 (테스트용)
        self.embedding_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=openai_api_key
        )
    
    # 컬렉션 초기화
    self.setup_collection()
    
    # 노드 파서 설정
    self.node_parser = SimpleNodeParser.from_defaults(
        chunk_size=1024,
        chunk_overlap=20
    )

def setup_collection(self):
    """ChromaDB 컬렉션 설정"""
    try:
        # 기존 컬렉션이 있으면 가져오기
        self.collection = self.chroma_client.get_collection(
            name=self.collection_name
        )
        logger.info(f"기존 컬렉션 '{self.collection_name}' 로드됨")
    except Exception:
        # 새 컬렉션 생성
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "회사 문서 벡터 저장소"}
        )
        logger.info(f"새 컬렉션 '{self.collection_name}' 생성됨")
    
    # LlamaIndex용 벡터 저장소 설정
    self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
    self.storage_context = StorageContext.from_defaults(
        vector_store=self.vector_store
    )

def load_documents_from_json(self, file_path: str) -> List[Document]:
    """JSON 파일에서 문서 로드"""
    documents = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # 문서 텍스트 구성
            if 'content' in item:
                text = f"제목: {item.get('title', '')}\n"
                text += f"날짜: {item.get('date', '')}\n"
                text += f"작성자: {item.get('author', '')}\n"
                
                if 'attendees' in item:
                    text += f"참석자: {', '.join(item['attendees'])}\n"
                
                text += f"내용:\n{item['content']}"
                
                # 메타데이터 설정
                metadata = {
                    'doc_id': item.get('meeting_id') or item.get('report_id'),
                    'title': item.get('title', ''),
                    'date': item.get('date', ''),
                    'author': item.get('author', ''),
                    'type': self._determine_doc_type(file_path),
                    'keywords': item.get('keywords', [])
                }
                
                if 'attendees' in item:
                    metadata['attendees'] = item['attendees']
                
                documents.append(Document(
                    text=text,
                    metadata=metadata
                ))
        
        logger.info(f"{file_path}에서 {len(documents)}개 문서 로드됨")
        
    except Exception as e:
        logger.error(f"문서 로드 실패 ({file_path}): {str(e)}")
    
    return documents

def _determine_doc_type(self, file_path: str) -> str:
    """파일 경로로부터 문서 타입 결정"""
    filename = Path(file_path).name
    if 'meeting' in filename:
        return 'meeting_minutes'
    elif 'monthly' in filename:
        return 'monthly_report'
    elif 'quality' in filename:
        return 'quality_report'
    elif 'safety' in filename:
        return 'safety_report'
    else:
        return 'document'

def load_all_documents(self, documents_dir: str = "documents") -> List[Document]:
    """documents 폴더의 모든 JSON 파일에서 문서 로드"""
    all_documents = []
    documents_path = Path(documents_dir)
    
    if not documents_path.exists():
        logger.warning(f"문서 폴더가 존재하지 않습니다: {documents_dir}")
        return all_documents
    
    json_files = list(documents_path.glob("*.json"))
    
    for json_file in json_files:
        documents = self.load_documents_from_json(str(json_file))
        all_documents.extend(documents)
    
    logger.info(f"총 {len(all_documents)}개 문서 로드됨")
    return all_documents

def build_index(self, documents: List[Document]) -> VectorStoreIndex:
    """문서들로부터 벡터 인덱스 구축"""
    try:
        # 문서를 노드로 분할
        nodes = self.node_parser.get_nodes_from_documents(documents)
        logger.info(f"{len(nodes)}개 노드로 분할됨")
        
        # 벡터 인덱스 생성
        index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            embed_model=self.embedding_model
        )
        
        logger.info("벡터 인덱스 구축 완료")
        return index
        
    except Exception as e:
        logger.error(f"인덱스 구축 실패: {str(e)}")
        raise

def get_existing_index(self) -> Optional[VectorStoreIndex]:
    """기존 인덱스 로드"""
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embedding_model
        )
        logger.info("기존 인덱스 로드됨")
        return index
    except Exception as e:
        logger.warning(f"기존 인덱스 로드 실패: {str(e)}")
        return None

def initialize_or_load_index(self, documents_dir: str = "documents") -> VectorStoreIndex:
    """인덱스 초기화 또는 로드"""
    # 기존 인덱스 확인
    existing_index = self.get_existing_index()
    
    if existing_index and self.collection.count() > 0:
        logger.info("기존 인덱스 사용")
        return existing_index
    
    # 새로운 인덱스 구축
    logger.info("새로운 인덱스 구축 시작")
    documents = self.load_all_documents(documents_dir)
    
    if not documents:
        logger.warning("로드된 문서가 없습니다")
        return VectorStoreIndex([], storage_context=self.storage_context)
    
    return self.build_index(documents)

def search_documents(self, 
                    query: str, 
                    top_k: int = 5,
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """문서 검색"""
    try:
        # 쿼리 조건 구성
        where_conditions = {}
        if filters:
            if 'doc_type' in filters:
                where_conditions['type'] = filters['doc_type']
            if 'author' in filters:
                where_conditions['author'] = filters['author']
            if 'date_from' in filters:
                # 날짜 필터링은 추후 구현
                pass
        
        # ChromaDB에서 직접 검색
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_conditions if where_conditions else None
        )
        
        # 결과 포맷팅
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0
                
                search_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity_score': 1 - distance,  # 유사도 점수로 변환
                    'doc_id': metadata.get('doc_id', ''),
                    'title': metadata.get('title', ''),
                    'author': metadata.get('author', ''),
                    'date': metadata.get('date', ''),
                    'type': metadata.get('type', '')
                })
        
        logger.info(f"검색 완료: {len(search_results)}개 결과")
        return search_results
        
    except Exception as e:
        logger.error(f"문서 검색 실패: {str(e)}")
        return []

def search_by_person(self, person_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """특정 인물이 언급된 문서 검색"""
    query = f"{person_name}이 언급된 문서"
    results = self.search_documents(query, top_k=top_k)
    
    # 추가 필터링: 내용에 해당 인물이 실제로 포함된 것만
    filtered_results = []
    for result in results:
        if person_name in result['content']:
            filtered_results.append(result)
    
    return filtered_results

def search_by_keywords(self, keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """키워드 기반 검색"""
    query = " ".join(keywords)
    return self.search_documents(query, top_k=top_k)

def search_by_date_range(self, 
                       start_date: str, 
                       end_date: str, 
                       query: str = "", 
                       top_k: int = 5) -> List[Dict[str, Any]]:
    """날짜 범위 기반 검색"""
    # 간단한 날짜 필터링 (추후 개선 가능)
    all_results = self.search_documents(query if query else "전체", top_k=100)
    
    filtered_results = []
    for result in all_results:
        doc_date = result['metadata'].get('date', '')
        if start_date <= doc_date <= end_date:
            filtered_results.append(result)
    
    return filtered_results[:top_k]

def get_collection_stats(self) -> Dict[str, Any]:
    """컬렉션 통계 정보"""
    try:
        count = self.collection.count()
        
        # 문서 타입별 분포 (샘플링을 통해 확인)
        sample_results = self.collection.get(limit=min(count, 100))
        type_counts = {}
        
        if sample_results['metadatas']:
            for metadata in sample_results['metadatas']:
                doc_type = metadata.get('type', 'unknown')
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            'total_documents': count,
            'document_types': type_counts,
            'collection_name': self.collection_name,
            'db_path': self.chroma_db_path
        }
        
    except Exception as e:
        logger.error(f"통계 정보 조회 실패: {str(e)}")
        return {}

def delete_collection(self):
    """컬렉션 삭제"""
    try:
        self.chroma_client.delete_collection(name=self.collection_name)
        logger.info(f"컬렉션 '{self.collection_name}' 삭제됨")
    except Exception as e:
        logger.error(f"컬렉션 삭제 실패: {str(e)}")

def rebuild_index(self, documents_dir: str = "documents"):
    """인덱스 재구축"""
    logger.info("인덱스 재구축 시작")
    
    # 기존 컬렉션 삭제
    self.delete_collection()
    
    # 새 컬렉션 생성
    self.setup_collection()
    
    # 문서 로드 및 인덱스 구축
    documents = self.load_all_documents(documents_dir)
    if documents:
        return self.build_index(documents)
    else:
        return VectorStoreIndex([], storage_context=self.storage_context)
```

class DocumentQueryEngine:
“”“문서 검색을 위한 쿼리 엔진”””

```
def __init__(self, vector_db_manager: VectorDBManager):
    self.vector_db = vector_db_manager
    self.index = vector_db_manager.initialize_or_load_index()
    
    # LlamaIndex 쿼리 엔진 생성
    self.query_engine = self.index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact"
    )

def query(self, question: str) -> Dict[str, Any]:
    """질의 처리"""
    try:
        # LlamaIndex 쿼리 엔진 사용
        response = self.query_engine.query(question)
        
        # ChromaDB 직접 검색도 함께 수행
        search_results = self.vector_db.search_documents(question, top_k=3)
        
        return {
            'answer': str(response),
            'source_documents': search_results,
            'query': question
        }
        
    except Exception as e:
        logger.error(f"쿼리 처리 실패: {str(e)}")
        return {
            'answer': f"쿼리 처리 중 오류가 발생했습니다: {str(e)}",
            'source_documents': [],
            'query': question
        }

def search_person_mentions(self, person_name: str) -> Dict[str, Any]:
    """특정 인물 언급 검색"""
    results = self.vector_db.search_by_person(person_name)
    
    summary = f"{person_name}님이 언급된 문서 {len(results)}건을 찾았습니다."
    if results:
        doc_types = list(set([r['type'] for r in results]))
        summary += f" 문서 유형: {', '.join(doc_types)}"
    
    return {
        'answer': summary,
        'source_documents': results,
        'query': f"{person_name}이 언급된 문서"
    }
```

def main():
“”“벡터 DB 관리자 테스트”””
try:
# 벡터 DB 관리자 초기화
print(“벡터 데이터베이스 초기화 중…”)
vector_db = VectorDBManager(
chroma_db_path=”./chroma_db”,
collection_name=“company_documents”,
embedding_model=“text-embedding-ada-002”,
openai_api_base=“http://localhost:8000/v1”,  # 예시 로컬 API
openai_api_key=“fake-key”
)

```
    # 인덱스 구축/로드
    print("인덱스 구축/로드 중...")
    index = vector_db.initialize_or_load_index()
    
    # 통계 정보 출력
    stats = vector_db.get_collection_stats()
    print(f"컬렉션 통계: {stats}")
    
    # 쿼리 엔진 초기화
    query_engine = DocumentQueryEngine(vector_db)
    
    # 테스트 쿼리들
    test_queries = [
        "김영철이 언급된 문서를 찾아줘",
        "품질 이슈 관련 문서",
        "4월 보고서",
        "안전사고 예방 방안"
    ]
    
    print("\n=== 테스트 쿼리 실행 ===")
    for query in test_queries:
        print(f"\n질의: {query}")
        if "김영철" in query:
            result = query_engine.search_person_mentions("김영철")
        else:
            result = query_engine.query(query)
        
        print(f"답변: {result['answer'][:200]}...")
        print(f"관련 문서 수: {len(result['source_documents'])}")
    
except Exception as e:
    print(f"테스트 실행 중 오류: {str(e)}")
```

if **name** == “**main**”:
main()