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
logger = logging.getLogger(__name__)

class VectorDBManager:
    """ChromaDB를 사용한 벡터 데이터베이스 관리 클래스"""
    
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
            logger.error(f"인덱스 구축 실
