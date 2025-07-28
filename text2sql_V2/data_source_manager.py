"""
데이터 소스 관리 모듈
Oracle DB와 ChromaDB를 통합 관리하는 모듈
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Oracle 연결
from text2sql_oracle import OracleConnectionManager, Text2SQLConverter

# ChromaDB 연결
import chromadb
from chromadb.config import Settings

# LlamaIndex
from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

logger = logging.getLogger(__name__)

@dataclass
class DataRetrievalResult:
    """데이터 조회 결과"""
    success: bool
    structured_data: Optional[pd.DataFrame] = None
    unstructured_data: Optional[List[Dict[str, Any]]] = None
    sql_query: Optional[str] = None
    search_query: Optional[str] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None

class ChromaDBManager:
    """ChromaDB 관리자"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8000,
                 embedding_model: str = "text-embedding-ada-002",
                 llm: Optional[OpenAILike] = None):
        """
        ChromaDB 관리자 초기화
        
        Args:
            host: ChromaDB 호스트
            port: ChromaDB 포트
            embedding_model: 임베딩 모델명
            llm: LLM 인스턴스
        """
        self.host = host
        self.port = port
        self.llm = llm
        
        # ChromaDB 클라이언트 초기화
        self.chroma_client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(
                chroma_server_host=host,
                chroma_server_http_port=port
            )
        )
        
        # 임베딩 모델 설정
        self.embedding_model = OpenAIEmbedding(
            model=embedding_model,
            api_key="your-api-key"  # 실제로는 환경변수에서 가져오기
        )
        
        # 컬렉션 캐시
        self.collections = {}
        self.vector_stores = {}
        self.indices = {}
        
        logger.info("ChromaDB 관리자 초기화 완료")
    
    def get_or_create_collection(self, collection_name: str) -> Any:
        """컬렉션 가져오기 또는 생성"""
        if collection_name not in self.collections:
            try:
                # 기존 컬렉션 가져오기 시도
                collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                # 컬렉션이 없으면 새로 생성
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": f"Collection for {collection_name}"}
                )
            
            self.collections[collection_name] = collection
            
            # VectorStore와 Index 설정
            vector_store = ChromaVectorStore(chroma_collection=collection)
            self.vector_stores[collection_name] = vector_store
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            service_context = ServiceContext.from_defaults(
                embed_model=self.embedding_model,
                llm=self.llm
            )
            
            # 인덱스 생성 또는 로드
            try:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    service_context=service_context
                )
            except Exception:
                index = VectorStoreIndex(
                    [],
                    storage_context=storage_context,
                    service_context=service_context
                )
            
            self.indices[collection_name] = index
            logger.info(f"컬렉션 설정 완료: {collection_name}")
        
        return self.collections[collection_name]
    
    def add_documents(self, collection_name: str, documents: List[Document]) -> bool:
        """문서 추가"""
        try:
            collection = self.get_or_create_collection(collection_name)
            index = self.indices[collection_name]
            
            # 문서를 인덱스에 추가
            for doc in documents:
                index.insert(doc)
            
            logger.info(f"문서 {len(documents)}개가 {collection_name}에 추가됨")
            return True
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            return False
    
    def search_documents(self, 
                        collection_name: str, 
                        query: str, 
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """문서 검색"""
        try:
            collection = self.get_or_create_collection(collection_name)
            index = self.indices[collection_name]
            
            # 쿼리 엔진 생성
            query_engine = index.as_query_engine(similarity_top_k=top_k)
            
            # 검색 실행
            response = query_engine.query(query)
            
            # 결과 포맷팅
            results = []
            for node in response.source_nodes:
                results.append({
                    'content': node.text,
                    'score': node.score,
                    'metadata': node.metadata,
                    'doc_id': node.node_id
                })
            
            logger.info(f"{collection_name}에서 '{query}' 검색: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            collection = self.get_or_create_collection(collection_name)
            count = collection.count()
            
            return {
                'name': collection_name,
                'document_count': count,
                'metadata': collection.metadata if hasattr(collection, 'metadata') else {}
            }
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {}
    
    def list_collections(self) -> List[str]:
        """모든 컬렉션 목록 조회"""
        try:
            collections = self.chroma_client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"컬렉션 목록 조회 실패: {e}")
            return []

class HybridDataSourceManager:
    """하이브리드 데이터 소스 관리자"""
    
    def __init__(self,
                 oracle_config: Dict[str, Any],
                 chromadb_config: Dict[str, Any],
                 llm: OpenAILike):
        """
        하이브리드 데이터 소스 관리자 초기화
        
        Args:
            oracle_config: Oracle 설정
            chromadb_config: ChromaDB 설정
            llm: LLM 인스턴스
        """
        self.llm = llm
        
        # Oracle 연결 관리자 초기화
        self.oracle_manager = OracleConnectionManager(
            username=oracle_config['username'],
            password=oracle_config['password'],
            host=oracle_config['host'],
            port=oracle_config.get('port', 1521),
            service_name=oracle_config.get('service_name', 'SMIP_DEV')
        )
        
        # Text2SQL 변환기 초기화
        self.text2sql_converter = Text2SQLConverter(
            oracle_manager=self.oracle_manager,
            openai_api_base=oracle_config.get('api_base'),
            openai_api_key=oracle_config.get('api_key', 'fake-key'),
            model_name=oracle_config.get('model_name', 'gpt-3.5-turbo')
        )
        
        # ChromaDB 관리자 초기화
        self.chromadb_manager = ChromaDBManager(
            host=chromadb_config.get('host', 'localhost'),
            port=chromadb_config.get('port', 8000),
            embedding_model=chromadb_config.get('embedding_model', 'text-embedding-ada-002'),
            llm=llm
        )
        
        # 실행기 (비동기 처리용)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("하이브리드 데이터 소스 관리자 초기화 완료")
    
    async def retrieve_structured_data(self, 
                                     natural_query: str,
                                     max_rows: int = 1000) -> DataRetrievalResult:
        """정형 데이터 조회 (Oracle)"""
        start_time = datetime.now()
        
        try:
            # 비동기로 SQL 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.text2sql_converter.process_natural_query,
                natural_query,
                True,
                max_rows
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result['success']:
                return DataRetrievalResult(
                    success=True,
                    structured_data=result['data'],
                    sql_query=result['sql_query'],
                    metadata={
                        'row_count': result['row_count'],
                        'relevant_tables': result.get('relevant_tables', [])
                    },
                    execution_time=execution_time
                )
            else:
                return DataRetrievalResult(
                    success=False,
                    sql_query=result.get('sql_query'),
                    error_message=result['error_message'],
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"정형 데이터 조회 실패: {e}")
            return DataRetrievalResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def retrieve_unstructured_data(self, 
                                       search_query: str,
                                       collection_name: str = "documents",
                                       top_k: int = 5) -> DataRetrievalResult:
        """비정형 데이터 조회 (ChromaDB)"""
        start_time = datetime.now()
        
        try:
            # 비동기로 문서 검색
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self.chromadb_manager.search_documents,
                collection_name,
                search_query,
                top_k
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if results:
                return DataRetrievalResult(
                    success=True,
                    unstructured_data=results,
                    search_query=search_query,
                    metadata={
                        'collection_name': collection_name,
                        'result_count': len(results),
                        'top_k': top_k
                    },
                    execution_time=execution_time
                )
            else:
                return DataRetrievalResult(
                    success=False,
                    search_query=search_query,
                    error_message="검색 결과가 없습니다.",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"비정형 데이터 조회 실패: {e}")
            return DataRetrievalResult(
                success=False,
                search_query=search_query,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def retrieve_hybrid_data(self, 
                                 structured_query: str,
                                 unstructured_query: str,
                                 collection_name: str = "documents",
                                 max_rows: int = 1000,
                                 top_k: int = 5) -> Tuple[DataRetrievalResult, DataRetrievalResult]:
        """하이브리드 데이터 조회 (Oracle + ChromaDB)"""
        # 병렬로 두 데이터 소스에서 조회
        structured_task = self.retrieve_structured_data(structured_query, max_rows)
        unstructured_task = self.retrieve_unstructured_data(unstructured_query, collection_name, top_k)
        
        structured_result, unstructured_result = await asyncio.gather(
            structured_task, unstructured_task
        )
        
        logger.info(f"하이브리드 조회 완료 - 정형: {structured_result.success}, 비정형: {unstructured_result.success}")
        
        return structured_result, unstructured_result
    
    def enrich_structured_data_with_documents(self, 
                                            structured_data: pd.DataFrame,
                                            unstructured_data: List[Dict[str, Any]],
                                            join_column: str = None) -> pd.DataFrame:
        """정형 데이터에 비정형 데이터 결합"""
        try:
            if structured_data.empty or not unstructured_data:
                return structured_data
            
            # 문서 데이터를 DataFrame으로 변환
            doc_df = pd.DataFrame([
                {
                    'doc_id': doc.get('doc_id', i),
                    'content': doc.get('content', ''),
                    'score': doc.get('score', 0.0),
                    'metadata': str(doc.get('metadata', {}))
                }
                for i, doc in enumerate(unstructured_data)
            ])
            
            if join_column and join_column in structured_data.columns:
                # 지정된 컬럼으로 조인
                enriched_data = structured_data.merge(
                    doc_df, 
                    left_on=join_column, 
                    right_on='doc_id', 
                    how='left'
                )
            else:
                # 단순히 상위 문서를 추가 컬럼으로 첨부
                if len(unstructured_data) > 0:
                    top_doc = unstructured_data[0]
                    structured_data['related_document'] = top_doc.get('content', '')[:200] + '...'
                    structured_data['document_score'] = top_doc.get('score', 0.0)
                
                enriched_data = structured_data
            
            logger.info(f"데이터 결합 완료: {len(enriched_data)}행")
            return enriched_data
            
        except Exception as e:
            logger.error(f"데이터 결합 실패: {e}")
            return structured_data
    
    def get_oracle_tables(self) -> Dict[str, str]:
        """Oracle 테이블 목록 조회"""
        return self.text2sql_converter.available_tables
    
    def get_chromadb_collections(self) -> List[str]:
        """ChromaDB 컬렉션 목록 조회"""
        return self.chromadb_manager.list_collections()
    
    def get_oracle_table_schema(self, table_name: str, schema_name: str = None) -> Dict[str, Any]:
        """Oracle 테이블 스키마 조회"""
        return self.oracle_manager.get_table_schema(table_name, schema_name)
    
    def get_chromadb_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """ChromaDB 컬렉션 정보 조회"""
        return self.chromadb_manager.get_collection_info(collection_name)
    
    def add_documents_to_chromadb(self, 
                                collection_name: str, 
                                documents: List[Dict[str, Any]]) -> bool:
        """ChromaDB에 문서 추가"""
        try:
            # 딕셔너리를 LlamaIndex Document로 변환
            doc_objects = []
            for doc in documents:
                doc_obj = Document(
                    text=doc.get('content', ''),
                    metadata=doc.get('metadata', {})
                )
                doc_objects.append(doc_obj)
            
            return self.chromadb_manager.add_documents(collection_name, doc_objects)
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            return False
    
    def test_connections(self) -> Dict[str, bool]:
        """모든 연결 테스트"""
        results = {}
        
        # Oracle 연결 테스트
        try:
            results['oracle'] = self.oracle_manager.test_connection()
        except Exception as e:
            logger.error(f"Oracle 연결 테스트 실패: {e}")
            results['oracle'] = False
        
        # ChromaDB 연결 테스트
        try:
            collections = self.chromadb_manager.list_collections()
            results['chromadb'] = True
        except Exception as e:
            logger.error(f"ChromaDB 연결 테스트 실패: {e}")
            results['chromadb'] = False
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 조회"""
        oracle_tables = self.get_oracle_tables()
        chromadb_collections = self.get_chromadb_collections()
        connections = self.test_connections()
        
        return {
            'oracle': {
                'connected': connections.get('oracle', False),
                'table_count': len(oracle_tables),
                'tables': list(oracle_tables.keys())[:10]  # 상위 10개만
            },
            'chromadb': {
                'connected': connections.get('chromadb', False),
                'collection_count': len(chromadb_collections),
                'collections': chromadb_collections
            },
            'hybrid_capabilities': {
                'structured_data': connections.get('oracle', False),
                'unstructured_data': connections.get('chromadb', False),
                'hybrid_queries': all(connections.values())
            }
        }
    
    def close(self):
        """모든 연결 종료"""
        try:
            self.oracle_manager.close()
            self.executor.shutdown(wait=True)
            logger.info("하이브리드 데이터 소스 관리자 종료 완료")
        except Exception as e:
            logger.error(f"연결 종료 중 오류: {e}")

class DataSourceRouter:
    """데이터 소스 라우터 - 쿼리에 따라 적절한 데이터 소스로 라우팅"""
    
    def __init__(self, hybrid_manager: HybridDataSourceManager):
        self.hybrid_manager = hybrid_manager
    
    async def route_and_execute(self, 
                              analysis_result,
                              refined_query: str = None) -> Union[DataRetrievalResult, Tuple[DataRetrievalResult, DataRetrievalResult]]:
        """
        분석 결과에 따라 적절한 데이터 소스로 라우팅하여 실행
        
        Args:
            analysis_result: 의도 분석 결과
            refined_query: 정제된 쿼리 (선택적)
            
        Returns:
            데이터 조회 결과
        """
        query = refined_query or analysis_result.extracted_entities.get('original_query', '')
        
        if analysis_result.data_source_type.value == 'oracle_only':
            return await self.hybrid_manager.retrieve_structured_data(query)
        
        elif analysis_result.data_source_type.value == 'chromadb_only':
            collection_name = self._determine_collection_name(analysis_result)
            search_terms = ' '.join(analysis_result.extracted_entities.get('text_search_terms', [query]))
            return await self.hybrid_manager.retrieve_unstructured_data(search_terms, collection_name)
        
        elif analysis_result.data_source_type.value == 'oracle_chromadb':
            # 하이브리드 쿼리 처리
            structured_query = self._extract_structured_part(query, analysis_result)
            unstructured_query = self._extract_unstructured_part(query, analysis_result)
            collection_name = self._determine_collection_name(analysis_result)
            
            return await self.hybrid_manager.retrieve_hybrid_data(
                structured_query, unstructured_query, collection_name
            )
        
        else:
            # 기본적으로 Oracle 조회
            return await self.hybrid_manager.retrieve_structured_data(query)
    
    def _determine_collection_name(self, analysis_result) -> str:
        """컬렉션명 결정"""
        collections = analysis_result.extracted_entities.get('chromadb_collections', [])
        if collections:
            return collections[0]
        return "documents"  # 기본 컬렉션
    
    def _extract_structured_part(self, query: str, analysis_result) -> str:
        """쿼리에서 정형 데이터 부분 추출"""
        # 테이블 관련 키워드가 있는 부분 추출
        oracle_tables = analysis_result.extracted_entities.get('oracle_tables', [])
        if oracle_tables:
            return f"{oracle_tables[0]} 테이블에서 " + query
        return query
    
    def _extract_unstructured_part(self, query: str, analysis_result) -> str:
        """쿼리에서 비정형 데이터 부분 추출"""
        # 텍스트 검색 키워드 추출
        search_terms = analysis_result.extracted_entities.get('text_search_terms', [])
        if search_terms:
            return ' '.join(search_terms)
        
        # 문서 관련 키워드 추출
        doc_keywords = ['문서', '내용', '원문', '텍스트']
        for keyword in doc_keywords:
            if keyword in query:
                return query.replace(keyword, '').strip()
        
        return query