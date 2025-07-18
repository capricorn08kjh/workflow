#!/usr/bin/env python3
"""
MCP RAG 서버 - LlamaIndex + 회사 API 연동
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import re
from datetime import datetime
import os
from dataclasses import dataclass

# MCP 관련 imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, 
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource,
    LoggingLevel
)

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    StorageContext,
    Settings,
    ServiceContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    BaseExtractor,
    KeywordExtractor,
    EntityExtractor
)
from llama_index.core.schema import BaseNode
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import ResponseMode

# 외부 라이브러리
import chromadb
import httpx
import spacy

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompanyAPIConfig:
    """회사 API 설정"""
    base_url: str
    api_key: str
    model_name: str = "your-company-model"
    embedding_model: str = "your-company-embedding"
    timeout: int = 30

class KoreanEntityExtractor(BaseExtractor):
    """한국어 엔티티 추출기"""
    
    def __init__(self, llm: OpenAILike, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        
        # 한국어 패턴 정의
        self.patterns = {
            "persons": r"[가-힣]{2,4}(?=\s*(?:님|씨|담당자|매니저|팀장|부장|이사|대리|과장))",
            "meeting_dates": r"(\d{4}[-/년]\d{1,2}[-/월]\d{1,2}[일]?)",
            "meeting_types": r"(정기회의|임시회의|킥오프|결과보고|프로젝트\s*회의|워크숍|브리핑)",
            "companies": r"([가-힣\w]+(?:주식회사|회사|그룹|코퍼레이션|파트너스|솔루션|시스템|테크놀로지))",
            "roles": r"(담당자|매니저|팀장|부장|이사|대리|과장|PM|PL|개발자|디자이너)",
            "projects": r"([가-힣\w]+\s*(?:프로젝트|과제|업무|개발|구축|운영))",
            "customer_indicators": r"(고객사|클라이언트|발주처|파트너사)"
        }
    
    async def aextract(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        """비동기 엔티티 추출"""
        metadata_list = []
        
        for node in nodes:
            try:
                # 패턴 기반 추출
                pattern_metadata = self.extract_with_patterns(node.text)
                
                # LLM 기반 추출 (선택적)
                if self.should_use_llm(pattern_metadata):
                    llm_metadata = await self.extract_with_llm(node.text)
                    metadata = self.merge_metadata(pattern_metadata, llm_metadata)
                else:
                    metadata = pattern_metadata
                
                metadata_list.append(metadata)
                
            except Exception as e:
                logger.error(f"엔티티 추출 실패: {e}")
                metadata_list.append({})
        
        return metadata_list
    
    def extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """패턴 기반 메타데이터 추출"""
        metadata = {}
        
        for key, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # 중복 제거 및 정리
                unique_matches = list(set(match.strip() for match in matches if match.strip()))
                metadata[key] = unique_matches
        
        # 고객사 담당자 특별 처리
        if metadata.get("customer_indicators") and metadata.get("persons"):
            metadata["customer_contacts"] = metadata["persons"]
        
        return metadata
    
    def should_use_llm(self, pattern_metadata: Dict[str, Any]) -> bool:
        """LLM 추출 필요 여부 판단"""
        # 패턴으로 추출한 데이터가 부족한 경우에만 LLM 사용
        essential_fields = ["persons", "companies", "meeting_dates"]
        missing_fields = sum(1 for field in essential_fields if not pattern_metadata.get(field))
        return missing_fields >= 2
    
    async def extract_with_llm(self, text: str) -> Dict[str, Any]:
        """LLM을 사용한 정밀 추출"""
        prompt = f"""
        다음 회의록 텍스트에서 구조화된 정보를 추출해주세요.
        
        텍스트: {text[:1500]}
        
        다음 JSON 형태로 반환하세요:
        {{
            "persons": ["인물명1", "인물명2"],
            "companies": ["회사명1", "회사명2"],
            "meeting_dates": ["날짜1", "날짜2"],
            "meeting_types": ["회의유형1"],
            "roles": ["역할1", "역할2"],
            "projects": ["프로젝트명1"],
            "customer_contacts": ["고객사담당자명"]
        }}
        """
        
        try:
            response = await self.llm.acomplete(prompt)
            # JSON 파싱 시도
            result = json.loads(response.text)
            return result
        except Exception as e:
            logger.error(f"LLM 추출 실패: {e}")
            return {}
    
    def merge_metadata(self, pattern_data: Dict[str, Any], llm_data: Dict[str, Any]) -> Dict[str, Any]:
        """패턴 기반과 LLM 기반 결과 병합"""
        merged = pattern_data.copy()
        
        for key, value in llm_data.items():
            if key in merged and merged[key]:
                # 기존 데이터와 LLM 데이터 병합
                merged[key] = list(set(merged[key] + value))
            elif value:
                # LLM에서만 추출된 데이터 추가
                merged[key] = value
        
        return merged

class CompanyAPIClient:
    """회사 API 클라이언트"""
    
    def __init__(self, config: CompanyAPIConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)
    
    async def get_embedding(self, text: str) -> List[float]:
        """임베딩 생성"""
        try:
            response = await self.client.post(
                f"{self.config.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={
                    "model": self.config.embedding_model,
                    "input": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    async def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """채팅 완성"""
        try:
            response = await self.client.post(
                f"{self.config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={
                    "model": self.config.model_name,
                    "messages": messages,
                    "temperature": 0.1
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"채팅 완성 실패: {e}")
            raise
    
    async def close(self):
        """클라이언트 종료"""
        await self.client.aclose()

class RAGDocumentProcessor:
    """LlamaIndex 기반 문서 처리기"""
    
    def __init__(self, api_config: CompanyAPIConfig):
        self.api_config = api_config
        self.api_client = CompanyAPIClient(api_config)
        
        # LlamaIndex 설정
        self.setup_llamaindex()
        
        # ChromaDB 설정
        self.setup_vector_store()
        
        # 인덱스 초기화
        self.index = None
        self.query_engine = None
        
        # 메타데이터 추출기
        self.entity_extractor = None
    
    def setup_llamaindex(self):
        """LlamaIndex 설정"""
        # 회사 API를 사용하는 LLM 설정
        self.llm = OpenAILike(
            api_base=self.api_config.base_url,
            api_key=self.api_config.api_key,
            model=self.api_config.model_name,
            timeout=self.api_config.timeout
        )
        
        # 회사 API를 사용하는 임베딩 설정
        self.embed_model = OpenAIEmbedding(
            api_base=self.api_config.base_url,
            api_key=self.api_config.api_key,
            model=self.api_config.embedding_model
        )
        
        # 전역 설정
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # 한국어 친화적 청킹
        Settings.node_parser = SentenceSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    
    def setup_vector_store(self):
        """벡터 스토어 설정"""
        # ChromaDB 클라이언트 생성
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # 컬렉션 생성/가져오기
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name="meeting_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # ChromaVectorStore 설정
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection
        )
        
        # StorageContext 생성
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
    
    async def initialize_extractors(self):
        """추출기 초기화"""
        if not self.entity_extractor:
            self.entity_extractor = KoreanEntityExtractor(llm=self.llm)
    
    async def process_documents(self, documents: List[Dict[str, Any]]):
        """문서 처리 및 인덱싱"""
        await self.initialize_extractors()
        
        # Document 객체 생성
        llama_documents = []
        for doc_data in documents:
            # 메타데이터 추출
            metadata = await self.extract_metadata(doc_data["content"])
            
            # 기본 메타데이터와 병합
            full_metadata = {
                **doc_data.get("metadata", {}),
                **metadata,
                "document_id": doc_data["id"],
                "processed_at": datetime.now().isoformat()
            }
            
            # LlamaIndex Document 생성
            doc = Document(
                text=doc_data["content"],
                metadata=full_metadata,
                id_=doc_data["id"]
            )
            llama_documents.append(doc)
        
        # 인덱스 생성/업데이트
        if self.index is None:
            self.index = VectorStoreIndex.from_documents(
                llama_documents,
                storage_context=self.storage_context,
                show_progress=True
            )
        else:
            # 기존 인덱스에 문서 추가
            for doc in llama_documents:
                self.index.insert(doc)
        
        # 쿼리 엔진 생성
        self.setup_query_engine()
    
    async def extract_metadata(self, text: str) -> Dict[str, Any]:
        """메타데이터 추출"""
        # 임시 노드 생성
        node = BaseNode(text=text)
        
        # 엔티티 추출
        metadata_list = await self.entity_extractor.aextract([node])
        return metadata_list[0] if metadata_list else {}
    
    def setup_query_engine(self):
        """쿼리 엔진 설정"""
        if not self.index:
            return
        
        # 고급 리트리버 설정
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10
        )
        
        # 후처리기 설정
        postprocessor = SimilarityPostprocessor(
            similarity_cutoff=0.7
        )
        
        # 쿼리 엔진 생성
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor],
            response_mode=ResponseMode.COMPACT
        )
    
    async def search_documents(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """문서 검색"""
        if not self.query_engine:
            return {"error": "인덱스가 초기화되지 않았습니다."}
        
        try:
            # 필터 적용된 검색 (메타데이터 필터링)
            if filters:
                # 메타데이터 필터를 쿼리에 추가
                enhanced_query = self.enhance_query_with_filters(query, filters)
            else:
                enhanced_query = query
            
            # 검색 실행
            response = await self.query_engine.aquery(enhanced_query)
            
            # 결과 포맷팅
            return {
                "response": str(response),
                "source_nodes": [
                    {
                        "content": node.text,
                        "metadata": node.metadata,
                        "score": node.score
                    }
                    for node in response.source_nodes
                ],
                "metadata": response.metadata
            }
        
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return {"error": str(e)}
    
    def enhance_query_with_filters(self, query: str, filters: Dict[str, Any]) -> str:
        """필터를 적용한 쿼리 강화"""
        enhanced_parts = [query]
        
        if filters.get("person"):
            enhanced_parts.append(f"참석자: {filters['person']}")
        
        if filters.get("company"):
            enhanced_parts.append(f"회사: {filters['company']}")
        
        if filters.get("date_range"):
            enhanced_parts.append(f"날짜: {filters['date_range']}")
        
        if filters.get("meeting_type"):
            enhanced_parts.append(f"회의유형: {filters['meeting_type']}")
        
        return " ".join(enhanced_parts)
    
    async def summarize_results(self, query: str, person: str = None) -> str:
        """결과 요약"""
        # 필터 적용
        filters = {}
        if person:
            filters["person"] = person
        
        # 검색 실행
        search_results = await self.search_documents(query, filters)
        
        if "error" in search_results:
            return f"검색 중 오류 발생: {search_results['error']}"
        
        if not search_results["source_nodes"]:
            return f"'{person or '해당 조건'}' 관련 문서를 찾을 수 없습니다."
        
        # 요약 생성
        summary_prompt = f"""
        검색 쿼리: {query}
        {f"관련 인물: {person}" if person else ""}
        
        검색 결과를 바탕으로 간결하고 명확한 요약을 제공해주세요.
        다음 정보를 포함해주세요:
        1. 찾은 문서 수와 관련성
        2. 주요 내용 요약
        3. 언급된 인물들의 역할과 발언 내용
        4. 중요한 날짜나 결정사항
        
        검색 결과:
        {search_results['response']}
        """
        
        try:
            summary = await self.api_client.chat_completion([
                {"role": "user", "content": summary_prompt}
            ])
            return summary
        except Exception as e:
            logger.error(f"요약 생성 실패: {e}")
            return f"요약 생성 중 오류 발생: {str(e)}"

class MCPRAGServer:
    """MCP RAG 서버"""
    
    def __init__(self, api_config: CompanyAPIConfig):
        self.server = Server("rag-server")
        self.processor = RAGDocumentProcessor(api_config)
        self.setup_handlers()
    
    def setup_handlers(self):
        """MCP 핸들러 설정"""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="documents://index",
                    name="Document Index",
                    description="LlamaIndex 기반 문서 인덱스",
                    mimeType="application/json"
                ),
                Resource(
                    uri="metadata://extractor",
                    name="Metadata Extractor",
                    description="한국어 메타데이터 추출기",
                    mimeType="application/json"
                )
            ]
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="upload_documents",
                    description="문서 업로드 및 인덱싱",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "documents": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "content": {"type": "string"},
                                        "metadata": {"type": "object"}
                                    },
                                    "required": ["id", "content"]
                                }
                            }
                        },
                        "required": ["documents"]
                    }
                ),
                Tool(
                    name="search_documents",
                    description="문서 검색",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "person": {"type": "string"},
                            "company": {"type": "string"},
                            "meeting_type": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="summarize_meeting_records",
                    description="회의록 검색 및 요약",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "person": {"type": "string"},
                            "include_metadata": {"type": "boolean", "default": True}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="extract_metadata",
                    description="텍스트 메타데이터 추출",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"}
                        },
                        "required": ["text"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            
            if name == "upload_documents":
                documents = arguments["documents"]
                
                try:
                    await self.processor.process_documents(documents)
                    return [TextContent(
                        type="text",
                        text=f"✅ {len(documents)}개 문서 업로드 및 인덱싱 완료"
                    )]
                except Exception as e:
                    return [TextContent(
                        type="text",
                        text=f"❌ 문서 처리 실패: {str(e)}"
                    )]
            
            elif name == "search_documents":
                query = arguments["query"]
                filters = {k: v for k, v in arguments.items() if k != "query" and v}
                
                try:
                    results = await self.processor.search_documents(query, filters)
                    return [TextContent(
                        type="text",
                        text=json.dumps(results, ensure_ascii=False, indent=2)
                    )]
                except Exception as e:
                    return [TextContent(
                        type="text",
                        text=f"❌ 검색 실패: {str(e)}"
                    )]
            
            elif name == "summarize_meeting_records":
                query = arguments["query"]
                person = arguments.get("person")
                
                try:
                    summary = await self.processor.summarize_results(query, person)
                    return [TextContent(
                        type="text",
                        text=summary
                    )]
                except Exception as e:
                    return [TextContent(
                        type="text",
                        text=f"❌ 요약 생성 실패: {str(e)}"
                    )]
            
            elif name == "extract_metadata":
                text = arguments["text"]
                
                try:
                    metadata = await self.processor.extract_metadata(text)
                    return [TextContent(
                        type="text",
                        text=json.dumps(metadata, ensure_ascii=False, indent=2)
                    )]
                except Exception as e:
                    return [TextContent(
                        type="text",
                        text=f"❌ 메타데이터 추출 실패: {str(e)}"
                    )]
            
            else:
                return [TextContent(
                    type="text",
                    text=f"❌ 알 수 없는 도구: {name}"
                )]
    
    async def run(self):
        """서버 실행"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)

def main():
    """메인 함수"""
    # 환경 변수에서 API 설정 로드
    api_config = CompanyAPIConfig(
        base_url=os.getenv("COMPANY_API_BASE_URL", "https://api.yourcompany.com"),
        api_key=os.getenv("COMPANY_API_KEY", "your-api-key"),
        model_name=os.getenv("COMPANY_MODEL_NAME", "your-model"),
        embedding_model=os.getenv("COMPANY_EMBEDDING_MODEL", "your-embedding-model")
    )
    
    # 서버 생성 및 실행
    server = MCPRAGServer(api_config)
    asyncio.run(server.run())

if __name__ == "__main__":
    main()
