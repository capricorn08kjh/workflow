#!/usr/bin/env python3
"""
MCP RAG 서버 - 회의록 검색 및 메타데이터 추출
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import re
from datetime import datetime

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

# 외부 라이브러리 imports
import spacy
from transformers import pipeline
import chromadb
from chromadb.config import Settings
import openai
from sentence_transformers import SentenceTransformer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """문서 처리 및 메타데이터 추출 클래스"""
    
    def __init__(self):
        # NER 모델 로드
        try:
            self.nlp = spacy.load("ko_core_news_sm")
        except OSError:
            logger.warning("Korean spaCy model not found. Using basic extraction.")
            self.nlp = None
        
        # 임베딩 모델 로드
        self.embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # 벡터DB 초기화
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="meeting_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 패턴 정의
        self.patterns = {
            "participants": r"참석자[:\s]*([^\n]+)",
            "date": r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
            "meeting_type": r"(정기회의|임시회의|킥오프|결과보고|프로젝트\s*회의)",
            "customer_company": r"고객사[:\s]*([^\s,\n]+)",
            "roles": r"(담당자|매니저|팀장|부장|이사|대리|과장)",
            "korean_names": r"[가-힣]{2,4}(?=\s*(?:님|씨|담당자|매니저|팀장|부장|이사|대리|과장)|\s*[가-힣]*\s*(?:담당자|매니저))"
        }
    
    def extract_entities_with_ner(self, text: str) -> Dict[str, List[str]]:
        """NER을 사용한 엔티티 추출"""
        entities = {"persons": [], "organizations": [], "locations": []}
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "PER"]:
                    entities["persons"].append(ent.text)
                elif ent.label_ in ["ORG", "ORGANIZATION"]:
                    entities["organizations"].append(ent.text)
                elif ent.label_ in ["LOC", "LOCATION"]:
                    entities["locations"].append(ent.text)
        
        return entities
    
    def extract_with_patterns(self, text: str) -> Dict[str, List[str]]:
        """정규식 패턴을 사용한 정보 추출"""
        metadata = {}
        
        for key, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            metadata[key] = list(set(matches)) if matches else []
        
        return metadata
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """종합적인 메타데이터 추출"""
        # NER 추출
        ner_entities = self.extract_entities_with_ner(text)
        
        # 패턴 매칭
        pattern_metadata = self.extract_with_patterns(text)
        
        # 결과 통합
        combined_metadata = {
            "persons": list(set(ner_entities["persons"] + pattern_metadata.get("korean_names", []))),
            "organizations": ner_entities["organizations"],
            "locations": ner_entities["locations"],
            "meeting_dates": pattern_metadata.get("date", []),
            "meeting_types": pattern_metadata.get("meeting_type", []),
            "customer_companies": pattern_metadata.get("customer_company", []),
            "roles": pattern_metadata.get("roles", []),
            "participants": pattern_metadata.get("participants", []),
            "extracted_at": datetime.now().isoformat()
        }
        
        return combined_metadata
    
    def chunk_document(self, text: str, chunk_size: int = 1000) -> List[str]:
        """문서를 청크로 분할"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def store_document(self, document_id: str, content: str, metadata: Dict[str, Any]):
        """문서를 벡터DB에 저장"""
        # 문서 청킹
        chunks = self.chunk_document(content)
        
        # 각 청크에 대해 임베딩 생성 및 저장
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            embedding = self.embedding_model.encode(chunk).tolist()
            
            # 청크별 메타데이터 생성
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "document_id": document_id
            })
            
            # ChromaDB에 저장
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[chunk_metadata]
            )
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """문서 검색"""
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # 벡터 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 결과 포맷팅
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # 유사도 점수로 변환
            })
        
        return formatted_results
    
    def advanced_search(self, query: str, person: str = None, company: str = None) -> List[Dict[str, Any]]:
        """고급 검색 (메타데이터 필터링 포함)"""
        # 필터 조건 구성
        where_conditions = {}
        
        if person:
            where_conditions["persons"] = {"$contains": person}
        
        if company:
            where_conditions["customer_companies"] = {"$contains": company}
        
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # 필터링된 검색
        if where_conditions:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=20,
                where=where_conditions,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=20,
                include=["documents", "metadatas", "distances"]
            )
        
        # 결과 포맷팅
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            })
        
        return formatted_results


class MCPRAGServer:
    """MCP RAG 서버 메인 클래스"""
    
    def __init__(self):
        self.server = Server("rag-server")
        self.processor = DocumentProcessor()
        self.setup_handlers()
    
    def setup_handlers(self):
        """MCP 핸들러 설정"""
        
        # 리소스 핸들러
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """사용 가능한 리소스 목록 반환"""
            return [
                Resource(
                    uri="documents://all",
                    name="All Documents",
                    description="모든 저장된 문서",
                    mimeType="application/json"
                ),
                Resource(
                    uri="metadata://extraction",
                    name="Metadata Extraction",
                    description="메타데이터 추출 서비스",
                    mimeType="application/json"
                )
            ]
        
        # 도구 핸들러
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """사용 가능한 도구 목록 반환"""
            return [
                Tool(
                    name="upload_document",
                    description="문서 업로드 및 메타데이터 추출",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string", "description": "문서 ID"},
                            "content": {"type": "string", "description": "문서 내용"},
                            "title": {"type": "string", "description": "문서 제목 (선택사항)"}
                        },
                        "required": ["document_id", "content"]
                    }
                ),
                Tool(
                    name="search_documents",
                    description="문서 검색",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "검색 쿼리"},
                            "top_k": {"type": "integer", "description": "반환할 결과 수", "default": 10}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="advanced_search",
                    description="고급 문서 검색 (메타데이터 필터링)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "검색 쿼리"},
                            "person": {"type": "string", "description": "찾을 인물명 (선택사항)"},
                            "company": {"type": "string", "description": "찾을 회사명 (선택사항)"},
                            "top_k": {"type": "integer", "description": "반환할 결과 수", "default": 10}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="extract_metadata",
                    description="텍스트에서 메타데이터 추출",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "분석할 텍스트"}
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="summarize_search_results",
                    description="검색 결과 요약",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "원본 쿼리"},
                            "person": {"type": "string", "description": "찾을 인물명"},
                            "max_results": {"type": "integer", "description": "최대 검색 결과 수", "default": 5}
                        },
                        "required": ["query", "person"]
                    }
                )
            ]
        
        # 도구 실행 핸들러
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """도구 실행"""
            
            if name == "upload_document":
                document_id = arguments["document_id"]
                content = arguments["content"]
                title = arguments.get("title", document_id)
                
                # 메타데이터 추출
                metadata = self.processor.extract_metadata(content)
                metadata["title"] = title
                
                # 문서 저장
                self.processor.store_document(document_id, content, metadata)
                
                return [TextContent(
                    type="text",
                    text=f"문서 '{document_id}' 업로드 완료.\n추출된 메타데이터:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}"
                )]
            
            elif name == "search_documents":
                query = arguments["query"]
                top_k = arguments.get("top_k", 10)
                
                results = self.processor.search_documents(query, top_k)
                
                return [TextContent(
                    type="text",
                    text=f"검색 결과 ({len(results)}개):\n{json.dumps(results, ensure_ascii=False, indent=2)}"
                )]
            
            elif name == "advanced_search":
                query = arguments["query"]
                person = arguments.get("person")
                company = arguments.get("company")
                top_k = arguments.get("top_k", 10)
                
                results = self.processor.advanced_search(query, person, company)
                
                return [TextContent(
                    type="text",
                    text=f"고급 검색 결과 ({len(results)}개):\n{json.dumps(results, ensure_ascii=False, indent=2)}"
                )]
            
            elif name == "extract_metadata":
                text = arguments["text"]
                metadata = self.processor.extract_metadata(text)
                
                return [TextContent(
                    type="text",
                    text=f"추출된 메타데이터:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}"
                )]
            
            elif name == "summarize_search_results":
                query = arguments["query"]
                person = arguments["person"]
                max_results = arguments.get("max_results", 5)
                
                # 고급 검색 실행
                results = self.processor.advanced_search(query, person=person)[:max_results]
                
                if not results:
                    return [TextContent(
                        type="text",
                        text=f"'{person}'님이 언급된 회의록을 찾을 수 없습니다."
                    )]
                
                # 결과 요약
                summary = self.generate_summary(results, query, person)
                
                return [TextContent(
                    type="text",
                    text=summary
                )]
            
            else:
                return [TextContent(
                    type="text",
                    text=f"알 수 없는 도구: {name}"
                )]
    
    def generate_summary(self, results: List[Dict[str, Any]], query: str, person: str) -> str:
        """검색 결과 요약 생성"""
        if not results:
            return f"'{person}'님이 언급된 회의록을 찾을 수 없습니다."
        
        summary = f"## {person}님이 언급된 회의록 검색 결과\n\n"
        summary += f"**검색 쿼리**: {query}\n"
        summary += f"**찾은 문서 수**: {len(results)}개\n\n"
        
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            content = result["content"]
            
            summary += f"### {i}. 문서 ID: {metadata.get('document_id', 'Unknown')}\n"
            
            # 메타데이터 정보
            if metadata.get("meeting_dates"):
                summary += f"**회의 날짜**: {', '.join(metadata['meeting_dates'])}\n"
            
            if metadata.get("meeting_types"):
                summary += f"**회의 유형**: {', '.join(metadata['meeting_types'])}\n"
            
            if metadata.get("customer_companies"):
                summary += f"**고객사**: {', '.join(metadata['customer_companies'])}\n"
            
            if metadata.get("persons"):
                summary += f"**참석자**: {', '.join(metadata['persons'])}\n"
            
            # 유사도 점수
            summary += f"**관련도**: {result['score']:.2f}\n"
            
            # 내용 미리보기 (200자 제한)
            preview = content[:200] + "..." if len(content) > 200 else content
            summary += f"**내용 미리보기**: {preview}\n\n"
        
        return summary
    
    async def run(self):
        """서버 실행"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)


if __name__ == "__main__":
    server = MCPRAGServer()
    asyncio.run(server.run())
