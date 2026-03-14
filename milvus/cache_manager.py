from datetime import datetime
from typing import Dict, Optional, List, Union
import asyncio
from loguru import logger
from .vector_store import VectorStore
from .models import CacheEntry, CacheSearchResult
import hashlib
from logos_server.conf.config import Config
import json
from app_chatting.config import embeddings
from rich.progress import Progress, SpinnerColumn, TextColumn
from pymilvus import Collection

class CacheManager:
    def __init__(self, collection: Collection, collection_name: str):
        self.collection = collection
        self.collection_name = collection_name
        self.vector_store = VectorStore(collection, collection_name)
        self.min_similarity_threshold = 0.85
        self.page_size = 10  # Assuming a default page_size
        # self.embedding_model = get_embedding_model()

    def _generate_query_id(self, query_text: str, email: str, project_id: str) -> str:
        """고유한 쿼리 ID 생성"""
        combined = f"{query_text}:{email}:{project_id}:{datetime.now().isoformat()}"
        return hashlib.sha256(combined.encode()).hexdigest()

    async def _generate_embedding(self, text: str) -> List[float]:
        """텍스트의 임베딩 벡터 생성"""
        try:
            embedding = embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 768  # 768 차원의 0 벡터 반환

    async def increment_hit_count(self, entry_id: str) -> bool:
        """캐시 엔트리의 조회수 증가"""
        try:
            collection = self.vector_store.get_collection()
            collection.load()
            
            # 현재 메타데이터 조회
            result = collection.query(
                expr=f'id == "{entry_id}"',
                output_fields=["metadata"]
            )
            
            if not result:
                logger.error(f"Entry not found: {entry_id}")
                return False
            
            # 메타데이터 업데이트
            metadata = result[0].get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            
            current_hits = metadata.get("hit_count", 0)
            metadata["hit_count"] = current_hits + 1
            
            # 업데이트 실행
            collection.update(
                expr=f'id == "{entry_id}"',
                data={"metadata": metadata}
            )
            
            logger.info(f"Hit count incremented for entry {entry_id}: {current_hits + 1}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to increment hit count: {e}")
            return False

    async def search_cache(self, query_text: str, query_vector: List[float], 
                         email: str, project_id: str, cache_management: Dict) -> CacheSearchResult:
        """캐시 검색"""
        try:
            # 결과 벡터도 생성
            result_vector = await self._generate_embedding(query_text)
            
            # 검색 조건 생성
            expr = f'email == "{email}" && project_id == "{project_id}"'
            
            # 검색 결과 조회
            results = await self.vector_store.search_similar(
                query_vector=query_vector,
                expr=expr,  # 이메일과 프로젝트 ID로 필터링
                top_k=5
            )
            
            if not results:
                return CacheSearchResult(found=False)
            
            best_match = results[0]
            similarity_score = best_match["score"]
            
            # 유사도가 너무 낮으면 캐시 미스로 처리
            if similarity_score < 0.7:
                return CacheSearchResult(found=False)
            
            # hit count 증가
            await self.increment_hit_count(best_match["id"])
            
            # CacheEntry 생성 및 반환
            entry_dict = {
                "query_id": best_match["id"],
                "query_text": best_match["query_text"],
                "query_vector": query_vector,
                "result_vector": result_vector,
                "email": best_match["email"],
                "result_text": best_match["result_text"],
                "references": best_match["metadata"].get("references", []),
                "pdf_names": best_match["metadata"].get("pdf_names", []),
                "cited_refs": best_match["metadata"].get("cited_refs", []),
                "pdf_info": best_match["metadata"].get("pdf_info", {}),
                "created_at": datetime.fromisoformat(best_match["created_at"]),
                "last_accessed": datetime.now(),
                "hit_count": best_match["metadata"].get("hit_count", 0) + 1,  # 증가된 hit count 반영
                "relevance_score": similarity_score,
                "project_id": best_match["project_id"],
                "cache_management": cache_management
            }
            
            return CacheSearchResult(
                found=True,
                entry=CacheEntry(**entry_dict),
                similarity_score=similarity_score
            )
            
        except Exception as e:
            logger.error(f"Cache search error: {e}")
            return CacheSearchResult(found=False)

    async def add_to_cache(self,
                          query_text: str,
                          query_vector: List[float],
                          result_vector: List[float],
                          email: str,
                          result_text: str,
                          references: List[Union[Dict, str, List]],
                          pdf_names: List[str],
                          cited_refs: List[Union[Dict, int]],
                          pdf_info: Union[List[Dict], Dict],
                          project_id: str,
                          cache_management: Dict) -> bool:
        """새로운 검색 결과를 캐시에 추가"""
        try:
            # references 데이터 변환
            processed_references = []
            for ref in references:
                if isinstance(ref, dict):
                    processed_references.append(ref)
                elif isinstance(ref, (list, tuple)):
                    # [score, text, filename, page] 형식 가정
                    processed_references.append({
                        "text": str(ref[1]) if len(ref) > 1 else "",
                        "score": float(ref[0]) if ref[0] else 0.0,
                        "file_name": str(ref[2]) if len(ref) > 2 else None,
                        "page": int(ref[3]) if len(ref) > 3 and ref[3] else None
                    })
                else:
                    # 문자열이나 다른 형식의 데이터는 text로 처리
                    processed_references.append({
                        "text": str(ref),
                        "score": 0.0,
                        "file_name": None,
                        "page": None
                    })

            # pdf_info 데이터 변환
            processed_pdf_info = {}
            if isinstance(pdf_info, list):
                for item in pdf_info:
                    if isinstance(item, dict):
                        processed_pdf_info.update(item)
            elif isinstance(pdf_info, dict):
                processed_pdf_info = pdf_info
            else:
                processed_pdf_info = {"info": str(pdf_info)}

            # cited_refs 데이터 변환
            processed_cited_refs = []
            for ref in cited_refs:
                if isinstance(ref, dict):
                    processed_cited_refs.append(ref)
                elif isinstance(ref, int):
                    processed_cited_refs.append({
                        "index": ref,
                        "text": None,
                        "page": None,
                        "file_name": None
                    })
                else:
                    processed_cited_refs.append({
                        "text": str(ref),
                        "index": None,
                        "page": None,
                        "file_name": None
                    })

            entry = CacheEntry(
                query_id=self._generate_query_id(query_text, email, project_id),
                query_text=query_text,
                query_vector=query_vector,
                result_vector=result_vector,
                email=email,
                result_text=result_text,
                references=processed_references,  # 처리된 references
                pdf_names=pdf_names or [],
                cited_refs=processed_cited_refs,  # 처리된 cited_refs
                pdf_info=processed_pdf_info,      # 처리된 pdf_info
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                hit_count=1,
                relevance_score=1.0,
                project_id=project_id,
                cache_management=cache_management
            )
            
            logger.debug(f"Processed entry data: {entry}")
            await self.vector_store.insert_entry(entry)
            return True
            
        except Exception as e:
            logger.error(f"Failed to add to cache: {e}")
            logger.error(f"Input data - references: {references}, pdf_info: {pdf_info}")
            return False

    async def search_by_result(self, result_text: str, top_k: int = 5) -> List[Dict]:
        """결과 텍스트 기반 검색"""
        try:
            # 결과 텍스트의 임베딩 생성
            result_vector = await self._generate_embedding(result_text)
            return await self.vector_store.search_similar(
                result_vector=result_vector, 
                top_k=top_k,
                field="result_vector"  # result_vector 필드로 검색
            )
        except Exception as e:
            logger.error(f"Failed to search by result: {e}")
            return []

    async def get_cache_entries(self, page: int = 1) -> tuple[List[Dict], int]:
        """캐시 엔트리 조회 (페이지네이션)"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Fetching cache entries...", total=None)
                
                # 전체 데이터 조회
                all_entries = await self.vector_store.get_all_entries()
                total_count = len(all_entries)
                
                if total_count == 0:
                    logger.warning("No entries found")
                    return [], 0
                
                # 페이지네이션 적용
                start_idx = (page - 1) * self.page_size
                end_idx = min(start_idx + self.page_size, total_count)
                
                page_entries = all_entries[start_idx:end_idx]
                
                # result_vector가 없는 경우 임베딩 생성
                for entry in page_entries:
                    if "result_text" in entry and "result_vector" not in entry:
                        entry["result_vector"] = await self._generate_embedding(entry["result_text"])
                
                # 결과 데이터 로깅
                logger.debug(f"Page entries data: {page_entries}")
                
                return page_entries, total_count
                
        except Exception as e:
            logger.error(f"Failed to get cache entries: {e}")
            return [], 0