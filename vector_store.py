from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from typing import List, Dict, Optional
from loguru import logger
from .models import CacheEntry
import torch
import time

class VectorStore:
    def __init__(self, collection: Collection, collection_name: str):
        self.collection = collection
        self.collection_name = collection_name

    async def search_similar(self, query_vector: List[float], top_k: int = 5, 
                           field: str = "query_vector", expr: Optional[str] = None) -> List[Dict]:
        """벡터 유사도 검색"""
        try:
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_vector],
                anns_field=field,
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["query_text", "email", "result_text", "metadata", 
                             "created_at", "project_id"]
            )
            
            logger.info(f"Vector search found {len(results[0])} results")
            return [{
                "id": hit.id,
                "score": hit.score,
                "query_text": hit.entity.get("query_text"),
                "email": hit.entity.get("email"),
                "result_text": hit.entity.get("result_text"),
                "metadata": hit.entity.get("metadata"),
                "created_at": hit.entity.get("created_at"),
                "project_id": hit.entity.get("project_id")
            } for hit in results[0]]
            
        except Exception as e:
            logger.error(f"Failed to search similar vectors: {e}")
            return []

    async def keyword_search(self, keyword: str, field: str = "result_text", 
                           limit: int = 10) -> List[Dict]:
        """키워드 기반 검색"""
        try:
            # collection = Collection(self.collection_name)
            # collection.load()
            self.collection.load()
            
            # 키워드 검색을 위한 표현식 생성
            expr = f'{field} like "%{keyword}%"'
            
            results = self.collection.query(
                expr=expr,
                output_fields=["id", "query_text", "email", "result_text", 
                             "metadata", "created_at", "project_id"],
                limit=limit
            )
            
            return [{
                "id": r["id"],
                "query_text": r["query_text"],
                "email": r["email"],
                "result_text": r["result_text"],
                "metadata": r["metadata"],
                "created_at": r["created_at"],
                "project_id": r["project_id"]
            } for r in results]
            
        except Exception as e:
            logger.error(f"Failed to perform keyword search: {e}")
            return []

    async def hybrid_search(self, query_text: str, query_vector: List[float], 
                          top_k: int = 5, expr: Optional[str] = None, keywords: List[str] = []) -> List[Dict]:
        """하이브리드 검색 (키워드 + 벡터)"""
        try:
            # collection = Collection(self.collection_name)
            # collection.load()
            self.collection.load()
            
            # 키워드 검색 표현식 생성
            keyword_conditions = []
            for keyword in keywords:  # 상위 3개 키워드만 사용
                # 각 키워드에 대해 query_text와 result_text 모두 검색
                condition = f'(query_text like "%{keyword}%" or result_text like "%{keyword}%")'
                keyword_conditions.append(condition)
            
            # 키워드 조건들을 OR로 결합
            # keyword_expr = " or ".join(keyword_conditions)
            # logger.info(f"Keyword expression: {keyword_expr}")

            # if expr:
            #     final_expr = f"({keyword_expr}) and ({expr})"
            # else:
            #     final_expr = keyword_expr

            # 기존 필터링 조건이 있으면 결합
            final_expr = None
            if keyword_conditions:  # 키워드 조건이 있는 경우만 처리
                keyword_expr = " or ".join(keyword_conditions)
                logger.info(f"Keyword expression: {keyword_expr}")
                
                if expr:
                    final_expr = f"({keyword_expr}) and ({expr})"
                else:
                    final_expr = keyword_expr
            else:  # 키워드 조건이 없는 경우
                final_expr = expr  # 기존 필터 조건만 사용
            
            logger.info(f"Final search expression: {final_expr}")
            
            
            logger.info(f"Performing hybrid search with expression: {final_expr}")
            
            # 벡터 검색 파라미터
            gpu_available = torch.cuda.is_available()
            search_params = {
                # "metric_type": "COSINE",
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_vector],
                anns_field="result_vector",
                param=search_params,
                limit=top_k,
                expr=final_expr,  # 키워드 + 필터 조건
                output_fields=["query_text", "email", "result_text", "metadata", 
                             "created_at", "project_id"]
            )
            
            logger.info(f"Hybrid search found {len(results[0])} results")
            logger.info(f"-------------------------------------------------")
            logger.info(f"Results: {results[0]}")
            logger.info(f"-------------------------------------------------")
            # return [{
            #     "id": hit.id,
            #     "score": hit.score,
            #     "query_text": hit.entity.get("query_text"),
            #     "email": hit.entity.get("email"),
            #     "result_text": hit.entity.get("result_text"),
            #     "metadata": hit.entity.get("metadata"),
            #     "created_at": hit.entity.get("created_at"),
            #     "project_id": hit.entity.get("project_id")
            # } for hit in results[0]]

            # L2 거리에 기반한 필터링
            filtered_results = []
            for hit in results[0]:
                # L2 거리가 특정 임계값보다 작은 결과만 포함
                # 임계값은 실험을 통해 조정 필요
                if hit.score < 1.5:  # L2 거리 임계값
                    filtered_results.append({
                        "id": hit.id,
                        "score": hit.score,
                        "query_text": hit.entity.get("query_text"),
                        "email": hit.entity.get("email"),
                        "result_text": hit.entity.get("result_text"),
                        "metadata": hit.entity.get("metadata"),
                        "created_at": hit.entity.get("created_at"),
                        "project_id": hit.entity.get("project_id")
                    })
            
            logger.info(f"Hybrid search found {len(results[0])} initial results")
            logger.info(f"After filtering: {len(filtered_results)} results")
            logger.info(f"Score range: {[r['score'] for r in filtered_results]}")
            
            return filtered_results[:top_k]  # 최종적으로 요청된 개수만 반환
           
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {e}")
            return []

    async def insert_entry(self, entry: CacheEntry):
        """캐시 엔트리 삽입"""
        try:
            # collection = Collection(self.collection_name)
            # collection.load()
            self.collection.load()
            
            # 삽입 전 컬렉션 상태 확인
            before_count = self.collection.num_entities
            logger.info(f"Collection size before insert: {before_count}")

            # 삽입 전 데이터 검증 로깅
            logger.info(f"-------------------------------------------------")
            logger.info("Inserting entry with data:")
            logger.info(f"Query ID: {entry.query_id}")
            logger.info(f"Query Text: {entry.query_text}")
            logger.info(f"Result: {entry.result_text}")  # result 값 로깅
            logger.info(f"Email: {entry.email}")
            logger.info(f"Cache Management: {entry.cache_management}")
            logger.info(f"-------------------------------------------------")
                
            # 메타데이터 준비
            metadata = {
                "references": entry.references,
                "pdf_names": entry.pdf_names,
                "cited_refs": entry.cited_refs,
                "pdf_info": entry.pdf_info,
                "hit_count": entry.hit_count,
                "relevance_score": entry.relevance_score,
                "cache_management": entry.cache_management
            }
            
            data = [
                [entry.query_id],
                [entry.query_vector],
                [entry.query_text],
                [entry.result_vector],  # 결과 임베딩 추가
                [entry.result_text],
                [entry.email],
                [metadata],
                [entry.created_at.isoformat()],
                [entry.project_id]
            ]
            
            try:
                # 데이터 삽입
                insert_result = self.collection.insert(data)
                logger.info(f"Raw insert result: {insert_result}")
                
                # 즉시 flush
                self.collection.flush()
                
                # 삽입 후 컬렉션 상태 확인
                after_count = self.collection.num_entities
                logger.info(f"Collection size after insert: {after_count}")
                
                # 삽입된 데이터 확인
                verify_result = self.collection.query(
                    expr=f'id == "{entry.query_id}"',
                    output_fields=["id", "query_text", "email"],
                    limit=1
                )
                logger.info(f"Verification query result: {verify_result}")
                
                # 전체 데이터 샘플 확인
                sample_data = self.collection.query(
                    expr="",
                    output_fields=["id", "query_text", "email"],
                    limit=5
                )
                logger.info(f"Sample data in collection: {sample_data}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to insert data: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Failed to insert cache entry: {e}")
            raise

    def get_collection(self) -> Collection:
        """Milvus 컬렉션 가져오기"""
        try:
            # collection = Collection(self.collection_name)
            # 컬렉션 로드
            try:
                self.collection.load()
                # 컬렉션 정보 로깅
                num_entities = self.collection.num_entities
                logger.info(f"Collection {self.collection_name} loaded successfully. "
                           f"Number of entities: {num_entities}")
            except Exception as load_error:
                logger.warning(f"Collection load warning (may be already loaded): {load_error}")
            
            return self.collection
        except Exception as e:
            logger.error(f"Failed to get collection: {e}")
            raise

    async def get_entry_count(self) -> int:
        """전체 엔트리 수 조회"""
        try:
            return self.collection.num_entities
        except Exception as e:
            logger.error(f"Failed to get entry count: {e}")
            return 0

    async def check_entry_exists(self, query_id: str) -> bool:
        """특정 엔트리 존재 여부 확인"""
        try:
            # collection = self.get_collection()
            result = self.collection.query(
                expr=f'id == "{query_id}"',
                output_fields=["id"],
                limit=1
            )
            return len(result) > 0
        except Exception as e:
            logger.error(f"Failed to check entry existence: {e}")
            return False

    async def get_all_entries(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """모든 엔트리 조회"""
        try:
            # collection = Collection(self.collection_name)
            # collection.load()
            self.collection.load()
            
            # 컬렉션 상태 확인
            count = self.collection.num_entities
            logger.info(f"Total entities in collection: {count}")
            
            if count == 0:
                logger.warning("Collection is empty")
                return []
            
            # result를 result_text로 변경
            results = self.collection.query(
                expr="",
                output_fields=["id", "query_text", "email", "result_text", "metadata", 
                             "created_at", "project_id"],
                limit=limit,
                offset=offset,
                sort_fields=["created_at"],  # 정렬 필드
                sort_orders=["DESC"]         # 내림차순
            )
            
            logger.info(f"Query returned {len(results)} results")
            if results:
                logger.info("Sample of first result:")
                logger.info(f"ID: {results[0].get('id')}")
                logger.info(f"Query Text: {results[0].get('query_text')}")
                logger.info(f"Result Text: {results[0].get('result_text')}")  # 필드명 변경
                logger.info(f"Email: {results[0].get('email')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get all entries: {e}")
            return []

    async def drop_collection(self) -> bool:
        """컬렉션 완전 삭제"""
        try:
            if utility.has_collection(self.collection_name):
                # 컬렉션이 존재하면 삭제
                utility.drop_collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} has been dropped successfully")
                return True
            else:
                logger.warning(f"Collection {self.collection_name} does not exist")
                return False
        except Exception as e:
            logger.error(f"Failed to drop collection: {e}")
            raise