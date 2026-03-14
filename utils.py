from typing import Dict, Optional, Union
from datetime import datetime, timedelta
from .models import CacheEntry
import json
from loguru import logger

def should_update_cache(cache_entry: Union[dict, CacheEntry]) -> bool:
    """캐시 업데이트 필요 여부 확인"""
    try:
        # CacheEntry 객체인 경우 딕셔너리로 변환
        entry_dict = cache_entry.dict() if hasattr(cache_entry, 'dict') else cache_entry
        
        # created_at 처리
        created_at = entry_dict.get('created_at')
        if not created_at:
            logger.warning("No created_at found in cache entry")
            return True
            
        # datetime 객체로 변환
        if isinstance(created_at, str):
            entry_date = datetime.fromisoformat(created_at)
        elif isinstance(created_at, datetime):
            entry_date = created_at
        else:
            logger.error(f"Unexpected created_at type: {type(created_at)}")
            return True

        # 현재 시간과 비교
        current_time = datetime.now()
        time_difference = current_time - entry_date
        
        # 값 추출
        hit_count = entry_dict.get('hit_count', 0)
        relevance_score = entry_dict.get('relevance_score', 0.0)

        # 업데이트 조건
        conditions = [
            # time_difference > timedelta(days=7),  # 7일 이상 경과
            # hit_count < 3,  # 조회수 3회 미만
            relevance_score < 0.6  # 관련성 점수 0.7 미만
        ]

        # 디버그 로깅
        logger.warning('----------------------------------------------')
        logger.warning(f"Cache entry data:")
        logger.warning(f"- Created at: {created_at}")
        logger.warning(f"- Age: {time_difference.days} days")
        logger.warning(f"- Hit count: {hit_count}")
        logger.warning(f"- Relevance score: {relevance_score}")
        logger.warning(f"- Update needed: {any(conditions)}")
        logger.warning('----------------------------------------------')

        return any(conditions)

    except Exception as e:
        logger.error(f"Error checking cache update need: {e}")
        logger.error(f"Cache entry: {cache_entry}")
        return True  # 에러 발생 시 업데이트 필요로 판단

def format_cache_response(cache_entry: dict) -> dict:
    """캐시 응답 포맷팅"""
    try:
        return {
            "result": cache_entry.get('result', ''),
            "references": cache_entry.get('references', []),
            "pdf_names": cache_entry.get('pdf_names', []),
            "cited_refs": cache_entry.get('cited_refs', []),
            "pdf_info": cache_entry.get('pdf_info', {}),
            "from_cache": True
        }
    except Exception as e:
        logger.error(f"Error formatting cache response: {e}")
        return {
            "result": "",
            "references": [],
            "pdf_names": [],
            "cited_refs": [],
            "pdf_info": {},
            "from_cache": False
        } 