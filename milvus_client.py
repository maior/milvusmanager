from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from loguru import logger
import torch
from typing import Optional
# from django.conf import settings
# from logos_server.conf.config import Config

class MilvusClientSingleton:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.collection_name = "logos_cache"
            self.dim = 768  # embedding dimension
            
            # Milvus 연결 설정
            # self.host = Config.get("MILVUS_HOST")
            # self.port = Config.get("MILVUS_PORT")

            self.host = "localhost"
            self.port = "19530"
            
            # 초기 연결
            self._connect()
            self._init_collection()
            
            MilvusClientSingleton._initialized = True
    
    def get_collection_name(self) -> str:
        """컬렉션 이름 가져오기"""
        return self.collection_name
    
    def _connect(self):
        """Milvus 서버에 연결"""
        try:
            # 기존 연결 해제
            try:
                connections.disconnect("default")
            except:
                pass
            
            # gRPC 옵션 설정
            grpc_options = [
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.keepalive_permit_without_calls', 1),  # 추가 옵션
                ('grpc.max_receive_message_length', 100*1024*1024)  # 100MB
            ]
            
            # 새로운 연결
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                timeout=10,
                grpc_options=grpc_options
            )
            
            # 연결 확인
            try:
                utility.list_collections()
                logger.info(f"Connected to Milvus server at {self.host}:{self.port}")
            except Exception as e:
                raise ConnectionError(f"Failed to establish Milvus connection: {e}")
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise
    
    def _init_collection(self):
        """컬렉션 초기화"""
        try:
            # 필드 정의
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="query_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="query_text", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="result_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="result_text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=30),
                FieldSchema(name="project_id", dtype=DataType.VARCHAR, max_length=100)
            ]
            
            schema = CollectionSchema(fields)
            
            # 컬렉션이 없으면 생성
            if not utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name, schema)
                
                # 인덱스 생성
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                
                # 쿼리 벡터 인덱스 생성
                collection.create_index(field_name="query_vector", index_params=index_params)
                # 결과 벡터 인덱스 생성
                collection.create_index(field_name="result_vector", index_params=index_params)
                
                # GPU 사용 가능한 경우 GPU로 로드
                if torch.cuda.is_available():
                    collection.load(replica_number=1, using=['gpu0'])
                    logger.info("Collection loaded to GPU")
                else:
                    collection.load()
                    logger.info("Collection loaded to CPU")
                
                logger.info(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def get_collection(self) -> Collection:
        """컬렉션 가져오기"""
        try:
            logger.debug(f'self.collection_name : {self.collection_name}')
            collection = Collection(name=self.collection_name)
            logger.debug(f'collection : {collection}')
            collection.load()
            logger.debug(f'collection.load() done')
            return collection
        except Exception as e:
            logger.error(f"Failed to get collection: {e}")
            raise
    
    @classmethod
    def get_instance(cls) -> 'MilvusClientSingleton':
        """싱글톤 인스턴스 가져오기"""
        if cls._instance is None:
            cls._instance = MilvusClientSingleton()
        return cls._instance
    
    def __del__(self):
        """소멸자에서 연결 해제"""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus server")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}") 