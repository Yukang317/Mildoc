"""配置管理模块 - 支持RAGAS自动化评估框架"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings:
    """评估模块配置类"""
    
    # ========== OSS配置 ==========
    OSS_ACCESS_KEY_ID: str = os.getenv("OSS_ACCESS_KEY_ID", "")
    OSS_ACCESS_KEY_SECRET: str = os.getenv("OSS_ACCESS_KEY_SECRET", "")
    OSS_ENDPOINT: str = os.getenv("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")
    OSS_BUCKET_NAME: str = os.getenv("OSS_BUCKET_NAME", "")
    OSS_TEST_DATA_PREFIX: str = os.getenv("OSS_TEST_DATA_PREFIX", "test_data/")
    OSS_DOCUMENTS_PREFIX: str = os.getenv("OSS_DOCUMENTS_PREFIX", "documents/")  # 知识库文档存储路径
    
    # ========== 本地配置 ==========
    LOCAL_TEST_DATA_DIR: Path = Path(__file__).parent / "test_data"
    LOCAL_DOCUMENTS_DIR: Path = Path(__file__).parent / "documents"  # 本地文档目录
    
    # ========== 评估配置 ==========
    TOP_K: int = int(os.getenv("EVAL_TOP_K", "5"))
    SCORE_THRESHOLD: float = float(os.getenv("EVAL_SCORE_THRESHOLD", "0.5"))
    
    # ========== RAG系统配置 ==========
    RAG_API_URL: str = os.getenv("RAG_API_URL", "http://localhost:8000")
    RAG_API_TIMEOUT: int = int(os.getenv("RAG_API_TIMEOUT", "60"))
    
    # Milvus配置（用于Chunk级评估）
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "mildoc_collection")
    
    # ========== RAGAS配置 ==========
    RAGAS_ENABLE: bool = os.getenv("RAGAS_ENABLE", "false").lower() == "true"
    
    # LLM配置（用于RAGAS评估）
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "qwen-max")
    
    # Embedding配置（用于RAGAS测试集生成）
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v3")
    
    # 报告配置
    REPORT_OUTPUT_DIR: Path = Path(__file__).parent / "reports"
    RAGAS_OUTPUT_DIR: Path = Path(__file__).parent / "ragas_output"
    
    # ========== 验证方法 ==========
    @classmethod
    def validate_oss_config(cls) -> bool:
        """验证OSS配置是否完整"""
        return all([
            cls.OSS_ACCESS_KEY_ID,
            cls.OSS_ACCESS_KEY_SECRET,
            cls.OSS_BUCKET_NAME
        ])
    
    @classmethod
    def validate_ragas_config(cls) -> bool:
        """验证RAGAS配置是否完整"""
        required_fields = [
            cls.LLM_API_KEY,
            cls.LLM_BASE_URL,
            cls.LLM_MODEL_NAME,  # ✅ 新增：验证模型名称
            cls.EMBEDDING_API_KEY,
            cls.EMBEDDING_BASE_URL,
            cls.EMBEDDING_MODEL_NAME  # ✅ 新增：验证嵌入模型名称
        ]
        
        if not all(required_fields):
            print("⚠️  RAGAS配置不完整，缺少以下字段:")
            if not cls.LLM_API_KEY:
                print("  - LLM_API_KEY")
            if not cls.LLM_BASE_URL:
                print("  - LLM_BASE_URL")
            if not cls.LLM_MODEL_NAME:
                print("  - LLM_MODEL_NAME")
            if not cls.EMBEDDING_API_KEY:
                print("  - EMBEDDING_API_KEY")
            if not cls.EMBEDDING_BASE_URL:
                print("  - EMBEDDING_BASE_URL")
            if not cls.EMBEDDING_MODEL_NAME:
                print("  - EMBEDDING_MODEL_NAME")
            return False
        
        return True
    
    @classmethod
    def ensure_directories(cls):
        """确保必要目录存在"""
        cls.LOCAL_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.RAGAS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOCAL_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
