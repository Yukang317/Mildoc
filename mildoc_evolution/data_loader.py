"""数据加载模块 - 支持从OSS和本地加载测试数据集"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from config import settings

try:
    import oss2
    OSS_AVAILABLE = True
except ImportError:
    OSS_AVAILABLE = False


class ChunkInfo(BaseModel):
    """文本块信息 - 用于Chunk级别评估"""
    chunk_id: str  # Chunk的唯一标识
    content: str   # Chunk的文本内容
    doc_name: str  # 所属文档名称
    metadata: Optional[Dict[str, Any]] = None  # 额外元数据


class TestSample(BaseModel):
    """测试样本数据结构 - 支持Chunk级别标注"""
    question: str
    ground_truth: Optional[str] = None            # 标准答案
    
    # 向后兼容：文档级标注（可选）
    expected_docs: Optional[List[str]] = None     # 期望文档列表
    expected_doc_source: Optional[str] = None     # 期望文档来源
    
    # ✨ 新增：Chunk级别标注（推荐）
    expected_chunks: Optional[List[ChunkInfo]] = None  # 期望的相关Chunk列表
    
    # RAGAS格式字段
    contexts: Optional[List[str]] = None          # 相关上下文
    
    metadata: Optional[Dict[str, Any]] = None


class TestDataset(BaseModel):
    """测试数据集结构"""
    name: str
    description: str
    version: str
    samples: List[TestSample]
    created_at: Optional[str] = None
    source: Optional[str] = None


class DataLoader:
    """数据加载器 - 支持从OSS和本地加载测试数据"""

    def __init__(self):
        self.oss_client = None
        if OSS_AVAILABLE and settings.validate_oss_config():
            self._init_oss_client()

    def _init_oss_client(self):
        """初始化OSS客户端"""
        try:
            auth = oss2.Auth(settings.OSS_ACCESS_KEY_ID, settings.OSS_ACCESS_KEY_SECRET)
            self.oss_client = oss2.Bucket(auth, settings.OSS_ENDPOINT, settings.OSS_BUCKET_NAME)
            print(f"OSS客户端初始化成功，Bucket: {settings.OSS_BUCKET_NAME}")
        except Exception as e:
            print(f"OSS客户端初始化失败: {e}")
            self.oss_client = None

    def load_from_oss(self, remote_path: str) -> Optional[TestDataset]:
        """从OSS加载测试数据集"""
        if not self.oss_client:
            print("OSS客户端未初始化，请检查配置")
            return None

        try:
            # 构造完整路径
            full_path = f"{settings.OSS_TEST_DATA_PREFIX}{remote_path}"
            print(f"从OSS加载测试数据: {full_path}")

            # 下载文件内容
            result = self.oss_client.get_object(full_path)
            content = result.read().decode('utf-8')

            # 解析JSON
            data = json.loads(content)
            return TestDataset(**data)

        except oss2.exceptions.NoSuchKey:
            print(f"OSS文件不存在: {remote_path}")
            return None
        except Exception as e:
            print(f"从OSS加载数据失败: {e}")
            return None

    def load_from_local(self, local_path: str) -> Optional[TestDataset]:
        """从本地文件加载测试数据集"""
        try:
            file_path = Path(local_path)
            if not file_path.exists():
                # 尝试从默认目录查找
                file_path = settings.LOCAL_TEST_DATA_DIR / local_path
                if not file_path.exists():
                    print(f"本地文件不存在: {local_path}")
                    return None

            print(f"从本地加载测试数据: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return TestDataset(**data)

        except Exception as e:
            print(f"从本地加载数据失败: {e}")
            return None

    def list_oss_datasets(self) -> List[str]:
        """列出OSS上所有可用的测试数据集"""
        if not self.oss_client:
            return []

        try:
            datasets = []
            for obj in oss2.ObjectIterator(self.oss_client, prefix=settings.OSS_TEST_DATA_PREFIX):
                if obj.key.endswith('.json'):
                    # 提取文件名
                    filename = obj.key.replace(settings.OSS_TEST_DATA_PREFIX, '')
                    datasets.append(filename)
            return datasets
        except Exception as e:
            print(f"列出OSS数据集失败: {e}")
            return []

    def list_local_datasets(self) -> List[str]:
        """列出本地所有可用的测试数据集"""
        try:
            datasets = []
            for file in settings.LOCAL_TEST_DATA_DIR.glob('*.json'):
                datasets.append(file.name)
            return datasets
        except Exception as e:
            print(f"列出本地数据集失败: {e}")
            return []

    def save_to_local(self, dataset: TestDataset, filename: str):
        """将数据集保存到本地"""
        try:
            file_path = settings.LOCAL_TEST_DATA_DIR / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dataset.dict(), f, ensure_ascii=False, indent=2)
            print(f"数据集已保存到本地: {file_path}")
        except Exception as e:
            print(f"保存数据集失败: {e}")

    def upload_to_oss(self, local_path: str, remote_filename: str):
        """将本地数据集上传到OSS"""
        if not self.oss_client:
            print("OSS客户端未初始化")
            return

        try:
            file_path = Path(local_path)
            if not file_path.exists():
                print(f"本地文件不存在: {local_path}")
                return

            remote_path = f"{settings.OSS_TEST_DATA_PREFIX}{remote_filename}"
            self.oss_client.put_object_from_file(remote_path, str(file_path))
            print(f"数据集已上传到OSS: {remote_path}")
        except Exception as e:
            print(f"上传数据集失败: {e}")


def create_sample_dataset() -> TestDataset:
    """创建示例测试数据集（用于演示）"""
    samples = [
        TestSample(
            question="什么是向量数据库？",
            expected_answers=[
                "向量数据库是一种专门用于存储、索引和查询向量数据的数据库系统",
                "向量数据库支持高效的相似性搜索"
            ],
            expected_docs=["vector_database_intro.md"],
            metadata={"category": "概念"}
        ),
        TestSample(
            question="Milvus有哪些特点？",
            expected_answers=[
                "Milvus是一个开源的向量数据库",
                "支持多种索引类型",
                "支持分布式部署"
            ],
            expected_docs=["milvus_features.md"],
            metadata={"category": "产品"}
        ),
        TestSample(
            question="如何使用RAG系统？",
            expected_answers=[
                "首先上传文档到系统",
                "系统会自动进行向量化处理",
                "然后就可以提问了"
            ],
            expected_docs=["rag_usage.md"],
            metadata={"category": "使用"}
        )
    ]

    return TestDataset(
        name="RAG系统基础测试集",
        description="用于评估RAG系统性能的基础测试数据集",
        version="1.0",
        created_at="2024-01-01",
        source="manual",
        samples=samples
    )


if __name__ == "__main__":
    # 演示如何使用数据加载器
    loader = DataLoader()

    # 列出可用数据集
    print("=== 本地数据集 ===")
    print(loader.list_local_datasets())

    print("\n=== OSS数据集 ===")
    print(loader.list_oss_datasets())

    # 创建示例数据集并保存
    sample_dataset = create_sample_dataset()
    loader.save_to_local(sample_dataset, "sample_test_data.json")
    print("\n示例数据集已创建")
