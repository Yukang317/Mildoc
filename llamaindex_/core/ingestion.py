# ==============================
# 模块: 文档摄取管道
# 所属文件: core/ingestion.py
# ==============================
"""
文档摄取管道 - 负责文档的解析、切分、向量化和索引构建

核心职责：
1. 多格式文档解析（PDF、TXT、Markdown等）
2. 智能分块（根据文档类型选择不同策略）
3. 向量化（使用HuggingFace嵌入模型）
4. 索引构建（Chroma向量库 + Redis缓存）
5. 增量更新支持（基于Redis缓存去重）

技术架构：
┌─────────────────────────────────────────────────────────────┐
│                  DocumentIngestionPipeline                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ PDF解析器       │  │ 文本解析器      │                 │
│  │ (转Markdown)    │  │ (SimpleReader)  │                 │
│  └────────┬────────┘  └────────┬────────┘                 │
│           │                     │                          │
│           ▼                     ▼                          │
│  ┌──────────────────────────────────────────┐              │
│  │           IngestionPipeline              │              │
│  │  ┌─────────────┐  ┌─────────────┐       │              │
│  │  │ NodeParser  │→│ EmbedModel  │→│      │              │
│  │  │ (分块器)    │  │ (向量化)    │      │              │
│  │  └─────────────┘  └─────────────┘       │              │
│  └─────────────────────┬───────────────────┘              │
│                        │                                  │
│           ┌────────────┼────────────┐                     │
│           ▼            ▼            ▼                     │
│  ┌─────────────┐ ┌───────────┐ ┌─────────────┐            │
│  │ Chroma      │ │  Redis    │ │  Redis      │            │
│  │ VectorStore │ │ DocStore  │ │ IndexStore  │            │
│  └─────────────┘ └───────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
"""

# ==============================
# 导入依赖
# ==============================
# 类型提示
from typing import List, Optional, Tuple
# 文件系统操作
from pathlib import Path
import os
# PyTorch（用于判断CUDA是否可用）
import torch
# LlamaIndex核心组件
from llama_index.core import (
    VectorStoreIndex,      # 向量索引核心类
    StorageContext,        # 存储上下文（管理多种存储）
    Settings,              # 全局设置（LLM、嵌入模型等）
    SimpleDirectoryReader, # 简单目录读取器（支持多种文本格式）
    load_index_from_storage # 从存储加载索引
)
# 节点解析器（分块策略）
from llama_index.core.node_parser import (
    SentenceSplitter,        # 句子级分块器（通用文本）
    MarkdownNodeParser,      # Markdown专用分块器
    MarkdownElementNodeParser # Markdown元素级分块器
)
# 提取器
from llama_index.core.extractors import TitleExtractor  # 标题提取器
# 摄取管道
from llama_index.core.ingestion import (
    IngestionPipeline,    # 摄取管道核心类
    IngestionCache,       # 摄取缓存（增量更新去重）
    DocstoreStrategy      # 文档存储策略
)
# 向量存储
from llama_index.vector_stores.chroma import ChromaVectorStore  # Chroma向量库
# Redis存储（文档、索引、缓存）
from llama_index.storage.docstore.redis import RedisDocumentStore  # Redis文档存储
from llama_index.storage.index_store.redis import RedisIndexStore  # Redis索引存储
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache  # Redis缓存
# Chroma客户端
import chromadb
# 嵌入模型
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # HuggingFace嵌入
# LLM模型
from llama_index.llms.dashscope import DashScope  # 阿里云DashScope
# 配置和工具
from config.settings import Settings as AppSettings  # 应用配置
from utils.logger import setup_logger                 # 日志工具
# PDF处理器（自定义）
from core.pdf_parser import MultimodalPDFProcessor   # 多模态PDF解析器

# ==============================
# 初始化组件
# ==============================
# 创建日志记录器
logger = setup_logger(__name__)


# ==============================
# 文档摄取管道类
# ==============================
class DocumentIngestionPipeline:
    """
    文档摄取管道 - 负责文档解析、切分、向量化和索引构建的完整流程
    
    核心属性：
    - index: 向量索引（VectorStoreIndex）
    - chroma_vector_store: Chroma向量存储
    - text_pipeline: 普通文本摄取管道
    - markdown_pipeline: Markdown专用摄取管道
    - pdf_processor: PDF多模态处理器
    - storage_context: 存储上下文（统一管理各类存储）
    """

    def __init__(self):
        # ------------------------------
        # 步骤1: 设置模型（LLM和嵌入模型）
        # ------------------------------
        self._setup_models()
        
        # ------------------------------
        # 初始化核心组件
        # ------------------------------
        self.index: Optional[VectorStoreIndex] = None           # 向量索引
        self.chroma_vector_store: Optional[ChromaVectorStore] = None  # Chroma向量存储
        self.text_pipeline: Optional[IngestionPipeline] = None  # 文本文件管道
        self.markdown_pipeline: Optional[IngestionPipeline] = None  # Markdown管道

        # ------------------------------
        # 步骤2: 初始化存储组件
        # ------------------------------
        # Redis: 存储文档和索引元数据
        # Chroma: 存储向量
        self._initialize_storage_components()

        # ------------------------------
        # 步骤3: 创建摄取管道
        # ------------------------------
        # 根据文档类型创建不同的管道：
        # - text_pipeline: 普通文本（TXT等）
        # - markdown_pipeline: Markdown格式（含PDF转换后）
        self._create_pipelines()

        # ------------------------------
        # 步骤4: 初始化PDF处理器
        # ------------------------------
        # 负责将PDF转换为Markdown格式
        self.pdf_processor = MultimodalPDFProcessor()

        # ------------------------------
        # 步骤5: 创建存储上下文
        # ------------------------------
        # 统一管理向量存储、文档存储、索引存储
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.chroma_vector_store,   # 向量存储
            docstore=self.redis_document_store,      # 文档存储
            index_store=self.redis_index_store       # 索引存储
        )

    # ==============================
    # 模型设置
    # ==============================
    def _setup_models(self):
        """
        设置全局模型配置（LLM和嵌入模型）
        
        LlamaIndex的Settings是全局配置，设置后所有组件共享
        """
        # 设置LLM（阿里云DashScope）
        Settings.llm = DashScope(
            api_key=AppSettings.API_KEY,        # API密钥
            api_base=AppSettings.API_BASE_URL,  # API地址
            model_name=AppSettings.MODEL,       # 模型名称
            temperature=AppSettings.TEMPERATURE # 生成温度
        )
        
        # 设置嵌入模型（HuggingFace本地模型）
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=AppSettings.EMBEDDING_MODEL_PATH,  # 模型路径
            # 自动选择设备：有GPU用GPU，否则用CPU
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # ==============================
    # 存储组件初始化
    # ==============================
    def _initialize_storage_components(self):
        """
        初始化所有存储组件
        
        存储架构：
        1. RedisIndexStore: 存储索引元数据（索引ID、配置等）
        2. RedisDocumentStore: 存储文档节点（分块后的小文档）
        3. ChromaVectorStore: 存储向量嵌入
        """
        # 1. 初始化索引存储（Redis）
        self.redis_index_store = RedisIndexStore.from_host_and_port(
            host="127.0.0.1", port=6379, namespace="redis_index"
        )
        
        # 2. 初始化文档存储（Redis）
        self.redis_document_store = RedisDocumentStore.from_host_and_port(
            host="127.0.0.1", port=6379, namespace="redis_docs"
        )
        
        # 3. 初始化向量存储（Chroma）
        self._create_chroma_db()

    def _create_chroma_db(self):
        """
        创建Chroma向量数据库
        
        持久化策略：
        - 使用PersistentClient将向量数据持久化到磁盘
        - 集合名称固定为"quickstart"
        - 数据存储在配置指定的目录（AppSettings.CHROMA_PERSIST_DIR）
        """
        # 创建持久化客户端（数据存储到磁盘）
        chroma_client = chromadb.PersistentClient(AppSettings.CHROMA_PERSIST_DIR)
        
        # 获取或创建集合（不存在则创建）
        chroma_collection = chroma_client.get_or_create_collection("quickstart")
        
        # 创建Chroma向量存储包装器
        self.chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # ==============================
    # 摄取管道创建
    # ==============================
    def _create_pipelines(self):
        """
        创建不同类型的文档摄取管道
        
        设计思路：
        - 根据文档类型创建专门的处理管道
        - 普通文本使用 SentenceSplitter
        - Markdown使用 MarkdownNodeParser + SentenceSplitter 组合
        
        管道处理流程：
        1. NodeParser: 将文档切分为节点
        2. TitleExtractor: 提取标题作为元数据
        3. EmbedModel: 生成向量嵌入
        """
        # ------------------------------
        # 通用配置（两个管道共享）
        # ------------------------------
        common_config = {
            "vector_store": self.chroma_vector_store,  # 向量存储
            "docstore": self.redis_document_store,      # 文档存储
            "cache": IngestionCache(
                # 使用Redis作为缓存，支持增量更新去重
                cache=RedisCache.from_host_and_port("localhost", 6379),
                collection="redis_cache",
            ),
            "docstore_strategy": DocstoreStrategy.UPSERTS_AND_DELETE  # 更新策略
        }

        # ------------------------------
        # 管道1: 普通文本文件处理
        # ------------------------------
        # 使用 SentenceSplitter（句子级切分）
        self.text_pipeline = IngestionPipeline(
            transformations=[
                # 步骤1: 句子级切分
                SentenceSplitter(
                    chunk_size=AppSettings.CHUNK_SIZE,      # 块大小
                    chunk_overlap=AppSettings.CHUNK_OVERLAP  # 块重叠
                ),
                # 步骤2: 提取标题
                TitleExtractor(nodes=AppSettings.TITLE_EXTRACTOR_NODES),
                # 步骤3: 向量化
                Settings.embed_model,
            ],
            **common_config
        )

        # ------------------------------
        # 管道2: Markdown文件处理（含PDF转换后）
        # ------------------------------
        # 使用 MarkdownNodeParser + SentenceSplitter 组合
        self.markdown_pipeline = IngestionPipeline(
            transformations=[
                # 步骤1: Markdown专用解析
                MarkdownNodeParser(
                    include_metadata=True,           # 保留元数据
                    include_prev_next_rel=True,      # 保留节点关系（上下文理解）
                ),
                # 步骤2: 段落级切分（避免在图片标签中间切分）
                SentenceSplitter(
                    chunk_size=AppSettings.CHUNK_SIZE,
                    chunk_overlap=AppSettings.CHUNK_OVERLAP,
                    separator="\n\n",              # 使用空行作为分隔符
                    paragraph_separator="\n\n"      # 段落级别切分
                ),
                # 步骤3: 提取标题
                TitleExtractor(nodes=AppSettings.TITLE_EXTRACTOR_NODES),
                # 步骤4: 向量化
                Settings.embed_model,
            ],
            **common_config
        )

    # ==============================
    # 模型配置更新
    # ==============================
    def update_model_config(self, model_name: str, temperature: float, max_tokens: int):
        """
        动态更新LLM模型配置
        
        Args:
            model_name: 模型名称（如 "qwen-plus"）
            temperature: 生成温度（0-1）
            max_tokens: 最大生成token数
        """
        Settings.llm = DashScope(
            api_key=AppSettings.API_KEY,
            api_base=AppSettings.API_BASE_URL,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        logger.info(f"🔄 模型已更新: model={model_name}, temperature={temperature}")

    # ==============================
    # 核心方法：文档摄取
    # ==============================
    def ingest_documents(self, file_paths: List[str]) -> Tuple:
        """
        摄取文档并创建/更新向量索引
        
        完整处理流程：
        1. 文件解析：根据文件类型选择解析器（PDF→Markdown，TXT→文本）
        2. 文档标准化：统一文档ID和元数据（支持增量更新去重）
        3. 智能分块：根据文档类型选择分块策略
        4. 向量化：生成向量嵌入
        5. 索引构建/更新：创建新索引或增量更新
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            Tuple: (status, result, nodes)
                   - status: "success" 或 "error"
                   - result: 结果描述
                   - nodes: 生成的节点列表（用于后续工作流更新）
        """
        try:
            documents = []
            
            # ------------------------------
            # Step 1: 文件解析
            # ------------------------------
            for file_path in file_paths:
                # 检查文件是否存在
                if not Path(file_path).exists():
                    logger.warning(f"文件不存在: {file_path}")
                    continue
                
                # 判断文件类型
                if Path(file_path).suffix.lower() == ".pdf":
                    # PDF文件：转换为Markdown
                    document = self.pdf_processor.convert_pdf_to_markdown_doc(file_path)
                    documents.append(document)
                else:
                    # 非PDF文件：直接读取
                    # 1. 提取文件名（作为唯一标识）
                    file_name = os.path.basename(file_path)
                    
                    # 2. 构造稳定的逻辑ID（支持增量更新）
                    # 关键设计：不管文件在哪个临时目录，ID始终不变
                    stable_id = f"knowledge_base/{file_name}"
                    
                    # 3. 读取文件内容
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    docs = reader.load_data()
                    
                    # 4. 标准化文档元数据（关键步骤）
                    for doc in docs:
                        # 强制覆盖ID为稳定ID
                        doc.id_ = stable_id

                        # === 必须操作：清除易变元数据 ===
                        # 临时文件的创建时间肯定是刚生成的，必须删掉，否则 Hash 会变
                        doc.metadata.pop("file_path", None)
                        doc.metadata.pop("last_modified_date", None)

                        # 我们可以把真实的原始文件名存进去方便以后展示
                        doc.metadata["file_name"] = file_name

                        # 如果你想让溯源更清晰，可以把 ID 也存一份在 metadata
                        doc.metadata["doc_id"] = stable_id
                    
                    documents.extend(docs)
                    logger.info(f"已处理文档: {file_name} (ID: {stable_id})")
                
                logger.info(f"已读取文档: {file_path}")

            # 检查是否有有效文档
            if not documents:
                return "error", "没有找到有效的文档", []

            # ------------------------------
            # Step 2: 按文档类型分类
            # ------------------------------
            logger.info("开始处理文档...")
            markdown_docs = []
            text_docs = []

            for doc in documents:
                if doc.metadata.get("content_type") == "markdown":
                    markdown_docs.append(doc)
                else:
                    text_docs.append(doc)

            # ------------------------------
            # Step 3: 执行摄取管道
            # ------------------------------
            pipeline_nodes = []

            # 处理Markdown文档（含PDF转换后的）
            if markdown_docs:
                logger.info(f"处理 {len(markdown_docs)} 个 Markdown 文档...")
                md_nodes = self.markdown_pipeline.run(
                    documents=markdown_docs, 
                    show_progress=True
                )
                pipeline_nodes.extend(md_nodes)
                logger.info(f"Markdown 文档处理完成，生成了 {len(md_nodes)} 个节点")

            # 处理普通文本文档
            if text_docs:
                logger.info(f"处理 {len(text_docs)} 个普通文本文档...")
                txt_nodes = self.text_pipeline.run(
                    documents=text_docs, 
                    show_progress=True
                )
                pipeline_nodes.extend(txt_nodes)
                logger.info(f"普通文本文档处理完成，生成了 {len(txt_nodes)} 个节点")

            logger.info(f"所有文档处理完成，共生成了 {len(pipeline_nodes)} 个节点")

            # ------------------------------
            # Step 4: 增量更新检查
            # ------------------------------
            # 如果所有文档都是重复的（缓存命中），pipeline_nodes将为空
            if not pipeline_nodes:
                logger.info("所有文档都是重复的，无需更新索引")
                result = "所有文档都是重复的，无需处理"
                return "success", result, pipeline_nodes

            # ------------------------------
            # Step 5: 存储节点并更新索引
            # ------------------------------
            # 保存切分后的节点到文档存储
            self.storage_context.docstore.add_documents(pipeline_nodes)

            # 创建或更新向量索引
            logger.info("创建/更新文档索引...")
            if not self.index:
                # 首次创建索引
                logger.info("首次创建索引对象")
                self.index = VectorStoreIndex(
                    pipeline_nodes, 
                    storage_context=self.storage_context,
                    embed_model=Settings.embed_model
                )
            else:
                # 增量更新索引
                logger.info("增量更新索引")
                self.index.insert_nodes(pipeline_nodes)

            # 调试信息
            if self.index:
                print(f"索引后 docstore 文档数: {len(self.index.docstore.docs)}")
                print(f"索引 ID: {self.index.index_id}")

            # 返回结果
            result = f"成功摄取了 {len(file_paths)} 个文档，生成了 {len(pipeline_nodes)} 个节点"
            logger.info(result)
            return "success", result, pipeline_nodes

        except Exception as e:
            error_msg = f"文档摄取失败: {str(e)}"
            logger.error(error_msg)
            return "error", error_msg, []

    # ==============================
    # 索引加载
    # ==============================
    def get_documents(self):
        """
        加载已存在的索引和文档（服务启动时调用）
        
        加载策略：
        1. 首选：从Redis索引存储加载（完整索引结构）
        2. 降级：从Chroma向量存储直接加载（仅向量数据）
        
        Returns:
            VectorStoreIndex 或 None（无索引时）
        """
        logger.info("读取已有向量数据库中的文档和索引...")
        try:
            # ------------------------------
            # 方案A: 从索引存储加载（首选）
            # ------------------------------
            index_store = self.storage_context.index_store
            index_ids = index_store.list_indexes()

            # 检查是否有索引
            if not index_ids:
                logger.info("未找到任何已有索引，将创建新索引")
                return None

            # 处理多个索引的情况
            if len(index_ids) > 1:
                logger.warning(f"发现 {len(index_ids)} 个索引，将使用第一个: {index_ids[0]}")

            # 加载第一个索引
            index_id = index_ids[0]
            self.index = load_index_from_storage(
                self.storage_context,
                index_id=index_id
            )
            logger.info(f"成功加载索引: {index_id}")
            print("index:", self.index)
            return self.index

        except Exception as e:
            # ------------------------------
            # 方案B: 降级加载（从向量存储直接加载）
            # ------------------------------
            logger.error(f"加载索引失败: {e}")
            try:
                logger.info("尝试降级方案：从向量存储直接加载...")
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.chroma_vector_store,
                    embed_model=Settings.embed_model,
                    storage_context=self.storage_context
                )
                print("index (fallback):", self.index)
                return self.index
            except Exception as fallback_error:
                logger.error(f"降级加载也失败: {fallback_error}")
                return None
