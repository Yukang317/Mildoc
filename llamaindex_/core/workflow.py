# ==============================
# 模块导入区
# ==============================
# LlamaIndex 核心模块：向量索引存储
from llama_index.core import VectorStoreIndex

# LlamaIndex 工作流模块：提供工作流编排能力
# Context: 工作流上下文对象，用于跨步骤传递数据
# Workflow: 工作流基类
# step: 装饰器，标记工作流中的一个步骤
# StopEvent/StartEvent: 工作流的起始和终止事件类型
from llama_index.core.workflow import Context, Workflow, step, StopEvent, StartEvent

# 向量检索器：基于向量相似度进行文档检索
from llama_index.core.retrievers import VectorIndexRetriever

# 响应合成器：将检索结果合成为最终回答
from llama_index.core.response_synthesizers import get_response_synthesizer

# 后处理器：对检索结果进行重排序和优化
# SentenceTransformerRerank: 使用SentenceTransformer模型进行语义重排序
# PrevNextNodePostprocessor: 维护节点间的前后关系
from llama_index.core.postprocessor import SentenceTransformerRerank, PrevNextNodePostprocessor

# BM25检索器：基于词频的传统信息检索算法
from llama_index.retrievers.bm25 import BM25Retriever

# 查询融合检索器：支持多种检索器融合（如RRF算法）
from llama_index.core.retrievers import QueryFusionRetriever

# 应用配置：全局配置项（如模型路径、阈值参数等）
from config.settings import Settings as AppSettings

# 自定义事件类型：定义工作流中各步骤传递的数据结构
from core.events import RAGEvents

# 日志工具：统一日志配置
from utils.logger import setup_logger

# 系统模块：文件系统操作
import os

# 初始化日志记录器，使用模块名作为日志标签
logger = setup_logger(__name__)


# ==============================
# RAG工作流核心类
# ==============================
class RAGWorkflow(Workflow):
    """
    RAG工作流核心类：负责协调混合检索与答案生成的完整流程
    
    核心职责：
    1. 初始化并管理检索组件（向量检索 + BM25）
    2. 执行检索重排序（RRF + SentenceTransformer）
    3. 生成最终回答（支持流式/非流式）
    4. 编排工作流步骤，确保数据正确传递
    
    工作流步骤：
    retrieve_step → rerank_step → generate_step → finalize_step
    """

    def __init__(
            self,
            index: VectorStoreIndex,      # 向量索引对象，提供向量检索能力
            pipeline_nodes: list = None,  # 新摄取的文档节点列表（用于构建BM25索引）
            streaming: bool = False,      # 是否启用流式输出模式
            **kwargs                     # 父类Workflow的额外参数
    ):
        # 调用父类构造方法，设置超时为None（不限制执行时间）
        super().__init__(**kwargs, timeout=None)
        
        # 保存向量索引实例
        self.index = index
        
        # 保存新摄取的节点（用于BM25索引构建）
        self.nodes = pipeline_nodes
        
        # 标记当前工作流模式：流式(True) 或 非流式(False)
        self.streaming = streaming
        
        # 初始化所有工作流组件
        self._setup_components()

    def _setup_components(self):
        """
        初始化工作流的核心组件
        
        组件初始化顺序：
        1. BM25检索器（基于词频的传统检索）
        2. 向量检索器（基于语义相似度）
        3. 混合检索器（融合前两者，使用RRF算法）
        4. 重排序器（精排阶段，提升结果质量）
        5. 响应合成器（生成最终回答）
        """

        # ------------------------------
        # 组件1: BM25检索器
        # ------------------------------
        # 基于词频-逆文档频率的传统检索算法
        # 任务：要么构建新的 BM25 索引，要么从硬盘加载已有的。
        self.bm25_retriever = self._initialize_bm25()

        # ------------------------------
        # 组件2: 向量检索器
        # ------------------------------
        # 基于向量相似度的语义检索
        # similarity_top_k: 每次检索返回的候选文档数量，config默认5个
        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=AppSettings.SIMILARITY_TOP_K
        )

        # ------------------------------
        # 组件3: 混合检索器（QueryFusion）
        # ------------------------------
        # 将向量检索与BM25检索结果融合
        if self.bm25_retriever:
            logger.info("启用混合检索: Vector + BM25")
            self.retriever = QueryFusionRetriever(
                # 指定参与融合的检索器列表
                retrievers=[self.vector_retriever, self.bm25_retriever],
                # 融合后的候选数量
                similarity_top_k=AppSettings.SIMILARITY_TOP_K,
                # 查询扩展数量：1表示不扩展，仅用原始查询
                # 大于1时会生成相似问题进行多轮检索
                num_queries=1,
                # 融合算法：reciprocal_rerank（RRF算法）
                # 把向量检索和 BM25 检索各自返回的文档排名取倒数相加，谁在两个排名里都靠前，最终分数就高。
                # RRF公式: score = 1 / (k + rank)，k通常取60-100
                mode="reciprocal_rerank",       # 开关
                # 启用异步模式，提升性能
                use_async=True,
                # 启用详细日志
                verbose=True
            )
        else:
            # BM25不可用时，降级为纯向量检索
            logger.warning("BM25 不可用，降级为纯向量检索")
            self.retriever = self.vector_retriever

        # ------------------------------
        # 组件4: 重排序器（Reranker）
        # ------------------------------
        # 混合检索召回量大但质量参差不齐，需要二次精排
        # 使用SentenceTransformer模型进行语义层面的重排序
        self.reranker = SentenceTransformerRerank(
            # 重排序模型路径（配置文件中指定）
            model=AppSettings.RERANK_MODEL_PATH,
            # 重排序后保留的文档数量
            top_n=AppSettings.RERANK_TOP_K  # 默认3篇
        )

        # ------------------------------
        # 组件5: 响应合成器
        # ------------------------------
        # 根据检索结果生成最终回答
        # streaming参数决定输出模式：
        # - True: 返回StreamingResponse，支持实时流式输出
        # - False: 返回完整Response对象
        self.synthesizer = get_response_synthesizer(
            streaming=self.streaming
        )

    def _initialize_bm25(self):
        """
        初始化 BM25 检索器的工厂方法
        
        三种初始化场景：
        1. 有新文档节点 → 构建新索引并持久化到磁盘
        2. 无新节点但有缓存 → 从磁盘加载已保存的索引
        3. 既无新节点也无缓存 → 返回None（降级为纯向量检索）
        
        BM25算法特点：
        - 基于词频统计，擅长关键词匹配
        - 对中文支持需要设置 language="zh"
        - 索引可持久化，避免重复构建
        """
        try:
            # ------------------------------
            # 场景A: 有新数据传入（文档上传后）
            # ------------------------------
            # 当有新摄取的文档节点时，构建新的BM25索引
            if self.nodes and len(self.nodes) > 0:
                logger.info(f"使用 {len(self.nodes)} 个新节点构建 BM25 索引...")
                
                # 从节点列表构建BM25检索器
                bm25 = BM25Retriever.from_defaults(
                    nodes=self.nodes,           # 文档节点列表
                    similarity_top_k=AppSettings.SIMILARITY_TOP_K,  # 返回数量，默认 5
                    language="zh"               # 中文分词优化
                )
                
                # 持久化索引到磁盘（确保目录存在）
                if not os.path.exists(AppSettings.BM25_PERSIST_DIR):
                    os.makedirs(AppSettings.BM25_PERSIST_DIR)
                bm25.persist(AppSettings.BM25_PERSIST_DIR)
                
                logger.info("BM25 索引已保存")
                return bm25

            # ------------------------------
            # 场景B: 服务重启/直接对话（无新数据）
            # ------------------------------
            # 尝试从磁盘加载已保存的BM25索引
            elif os.path.exists(AppSettings.BM25_PERSIST_DIR):
                logger.info("从磁盘加载现有的 BM25 索引...")
                return BM25Retriever.from_persist_dir(AppSettings.BM25_PERSIST_DIR)

            # ------------------------------
            # 场景C: 无数据无缓存（首次启动且无文档）
            # ------------------------------
            else:
                logger.warning("没有可用的节点来构建 BM25，也未找到本地缓存。")
                return None

        except Exception as e:
            # 捕获任何初始化异常，返回None让系统降级到纯向量检索
            logger.error(f"BM25 初始化失败: {e}")
            return None

    # ==============================
    # 工作流步骤1: 检索阶段
    # ==============================
    @step
    async def retrieve_step(
            self,
            ctx: Context,           # 工作流上下文，用于跨步骤传递数据
            ev: StartEvent          # 起始事件，包含用户查询
    ) -> RAGEvents.RetrievalEvent:
        """
        检索步骤：根据用户查询从索引中获取相关文档节点
        
        核心流程：
        1. 接收用户查询（来自StartEvent）
        2. 调用混合检索器（Vector + BM25，RRF融合）
        3. 返回检索到的文档节点列表
        
        检索器行为：
        - 如果配置了混合检索：同时执行向量检索和BM25，用RRF融合结果
        - 如果仅向量检索：直接查询Chroma向量库
        """
        logger.info(f"开始检索查询: {ev.query}")

        # ------------------------------
        # 执行异步检索
        # ------------------------------
        # 使用aretrieve()异步方法，避免阻塞事件循环
        # 检索器已配置为混合检索（Vector + BM25 + RRF）
        retrieve_nodes = await self.retriever.aretrieve(ev.query)

        # 记录检索结果数量
        logger.info(f"检索获得 {len(retrieve_nodes)} 个节点")

        # ------------------------------
        # 封装检索结果事件
        # ------------------------------
        # 将检索结果传递给下一个步骤（rerank_step）
        return RAGEvents.RetrievalEvent(query=ev.query, nodes=retrieve_nodes)

    # ==============================
    # 工作流步骤2: 重排序阶段
    # ==============================
    @step
    async def rerank_step(
            self,
            ctx: Context,                      # 工作流上下文
            ev: RAGEvents.RetrievalEvent       # 上一步的检索结果事件
    ) -> RAGEvents.RerankEvent:
        """
        重排序步骤：对检索结果进行二次精排
        
        核心目的：
        - 混合检索召回量大（通常20-50个），质量参差不齐
        - 使用SentenceTransformer模型进行语义级别的精排
        - 保留最相关的top_k个文档（通常3-5个）
        
        重排序原理：
        - 输入：检索到的文档节点 + 用户查询
        - 模型：SentenceTransformer（如BAAI/bge-reranker-large）
        - 输出：重新排序后的节点列表（按语义相似度降序）
        """
        logger.info("开始重排序")

        # ------------------------------
        # 执行语义重排序
        # ------------------------------
        # postprocess_nodes方法会：
        # 1. 计算每个节点与查询的语义相似度
        # 2. 根据相似度重新排序
        # 3. 保留top_n个（配置中指定）
        rerank_nodes = self.reranker.postprocess_nodes(ev.nodes, query_str=ev.query)

        # 记录重排序结果数量
        logger.info(f"重排序后保留 {len(rerank_nodes)} 个节点")

        # ------------------------------
        # 封装重排序结果事件
        # ------------------------------
        # 将精排后的节点传递给下一个步骤（generate_step）
        return RAGEvents.RerankEvent(query=ev.query, nodes=rerank_nodes)

    # ==============================
    # 工作流步骤3: 回答生成阶段
    # ==============================
    @step
    async def generate_step(
            self,
            ctx: Context,                    # 工作流上下文
            ev: RAGEvents.RerankEvent        # 上一步的重排序结果事件
    ) -> RAGEvents.ResponseEvent:
        """
        回答生成步骤：根据精排后的文档节点生成最终回答
        
        核心流程：
        1. 接收精排后的文档节点
        2. 调用LLM生成回答（支持流式/非流式）
        3. 返回回答对象
        
        关键设计：
        - 不重复检索：直接使用上游传递的nodes，避免重复查询
        - 模式适配：根据streaming配置返回不同类型的response
        """
        logger.info("开始生成回答")

        # ------------------------------
        # 调用合成器生成回答
        # ------------------------------
        # asynthesize()是异步方法，不会阻塞事件循环。非流式下返回完整答案字符串，在流式下返回一个生成器对象
        # 输入参数：
        # - query: 用户原始查询
        # - nodes: 精排后的文档节点（作为LLM的上下文参考）
        response = await self.synthesizer.asynthesize(
            query=ev.query,
            nodes=ev.nodes
        )

        # ------------------------------
        # 处理不同输出模式
        # ------------------------------
        # 流式模式：返回StreamingResponse对象（包含response_gen生成器）
        # 非流式模式：返回Response对象，需要转为字符串
        response_payload = response if self.streaming else str(response)

        logger.info("回答生成对象创建完成")

        # ------------------------------
        # 封装回答结果事件
        # ------------------------------
        # 将回答和来源节点传递给最后一个步骤（finalize_step）
        return RAGEvents.ResponseEvent(
            query=ev.query,
            nodes=ev.nodes,
            response=response_payload  # 流式传对象，非流式传字符串
        )

    # ==============================
    # 工作流步骤4: 最终化阶段
    # ==============================
    @step
    async def finalize_step(
            self,
            ctx: Context,                      # 工作流上下文
            ev: RAGEvents.ResponseEvent        # 上一步的回答结果事件
    ) -> StopEvent:
        """
        最终化步骤：封装最终结果，结束工作流
        
        核心职责：
        1. 提取文档来源信息（供前端展示引用出处）
        2. 根据模式封装不同的返回结构
        3. 发送StopEvent结束工作流
        
        返回结构设计：
        - 流式模式：包含response_gen生成器，供前端迭代
        - 非流式模式：包含完整的回答字符串
        - 两种模式都包含sources字段，记录来源文档信息
        """
        # ------------------------------
        # 构建来源信息列表
        # ------------------------------
        # 从节点中提取关键信息：
        # - content: 文档片段内容
        # - score: 检索相似度分数
        # - metadata: 文档元数据（文件名、路径等）
        source_info = [
            {
                "content": node.node.text,       # 文档片段文本
                "score": node.score,             # 相似度分数（0-1）
                "metadata": node.node.metadata   # 元数据（文件信息等）
            }
            for node in ev.nodes
        ]

        # ------------------------------
        # 根据输出模式构建结果
        # ------------------------------
        if self.streaming:
            # 流式模式：返回生成器供前端实时消费
            # ev.response 是 StreamingResponse 对象，包含 response_gen 异步生成器
            result = {
                "query": ev.query,                    # 用户原始查询
                "stream": ev.response.response_gen,   # 流式回答生成器
                "sources": source_info,               # 来源信息列表
                "is_streaming": True                  # 标记流式模式
            }
        else:
            # 非流式模式：返回完整的回答字符串
            result = {
                "query": ev.query,                    # 用户原始查询
                "response": ev.response,              # 完整回答文本
                "sources": source_info,               # 来源信息列表
                "is_streaming": False                 # 标记非流式模式
            }

        # ------------------------------
        # 发送停止事件，结束工作流
        # ------------------------------
        # StopEvent会将result传递给workflow.run()的返回值
        return StopEvent(result=result)
