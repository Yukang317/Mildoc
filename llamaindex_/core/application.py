# ==============================
# 模块: RAG 应用核心层
# 所属文件: core/application.py
# ==============================
"""
RAG应用核心层 - 负责协调文档摄取、检索生成和会话管理

核心职责：
1. 文档摄取：协调 DocumentIngestionPipeline 完成文档解析、切分、向量化
2. 检索生成：协调 RAGWorkflow 完成混合检索、重排序、LLM生成
3. 会话管理：维护多用户会话状态，支持上下文隔离
4. 流式输出：支持 SSE 实时推送，先返回来源再流式输出回答

技术架构：
┌─────────────────────────────────────────────────────────┐
│                    RAGApplication                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐   ┌──────────────────────┐   │
│  │ DocumentIngestion    │   │    RAGWorkflow      │   │
│  │    Pipeline          │   │  (Vector+BM25+RRF)  │   │
│  └──────────┬───────────┘   └──────────┬───────────┘   │
│             │                          │                │
│             ▼                          ▼                │
│  ┌──────────────────────────────────────────────────┐   │
│  │               sessions (会话管理)                │   │
│  │  {session_id: [{role, content, sources}, ...]}  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

核心设计模式：
- 单例模式：整个应用共享一个 RAGApplication 实例
- 懒加载：Workflow 按需创建，避免启动开销
- 状态隔离：通过 session_id 实现多用户数据隔离
"""

# ==============================
# 导入依赖
# ==============================
import json
from typing import List, Optional, Tuple, Dict, Any, AsyncGenerator

# 核心组件
from core.ingestion import DocumentIngestionPipeline  # 文档摄取管道
from core.workflow import RAGWorkflow                  # RAG工作流

# 工具模块
from utils.logger import setup_logger                  # 日志工具
from llama_index.core import Settings                  # LlamaIndex全局设置
from config.settings import Settings as AppSettings    # 应用配置

# 辅助模块
import base64                                         # Base64编码（图片处理）
import copy                                           # 深拷贝（历史记录处理）
from pathlib import Path                               # 路径处理
import os                                             # 文件系统操作

# ==============================
# 初始化组件
# ==============================
# 创建日志记录器
logger = setup_logger(__name__)


# ==============================
# 辅助函数：图片处理
# ==============================
def image_to_base64(path: str) -> str:
    """
    将图片文件转为 Base64 字符串，方便前端 <img src='data:...'> 内联展示
    
    应用场景：
    - PDF解析时提取的图片需要在前端展示
    - 将图片转为Base64内嵌到HTML中，避免额外的图片请求
    
    Args:
        path: 图片文件路径（可以是完整路径或仅文件名）
        
    Returns:
        Base64编码的图片字符串，失败时返回空字符串
    """
    try:
        # 提取文件名（处理完整路径或仅文件名两种情况）
        file_name = Path(path).name
        # 构建完整的图片存储路径（从配置中读取存储目录）
        file_path = AppSettings.PDF_IMAGE_DIR + rf"\{file_name}"
        # 读取文件并转为Base64
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"图片转Base64失败: {e}")
        return ""


# ==============================
# RAG应用主类
# ==============================
class RAGApplication:
    """
    RAG应用主类 - 业务逻辑的核心协调者
    
    核心职责：
    1. 文档摄取管理：协调 DocumentIngestionPipeline 完成文档解析、切分、向量化
    2. 检索生成协调：协调 RAGWorkflow 完成混合检索、重排序、LLM生成
    3. 多会话管理：维护多用户会话状态，实现上下文隔离
    4. 流式输出支持：支持 SSE 实时推送，先返回来源再流式输出回答
    
    对外暴露的核心方法：
    - upload_and_process_files(files): 上传并处理文件，构建索引
    - query_documents(session_id, query, knowledge_bool): 非流式查询
    - query_documents_stream(session_id, query, knowledge_bool): 流式查询
    - clear_session(session_id): 清空指定会话
    - reset(): 系统级重置
    
    数据结构：
        sessions: Dict[str, List[Dict[str, Any]]]
            - key: session_id（通常为用户名，确保用户隔离）
            - value: 消息列表，每条消息格式：{"role": "user"/"assistant", "content": "...", "sources": [...]}
    """

    def __init__(self) -> None:
        # ------------------------------
        # 初始化文档摄取管道
        # ------------------------------
        # 负责文档解析、切分、向量化、建索引的完整流程
        self.ingestion_pipeline = DocumentIngestionPipeline()
        
        # 启动时尝试加载已有索引（如果存在）
        # 这样服务重启后可以快速恢复之前的文档索引
        self.ingestion_pipeline.get_documents()
        
        # ------------------------------
        # 初始化工作流（懒加载）
        # ------------------------------
        # RAG工作流实例，基于索引构建
        # 采用懒加载模式：第一次查询时才创建，避免启动开销
        self.workflow: Optional[RAGWorkflow] = None

        # ------------------------------
        # 初始化会话存储
        # ------------------------------
        # 结构: { session_id: [ {"role": "...", "content": "...", "sources": [...]}, ... ] }
        # session_id 通常使用用户名，确保不同用户的历史记录完全隔离
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

    def update_model_config(self, model_name: str, temperature: float, max_tokens: int):
        """
        动态更新 LLM 模型配置
        
        应用场景：
        - 用户在前端切换模型时调用
        - 调整生成参数（温度、最大token数）
        
        Args:
            model_name: 模型名称（如 "qwen-plus"）
            temperature: 生成温度（0-1，越高越随机）
            max_tokens: 最大生成token数
        """
        self.ingestion_pipeline.update_model_config(model_name, temperature, max_tokens)

    # ==============================
    # 文档摄取与索引
    # ==============================
    def upload_and_process_files(self, files) -> Tuple:
        """
        上传并处理文件（构建或更新向量索引）
        
        处理流程：
        1. 解析文件路径（支持文件对象或字符串路径）
        2. 调用 DocumentIngestionPipeline 完成：
           - 文档解析（PDF→Markdown、文本提取）
           - 智能分块（根据文档类型选择不同策略）
           - 向量化（使用 HuggingFace 嵌入模型）
           - 索引构建（Chroma 向量库）
        3. 构建/更新 RAGWorkflow（支持混合检索）
        
        Args:
            files: 文件对象集合或路径字符串集合
                   - 若对象带 .name 属性则取其名称作为路径
                   - 否则直接转为字符串使用
        
        Returns:
            Tuple: (status, result)
                   - status: "success" 或 "error"
                   - result: 成功时为处理结果说明，失败时为错误信息
        """
        if not files:
            return "请上传至少一个文件"

        try:
            # ------------------------------
            # 步骤1: 解析文件路径
            # ------------------------------
            file_paths: List[str] = []
            for file in files:
                # 处理文件对象（如 UploadFile）或字符串路径
                if hasattr(file, "name"):
                    file_paths.append(file.name)
                else:
                    file_paths.append(str(file))

            # ------------------------------
            # 步骤2: 执行文档摄取
            # ------------------------------
            # 调用摄取管道完成解析、切分、向量化、建索引
            status, result, pipeline_nodes = self.ingestion_pipeline.ingest_documents(file_paths)
            
            # 检查是否有错误
            if status == "error":
                raise RuntimeError(result)

            # ------------------------------
            # 步骤3: 更新工作流
            # ------------------------------
            # 若有可用索引，重建工作流以包含新文档
            if self.ingestion_pipeline.index:
                self.workflow = RAGWorkflow(self.ingestion_pipeline.index, pipeline_nodes)

            return status, result

        except Exception as e:
            error_msg = f"文件处理失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

    # ==============================
    # 工具方法：会话管理与上下文处理
    # ==============================

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取指定会话的历史记录列表
        
        核心逻辑：
        - 如果会话不存在，自动创建空列表（惰性初始化）
        - 返回的是引用，可直接 append 修改
        
        数据隔离机制：
        - session_id 通常使用用户名
        - 不同用户的 session_id 不同，实现数据隔离
        
        Args:
            session_id: 会话ID（通常为用户名）
            
        Returns:
            会话历史列表的引用，每条消息格式：{"role": "...", "content": "...", "sources": [...]}
        """
        if session_id not in self.sessions:
            # 惰性初始化：会话不存在时创建空列表
            self.sessions[session_id] = []
        # 返回引用，调用方可以直接修改（如 append）
        return self.sessions[session_id]

    def get_safe_history(self, session_history: List[Dict[str, Any]],
                         max_history_length: int = 5,
                         max_token_limit: int = 4000) -> List[Dict[str, Any]]:
        """
        获取用于发送给 LLM 的"安全"历史记录（双重限制保护）
        
        核心优化策略（防止上下文溢出）：
        1. 轮数限制：只保留最近 N 轮对话（默认5轮）
        2. Token限制：字符数估算限制（默认4000 token ≈ 12000字符）
        3. Sources清理：移除历史中的 sources 字段（包含Base64图片和长文本）
        
        设计原因：
        - LLM 有上下文窗口限制（如 8k/16k/32k tokens）
        - 历史记录中的 sources 包含大量冗余信息（图片、完整文档片段）
        - 模型只需要理解对话逻辑，不需要看旧的检索源
        
        Args:
            session_history: 原始会话历史列表
            max_history_length: 保留最近 N 轮对话
            max_token_limit: 估算的最大字符限制 (粗略按 1 token ≈ 2-3 chars 估算，这里设为字符数阈值)

        Returns:
            List[Dict[str, Any]]: 精简后的历史列表
        """
        # 空历史直接返回
        if not session_history:
            return []

        # ------------------------------
        # 步骤1: 轮数限制 + 深拷贝
        # ------------------------------
        # 取最近的 N 轮对话（user + assistant = 2条，所以 * 2）
        # 深拷贝避免修改原始数据（原始数据包含图片用于前端展示）
        slice_index = max(0, len(session_history) - (max_history_length * 2))
        recent_history = copy.deepcopy(session_history[slice_index:])

        # ------------------------------
        # 步骤2: Token限制 + Sources清理
        # ------------------------------
        clean_history = []
        current_char_count = 0
        # 设定一个字符上限（例如 12000 字符，约 4k-6k tokens，根据你的模型调整）
        CHAR_LIMIT = max_token_limit * 3

        # 倒序处理：优先保留最近的对话
        for msg in reversed(recent_history):
            # --- 核心优化1: 移除 sources 字段 ---
            # 历史记录中的 sources 包含 Base64 图片和长文档片段
            # LLM 上下文不需要这些信息，移除可大幅减少 token 消耗
            if "sources" in msg:
                del msg["sources"]

            # --- 核心优化：移除可能混入 content 中的 Base64 ---
            # 如果你的 content 字段里也意外混入了 html 标签或 base64，可以在这里清理
            # 这里假设 content 是纯文本，如果不是，可以用正则清理
            content_str = str(msg.get("content", ""))

            # --- 核心优化3: 长度检查 ---
            msg_len = len(content_str)
            if current_char_count + msg_len > CHAR_LIMIT:
                logger.info(f"历史记录触达字符限制，已截断。当前保留字符数: {current_char_count}")
                break

            current_char_count += msg_len
            # 插入到最前面，恢复原始顺序
            clean_history.insert(0, msg)

        # ------------------------------
        # 步骤3: 格式校验
        # ------------------------------
        # 某些模型要求对话以 user 开头
        # 如果截断后第一条是 assistant，可选择丢弃或保留
        if clean_history and clean_history[0]["role"] == "assistant":
            # 如果截断后第一条是 assistant，通常为了上下文连贯性可以选择丢弃，或者保留（视模型鲁棒性而定）
            # 这里选择保留，但打个日志
            pass

        return clean_history

    # ==============================
    # 工作流状态管理
    # ==============================
    def _ensure_workflow(self, streaming: bool):
        """
        [状态管理] 确保 Workflow 已初始化，且处于正确的流式/非流式模式。
        
        核心设计与背景：
        - LlamaIndex 的 ResponseSynthesizer 在初始化时确定模式（流式/非流式）'streaming' 还是 'compact'
        - 如果模式不匹配（用户上一次请求是非流式，这一次是流式，），需要重新创建 Workflow 实例，否则调用 .run() 时行为会不符合预期。
        - 采用懒加载：第一次查询时才创建，避免启动开销
        
        触发重建的条件：
        1. Workflow 尚未创建（self.workflow is None）
        2. 当前模式与请求模式不一致
        
        Args:
            streaming: True=流式输出，False=非流式输出
            
        Raises:
            RuntimeError: 如果没有可用的索引
        """
        # ------------------------------
        # 步骤1: 确保索引已加载（懒加载）
        # ------------------------------
        if not self.ingestion_pipeline.index:
            # 尝试从存储加载索引
            self.ingestion_pipeline.get_documents()

        # 检查是否有可用索引
        if not self.ingestion_pipeline.index:
            raise RuntimeError("索引未构建，无法创建 Workflow")

        # ------------------------------
        # 步骤2: 检查并重建工作流（如果需要）
        # ------------------------------
        # 条件：workflow 不存在 或 模式不匹配
        current_mode = getattr(self.workflow, 'streaming', None)
        if (self.workflow is None) or (current_mode != streaming):
            logger.info(f"初始化 RAGWorkflow (Streaming={streaming})")
            self.workflow = RAGWorkflow(
                self.ingestion_pipeline.index,
                pipeline_nodes=None,  # 👈 重启模式，不添加新节点
                streaming=streaming
            )

    # ==============================
    # 核心业务方法：非流式查询（一次性返回）
    # ==============================
    async def query_documents(
            self,
            session_id: str,
            query: str,
            knowledge_bool: bool,
    ) -> Tuple[str, str]:
        """
        [主入口] 执行非流式查询，等待所有处理完成后一次性返回结果
        
        适用场景：
        - 不需要实时打字机效果的场景
        - 需要完整结果后再处理的业务逻辑
        
        处理流程：
        1. 准备上下文：获取会话历史，追加用户问题
        2. 分支判断：
           - RAG模式(knowledge_bool=True)：检索 + 重排序 + 生成
           - 纯LLM模式(knowledge_bool=False)：直接调用LLM
        3. 记录结果到会话历史
        4. 返回结果
        
        Args:
            session_id: 会话ID（用于隔离不同用户的上下文，通常为用户名）
            query: 用户的问题
            knowledge_bool: 是否启用知识库检索模式
            
        Returns:
            Tuple[str, str]: (回复文本, 来源信息列表)
        """

        # ------------------------------
        # Step 1: 上下文准备
        # ------------------------------
        # 获取该会话的历史记录（引用，可直接修改）
        hist_srv = self.get_session_history(session_id)
        # 将用户问题追加到历史，作为本次生成的上下文
        hist_srv.append({"role": "user", "content": query})

        try:
            if knowledge_bool:
                # ==============================
                # 分支 A: RAG 知识库模式（检索 + 生成）
                # ==============================

                # Step 1: 确保工作流为非流式模式
                # LlamaIndex 的 ResponseSynthesizer 在初始化时确定模式
                # 如果上一次是流式，需要重置为非流式
                self._ensure_workflow(streaming=False)

                # Step 2: 执行工作流
                # 非流式模式下，await 会阻塞直到所有步骤完成：
                # 检索(Retrieval) -> 重排序(Rerank) -> LLM生成(Generation)
                result: Dict[str, Any] = await self.workflow.run(
                    query=query,
                    timeout=60.0  # 60秒超时
                )

                # Step 3: 解析结果
                # 非流式模式下，response 是完整的字符串
                response_text = str(result.get("response", ""))

                # 提取来源节点（用于前端展示引用）
                sources = result.get("sources", [])

                # Step 4: 格式化来源信息转为前端可读的字符串
                # 添加相似度、文件名、图片预览等
                sources_info_list = self._format_sources(sources)

                # Step 5: 记录到会话历史
                hist_srv.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources_info_list
                })

                return response_text, sources_info_list

            else:
                # ==============================
                # 分支 B: 纯 LLM 聊天模式（无检索）
                # ==============================

                # Step 1: 安全处理历史记录
                # 清理 sources 字段、限制长度，防止上下文溢出
                safe_history = self.get_safe_history(hist_srv)
                
                # Step 2: 调用 LLM
                # 使用异步方法 acomplete，不阻塞事件循环
                try:
                    history_json = json.dumps(safe_history, ensure_ascii=False)
                    logger.debug(f"历史记录JSON大小: {len(history_json)} 字符")
                    llm_res = await Settings.llm.acomplete(history_json)
                except Exception as e:
                    logger.error(f"LLM调用失败: {str(e)}")
                    raise

                response_text = llm_res.text

                # Step 3: 记录到会话历史
                # 纯聊天模式没有来源信息
                hist_srv.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": []
                })

                return response_text, ""

        except Exception as e:
            # ------------------------------
            # 异常处理
            # ------------------------------
            error_msg = f"查询失败: {str(e)}"
            logger.error(error_msg)

            # 记录错误到历史，保证对话连贯性
            hist_srv.append({
                "role": "assistant",
                "content": error_msg,
                "sources": []
            })

            return error_msg, ""

    # ==============================
    # 核心业务方法：流式查询（支持 SSE）
    # ==============================
    async def query_documents_stream(
            self,
            session_id: str,
            query: str,
            knowledge_bool: bool,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行流式查询，逐步返回来源信息和生成的文本（支持 SSE 推送）
        - 实时反馈：用户可以看到打字机效果
        - 来源优先：先返回文档引用，再流式输出回答
        - 渐进式加载：提升用户体验

        流程:
            1. 预处理：获取历史，追加用户问题。
            2. 分支判断：
               - RAG模式: 检索 -> 返回来源 -> 流式生成文本。
               - 纯LLM模式: 直接流式生成文本。
            3. 后处理：拼接完整回复，存入历史，发送完成信号。
        
        推送顺序（RAG模式）：
        1. sources: 来源信息（文档引用、相似度、图片等）
        2. text: 文本片段（增量推送，每次一个token）
        3. complete: 完成信号（包含完整回答）
        
        Args:
            session_id: 会话ID（用户隔离）
            query: 用户问题
            knowledge_bool: 是否启用知识库检索
            
        Returns:
            AsyncGenerator: 异步生成器，逐块返回数据
        数据结构（yield 返回）：
        {
            "type": "sources" | "text" | "complete" | "error",      # 消息类型
            "content": str | List,      # 内容（来源列表或文本片段）
            "finished": bool            # 是否结束
        }
        """
        # ------------------------------
        # Step 1: 上下文准备
        # ------------------------------
        # 获取会话历史（引用，修改会影响存储）
        hist_srv = self.get_session_history(session_id)
        # 追加用户问题到历史
        hist_srv.append({"role": "user", "content": query})

        # 用于拼接完整回答（最后存入历史）
        full_response = ""
        # 存储格式化后的来源信息
        sources_info_list = []

        try:
            if knowledge_bool:
                # ==============================
                # 分支 A: RAG 知识库模式（检索 + 流式生成）
                # ==============================

                # Step 1: 确保工作流为流式模式
                self._ensure_workflow(streaming=True)

                # Step 2: 执行工作流（检索 + 重排序）
                #       注意: 这里的 await 主要是等待 "检索(Retrieval)" 和 "重排序(Rerank)" 完成。
                #       一旦 LLM 开始生成，它就会返回一个包含 Generator 的结果对象，不会阻塞到文本完全生成结束。
                result = await self.workflow.run(query=query, timeout=60.0)

                # Step 3: 优先推送来源信息（提升用户体验）
                raw_sources = result.get("sources", [])
                if raw_sources:
                    # 格式化来源 (添加相似度、图片预览等)
                    sources_info_list = self._format_sources(raw_sources)
                    
                    # 🚀 Yield 1: 推送来源卡片（用户先看到参考文档）
                    yield {
                        "type": "sources",
                        "finished": False,
                        "content": sources_info_list,
                    }

                # Step 4: 流式推送文本（核心）
                #    result["stream"] 是我们在 Workflow 的 finalize_step 中返回的 response_gen
                stream_gen = result.get("stream")
                
                if stream_gen:
                    # 实时迭代 LLM 的 token 生成器
                    async for token in stream_gen:
                        full_response += token  # 拼接完整回答
                        
                        # 🚀 Yield 2: 发送文本片段，推送单个 token（增量）
                        yield {
                            "type": "text",
                            "finished": False,
                            "content": token,  # 注意：增量字符，不是全量
                        }
                else:
                    # [兜底逻辑] 万一 Workflow 没返回流 (比如 fallback 到了非流式)
                    full_response = str(result.get("response", ""))
                    yield {"type": "text", "finished": False, "content": full_response}

            else:
                # ==============================
                # 分支 B: 纯 LLM 聊天模式（无检索）
                # ==============================

                # 安全处理历史记录
                safe_history = self.get_safe_history(hist_srv)
                
                # 调用 LLM 流式接口
                try:
                    history_json = json.dumps(safe_history, ensure_ascii=False)
                    logger.debug(f"流式历史记录JSON大小: {len(history_json)} 字符")
                    stream_response = await Settings.llm.astream_complete(history_json)
                except Exception as e:
                    logger.error(f"LLM流式调用失败: {str(e)}")
                    raise

                # 迭代流式响应
                async for chunk in stream_response:
                    token = chunk.delta  # 获取增量文本
                    full_response += token
                    
                    # 推送文本片段
                    yield {
                        "type": "text",
                        "finished": False,
                        "content": token,
                    }

            # ------------------------------
            # Step 3: 收尾工作
            # ------------------------------
            
            # 记录完整对话到历史
            hist_srv.append({
                "role": "assistant",
                "content": full_response.strip(),
                "sources": sources_info_list  # 只有 RAG 模式这里才有值
            })

            # 🚀 Yield 3: 发送完成信号
            # 前端收到这个信号后，可以停止加载动画，解锁输入框
            yield {
                "type": "complete",
                "finished": True,           # 前端收到 finished: True 后停止加载动画，解锁输入框。
                "content": full_response.strip(),
            }

        except Exception as e:
            # ------------------------------
            # 异常处理
            # ------------------------------
            error_msg = f"流式查询过程中发生错误: {str(e)}"
            logger.error(error_msg)

            # 也要记录错误到历史，避免上下文中断
            hist_srv.append({"role": "assistant", "content": error_msg, "sources": []})

            # 发送错误信号给前端
            yield {
                "type": "error",
                "content": error_msg,
                "finished": True
            }

    # ==============================
    # 来源信息格式化方法
    # ==============================
    def _extract_images_from_markdown(self, content_preview: str) -> str:
        """
        从Markdown内容中提取图片并转换为Base64 HTML格式
        
        应用场景：
        - PDF解析后可能包含图片引用（如 ![alt](path/to/image.png)）
        - 需要将这些图片转为Base64内联展示在前端
        
        处理流程：
        1. 使用正则提取Markdown图片标签
        2. 去重处理（避免重复展示同一张图片）
        3. 将图片转为Base64编码
        4. 构建HTML img标签
        
        Args:
            content_preview: Markdown格式的内容预览文本
            
        Returns:
            str: 包含HTML图片标签的字符串（可直接嵌入HTML显示）
        """
        img_html = ""
        try:
            import re
            # 正则匹配Markdown图片格式：![alt](path)，支持Windows路径格式
            img_pattern = r'!\[.*?\]\((D:/llm/[^)]+\.png)\)'    
            matches = re.findall(img_pattern, content_preview)
            
            # 去重集合，避免重复处理同一张图片
            processed_img_names = set()
            
            for img_path in matches:
                # 清理路径中的引号
                img_path = img_path.strip('"').strip("'")
                # 提取文件名
                img_name = os.path.basename(img_path)
                
                # 避免重复处理
                if img_name not in processed_img_names:
                    processed_img_names.add(img_name)
                    # 转换为Base64
                    b64_img = image_to_base64(img_name)
                    if b64_img:
                        # 构建HTML（带样式）
                        img_html += (
                            '<div style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">'
                            '<p style="margin: 0 0 10px 0; font-weight: bold;">文档相关的图片：</p>'
                            f'<img src="data:image/jpeg;base64,{b64_img}" width="300" style="border-radius: 3px; max-width: 100%;"/>'
                            '</div>'
                        )
        except Exception as md_img_e:
            logger.warning(f"从Markdown提取图片失败: {md_img_e}")
        
        return img_html
    
    def _format_single_source(self, source_index: int, source: Dict) -> str:
        """
        格式化单个来源信息为可读字符串
        
        输出格式：
        {序号}. 相似度: {分数} | 文件: {文件名}
           内容: {预览文本}
           [图片HTML]
        
        Args:
            source_index: 来源的序号（从1开始）
            source: 来源字典，包含score、content、metadata
            
        Returns:
            str: 格式化后的来源信息字符串
        """
        # 提取来源信息
        score = float(source.get("score"))
        content_preview = source.get("content", "")
        metadata = source.get("metadata", {}) or {}
        
        # 获取文档类型和文件名
        content_type = metadata.get("content_type", "text")
        file_name = metadata.get("file_name", "未知文件")

        sources_info = ""

        # 添加相似度和文件信息
        if isinstance(score, (int, float)):
            sources_info += f"{source_index}. 相似度: {score:.3f} | 文件: {file_name}\n"
        else:
            sources_info += f"{source_index}. 相似度: - | 文件: {file_name}\n"

        # 添加内容预览
        sources_info += f"   内容: {content_preview}\n\n"

        # 处理Markdown中的图片
        if content_type == "markdown" and content_preview:
            sources_info += self._extract_images_from_markdown(content_preview)
        
        return sources_info
    
    def _format_sources(self, sources: List[Dict]) -> List[str]:
        """
        批量格式化来源信息列表
        
        Args:
            sources: 来源列表，每个元素包含score、content、metadata
            
        Returns:
            List[str]: 格式化后的来源信息字符串列表
        """
        sources_info_list = []

        for i, source in enumerate(sources, 1):
            sources_info = self._format_single_source(i, source)
            sources_info_list.append(sources_info)

        return sources_info_list

    # ==============================
    # 安全调用封装
    # ==============================
    async def _run_workflow_safe(self, query: str, timeout: float) -> Dict[str, Any]:
        """
        安全调用工作流的封装方法（兼容不同版本）
        
        设计目的：
        - 处理不同版本的 RAGWorkflow.run 方法签名差异
        - 提供降级策略，保证系统稳定性
        
        Args:
            query: 用户查询
            timeout: 超时时间（秒）
            
        Returns:
            Dict: 工作流执行结果
            
        Raises:
            RuntimeError: 如果工作流未初始化
        """
        if not self.workflow:
            raise RuntimeError("RAG 工作流未初始化")

        try:
            return await self.workflow.run(query=query, timeout=timeout)
        except TypeError:
            # 处理参数不兼容的情况
            logger.warning("RAGWorkflow.run 不支持当前参数，已降级为默认检索数量")
            return await self.workflow.run(query=query, timeout=timeout)

    # ==============================
    # 会话管理
    # ==============================
    def clear_session(self, session_id: str) -> None:
        """
        清空指定会话的历史记录
        
        Args:
            session_id: 会话ID（通常为用户名）
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

    def reset(self) -> None:
        """
        系统级重置：清空所有会话历史
        
        注意：
        - 只清空内存中的会话数据
        - 不会影响向量库中的文档索引
        - 如需清空索引，请调用 DocumentIngestionPipeline 的相关方法
        """
        self.sessions.clear()
