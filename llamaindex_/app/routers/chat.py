# ==============================
# 模块: 聊天/检索生成 API 路由
# 所属文件: app/routers/chat.py
# ==============================
"""
聊天/检索生成相关的 API 路由定义：

已实现接口：
- POST /api/chat/query: 非流式聊天接口，等待完整回答后返回
- POST /api/chat/stream: 流式聊天接口（SSE），实时返回回答片段
- GET /api/chat/history: 获取当前用户聊天历史
- POST /api/chat/clear: 清空指定会话的历史记录

安全特性：
- 所有接口均需 JWT 认证
- 用户数据隔离（只能访问自己的会话）
- 自动配额检查和限制

技术栈：
- FastAPI: 高性能异步 Web 框架
- StreamingResponse: SSE（Server-Sent Events）流式响应
- Depends: 依赖注入，实现解耦和可测试性
"""

# ==============================
# 导入依赖
# ==============================
# FastAPI 核心组件
from fastapi import APIRouter, Depends, HTTPException

# 请求/响应数据模型
from app.schemas import (
    ChatRequest,      # 聊天请求体（查询、模型参数等）
    ChatResponse,     # 非流式响应体
    ClearRequest,     # 清空会话请求体
    CommonResponse,   # 通用响应体（如成功/失败）
    ChatMessage,      # 消息结构
    User              # 用户信息（认证后获取）
)

# SSE 流式响应支持
from fastapi.responses import StreamingResponse

# 认证依赖：获取当前登录用户信息
from app.routers.users import get_current_active_user

# RAG 服务依赖：获取业务逻辑服务实例
from app.services import get_rag_service
from app.services.rag_service import RAGService

# 日志工具
from utils.logger import setup_logger

# JSON 序列化工具
import json

# 类型提示
from typing import List

# ==============================
# 初始化组件
# ==============================
# 创建日志记录器
logger = setup_logger(__name__)

# 创建 API 路由器实例
# APIRouter 用于组织相关路由，便于模块化管理
router = APIRouter()


# ==============================
# 接口1: 非流式聊天接口
# 路径: POST /api/chat/query
# 功能: 发送问题，等待完整回答后一次性返回
# ==============================
@router.post("/chat", response_model=ChatResponse)
async def chat_query(
    req: ChatRequest,                              # 请求体：包含查询、模型参数等
    current_user: User = Depends(get_current_active_user),  # 依赖注入：自动完成JWT认证，获取当前用户
    svc: RAGService = Depends(get_rag_service)     # 依赖注入：获取RAG业务服务实例
):
    """
    非流式聊天对话接口 - 需要登录认证

    核心功能：
    - 用户发送问题后，等待AI生成完整回答后一次性返回
    - 支持两种模式：RAG知识库模式（检索+生成）和纯LLM聊天模式
    - 自动记录对话历史，支持上下文关联

    安全机制：
    - JWT令牌验证（通过 get_current_active_user 依赖实现）
    - 用户数据隔离：session_id 使用用户名，确保只能访问自己的会话
    - 异常捕获：所有异常统一处理，返回500错误

    Args:
        req (ChatRequest): 聊天请求数据，包含：
            - query: 用户问题
            - model: 模型名称
            - knowledge_bool: 是否启用知识库检索
            - temperature: 生成温度（控制随机性）
            - max_tokens: 最大生成长度

    Returns:
        ChatResponse: 非流式模式的完整响应,包含回答内容和来源引用信息
        StreamingResponse: 流式模式的SSE响应

    Raises:
        HTTPException:
            - 429: 配额已用完
            - 500: AI模型调用失败或其他服务器错误
    """
    try:
        # ------------------------------
        # 步骤1: 获取用户身份
        # ------------------------------
        # 从JWT认证信息中提取用户名，作为会话ID使用
        # 这样确保每个用户只能访问自己的聊天历史
        username = current_user.username

        # ------------------------------
        # 步骤2: 调用业务层执行查询
        # ------------------------------
        # svc.query() 方法执行完整的RAG流程：
        # 1. 更新模型配置（温度、最大token数）
        # 2. 根据knowledge_bool决定检索策略
        # 3. 返回回答文本和来源信息
        answer, sources = await svc.query(
            session_id=username,          # 会话ID（用户名）
            query=req.query,              # 用户问题
            model=req.model,              # 模型名称
            knowledge_bool=req.knowledge_bool or False,  # 是否启用知识库
            temperature=req.temperature,  # 生成温度
            max_tokens=req.max_tokens     # 最大token数
        )

        # 调试输出：打印来源信息（生产环境可移除）
        print(sources)

        # ------------------------------
        # 步骤3: 封装并返回响应
        # ------------------------------
        # 将回答和来源信息封装为标准响应格式
        return ChatResponse(
            messages=ChatMessage(
                role="assistant",    # 角色标识（助手）
                content=answer,      # 回答内容
                sources=sources or []  # 来源引用列表（可能为空）
            )
        )

    except Exception as e:
        # ------------------------------
        # 异常处理
        # ------------------------------
        # 记录错误日志
        logger.error(str(e))
        # 返回500错误，隐藏详细错误信息（安全考虑）
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# 接口2: 流式聊天接口（SSE）
# 路径: POST /api/chat/stream
# 功能: 发送问题，实时返回回答片段（打字机效果）
# ==============================
@router.post("/chat/stream")
async def chat_query_stream(
    req: ChatRequest,                              # 请求体：包含查询、模型参数等
    current_user: User = Depends(get_current_active_user),  # 依赖注入：自动完成JWT认证
    svc: RAGService = Depends(get_rag_service)     # 依赖注入：获取RAG业务服务实例
):
    """
    流式聊天对话接口（SSE）- 需要登录认证

    核心功能：
    - 使用 Server-Sent Events (SSE) 实现实时流式响应
    - 前端可以实时显示AI生成的每一个token，实现打字机效果
    - 支持分阶段推送：先返回来源信息，再流式输出回答内容

    SSE技术特点：
    - 基于HTTP协议，无需额外WebSocket连接
    - 服务器主动推送，客户端被动接收
    - 自动重连机制（浏览器原生支持）

    推送顺序：
    1. sources: 来源信息（文档引用）
    2. text: 回答片段（增量推送）
    3. complete: 完成信号
    4. error: 错误信号（异常情况）

    Args:
        req (ChatRequest): 聊天请求数据
        current_user (User): 通过JWT认证获取的当前用户信息
        svc (RAGService): RAG 应用主类

    Returns:
        ChatResponse: 非流式模式的完整响应
        StreamingResponse: 流式模式的SSE响应，content-type为 text/event-stream

    Raises:
        HTTPException:
            - 429: 配额已用完
            - 500: AI模型调用失败或其他服务器错误
    安全特性：
        - JWT令牌验证
        - 用户会话隔离
    """

    # ------------------------------
    # 获取用户身份
    # ------------------------------
    # 从JWT认证信息中提取用户名，作为会话ID
    username = current_user.username

    # ------------------------------
    # 异步事件生成器（核心）
    # ------------------------------
    # SSE要求响应体是一个异步生成器，逐块返回数据
    async def event_generator():
        """
        SSE事件生成器：负责将业务层返回的chunk转换为SSE格式
        
        SSE格式规范：
        - 每条消息格式: "data: {JSON数据}\n\n"
        - 支持的消息类型: sources, text, complete, error
        """
        try:
            # 调用业务层的流式查询方法
            # svc.query_stream() 返回异步生成器，每次yield一个chunk
            async for chunk in svc.query_stream(
                    session_id=username,          # 会话ID
                    query=req.query,              # 用户问题
                    model=req.model,              # 模型名称
                    knowledge_bool=req.knowledge_bool or [],  # 是否启用知识库
                    temperature=req.temperature,  # 生成温度
                    max_tokens=req.max_tokens     # 最大token数
            ):
                # 将chunk序列化为JSON字符串
                # ensure_ascii=False 支持中文输出
                data = json.dumps(chunk, ensure_ascii=False)
                
                # 按照SSE格式输出
                # 格式: data: {JSON}\n\n
                yield f"data: {data}\n\n"

        except Exception as e:
            # 捕获流式传输过程中的异常
            # 构建错误消息，通知前端
            error_data = {
                "type": "error",                  # 消息类型
                "content": f"流式传输错误: {str(e)}",  # 错误内容
                "done": True                      # 标记传输结束
            }
            # 发送错误消息
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    # ------------------------------
    # 返回流式响应
    # ------------------------------
    return StreamingResponse(
        content=event_generator(),                # 异步生成器作为响应体
        media_type="text/event-stream",           # SSE标准MIME类型，告诉浏览器“这是一个 SSE 流，请保持连接并持续解析”
        headers={
            "Cache-Control": "no-cache",          # 禁用缓存，确保实时性。防止中间代理缓存流数据
            "Connection": "keep-alive",           # 保持长连接。维持 TCP 长连接，避免每次发送都重新握手。
        })

# ==============================
# 接口3: 获取聊天历史
# 路径: GET /api/chat/history
# 功能: 获取当前用户的聊天历史记录
# ==============================
@router.get("/history")
async def get_user_history(
        current_user: User = Depends(get_current_active_user),  # 依赖注入：认证
        svc: RAGService = Depends(get_rag_service)             # 依赖注入：服务
) -> List[ChatMessage]:
    """
    获取当前用户的聊天历史 - 安全版本

    核心功能：
    - 返回当前认证用户的完整聊天历史
    - 自动过滤其他用户的会话数据
    - 保证数据隐私和安全性

    安全机制：
    - 只能访问自己的聊天历史（通过username隔离）
    - 返回空列表而不是错误，避免信息泄露

    返回值：
        List[ChatMessage]: 消息列表，按时间顺序排列
    
    数据格式：
        [{
            "role": "user" | "assistant",
            "content": "消息内容",
            "sources": ["来源1", "来源2", ...]  # 仅assistant消息有
        }, ...]
    """
    # 获取当前用户名
    username = current_user.username
    
    # 安全检查：用户名为空时返回空列表
    if username is None:
        return []

    # 从服务层获取会话历史
    chat_history = svc.get_session(username)
    
    # 历史为空时返回空列表
    if not chat_history:
        return []

    # ------------------------------
    # 转换为标准响应格式
    # ------------------------------
    # 将内部格式转换为前端期望的ChatMessage格式
    return [
        ChatMessage(
            role=msg["role"],                            # 角色（user/assistant）
            content=msg["content"],                      # 消息内容
            sources=(msg["sources"] if msg.get("sources") else [])  # 来源引用
        ) for msg in chat_history
    ]

# ==============================
# 接口4: 清空聊天历史
# 路径: POST /api/chat/clear
# 功能: 清空当前用户的会话历史记录
# ==============================
@router.post("/clear", response_model=CommonResponse)
async def chat_clear(
    req: ClearRequest,                             # 请求体（可为空）
    current_user: User = Depends(get_current_active_user),  # 依赖注入：认证
    svc: RAGService = Depends(get_rag_service)     # 依赖注入：服务
):
    """
    清空指定会话的历史记录

    核心功能：
    - 清除当前用户的聊天历史（内存中的会话数据）
    - 不影响向量库中的文档数据

    注意事项：
    - 只清空内存中的会话历史
    - 不会删除已上传的文档或向量索引
    - 不会影响其他用户的会话

    返回值：
        CommonResponse: 操作结果（success/failure）
    """
    # 调用服务层清空会话
    svc.clear_session(current_user.username)
    
    # 返回成功响应
    return CommonResponse(status="success", message="会话已清空")
