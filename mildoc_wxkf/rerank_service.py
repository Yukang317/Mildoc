"""
重排序服务模块

支持多个重排序服务提供商：
- 阿里百炼平台 (dashscope)
- 硅基流动平台 (siliconflow)
"""

import logging
import requests # 导入requests库，这是Python里最著名的发HTTP网络请求的库（相当于浏览器发请求）
import json
from enum import Enum   # 创建“只能选固定几个值”的特殊变量类类型
from typing import List, Optional, Dict, Any
from pydantic import BaseModel # 导入数据打包盒，用来规范返回的数据长什么样
from config import Config

# 配置日志
logging.basicConfig(level=logging.INFO) # 全局日志级别为INFO，只打印重要信息，不打印废话
logger = logging.getLogger(__name__)


class RerankProvider(Enum):
    """重排序服务提供商"""
    DASHSCOPE = "dashscope"  # 阿里百炼
    SILICONFLOW = "siliconflow"  # 硅基流动


class RerankDocument(BaseModel):
    """重排序文档模型"""
    index: int  # 原始文档索引
    content: str  # 文档内容
    relevance_score: float  # 相关性分数
    metadata: Optional[Dict[str, Any]] = None  # 元数据，可选


class RerankResponse(BaseModel):
    """重排序响应模型"""
    documents: List[RerankDocument]  # 重排序后的文档
    success: bool = True
    error_message: Optional[str] = None


class RerankService:
    """重排序服务类
    
    支持多个重排序服务提供商，通过统一接口提供文档重排序功能
    """
    
    def __init__(self, provider: RerankProvider, api_key: str, model_name: str, endpoint: Optional[str] = None):
        """初始化重排序服务
        
        Args:
            provider: 服务提供商
            api_key: API密钥
            model_name: 模型名称
            endpoint: 自定义API端点（可选）
        """
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.endpoint = endpoint
        logger.info(f"重排序服务初始化完成: provider={provider.value}, model={model_name}, endpoint={self.endpoint}")
    
    # 对外暴露的核心方法
    def rerank_documents(self, query: str, documents: List[str], top_n: Optional[int] = None) -> RerankResponse:
        """重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回前N个文档，默认返回全部
            
        Returns:
            RerankResponse: 重排序结果
        """
        try:
            if not query or not documents:
                logger.warning("查询或文档列表为空")
                return RerankResponse(
                    documents=[],
                    success=False,
                    error_message="查询或文档列表为空"
                )
            
            logger.info(f"🔄 开始重排序: 查询='{query[:50]}...', 文档数量={len(documents)}, top_n={top_n}")
            
            if self.provider == RerankProvider.DASHSCOPE:
                response = self._rerank_dashscope(query, documents, top_n)
            elif self.provider == RerankProvider.SILICONFLOW:
                response = self._rerank_siliconflow(query, documents, top_n)
            else:
                raise ValueError(f"不支持的重排序提供商: {self.provider}")
            
            if response.success:
                logger.info(f"✅ 重排序完成: 返回{len(response.documents)}个文档")
                for i, doc in enumerate(response.documents): 
                    logger.info(f"  #{i+1}: 相关性={doc.relevance_score:.4f}, 内容='{doc.content[:50]}...'")
            
            return response
                
        except Exception as e:
            logger.error(f"❌ 重排序失败: {e}")
            return RerankResponse(
                documents=[],
                success=False,
                error_message=str(e)
            )
    
    def _rerank_dashscope(self, query: str, documents: List[str], top_n: Optional[int] = None) -> RerankResponse:
        """阿里百炼平台重排序"""
        # 请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",  # 身份验证
            "Content-Type": "application/json"          # 请求体为JSON格式，别让服务器当作普通文本思密达
        }
        
        # 请求体
        data = {
            "model": self.model_name,
            "input": {
                "query": query,
                "documents": documents  # 文档列表
            },
            "parameters": {
                "return_documents": True,  # 显式设置返回文档内容
                "top_n": top_n or len(documents)
            }
        }
        
        logger.debug(f"🌐 发送请求到百炼平台: {self.endpoint}")
        response = requests.post(self.endpoint, headers=headers, json=data, timeout=30) # 30s
        response.raise_for_status()  # 检查网络状态码，如果厂家返回404或者500错误，直接抛出异常进入上面的except
        
        result = response.json()
        logger.debug(f"📨 百炼平台响应: {json.dumps(result, ensure_ascii=False)[:200]}...") # 保证中文不乱码，截取前200字符保证日志不爆炸
        
        # 解析响应
        rerank_docs = []
        # 防御性编程：先检查厂家返回的字典里有没有这两层钥匙，没有说明格式大错特错
        if "output" in result and "results" in result["output"]:
            for item in result["output"]["results"]:
                # 百炼平台返回格式: document.text
                document = item.get("document", {})
                # 如果document是字典，就拿里面的text；如果它发神经发了别的类型，就强制转成字符串
                content = document.get("text", "") if isinstance(document, dict) else str(document)
                
                rerank_doc = RerankDocument(
                    index=item.get("index", 0),
                    content=content,        # 内容
                    relevance_score=float(item.get("relevance_score", 0.0)) # 相关性分数
                )
                rerank_docs.append(rerank_doc)
        else:
            logger.warning(f"百炼平台响应格式异常: {result}")
            return RerankResponse(
                documents=[],
                success=False,
                error_message="响应格式异常"
            )
        
        return RerankResponse(documents=rerank_docs)
    
    def _rerank_siliconflow(self, query: str, documents: List[str], top_n: Optional[int] = None) -> RerankResponse:
        """硅基流动平台重排序"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "query": query,
            "documents": documents,   # 文档列表
            "return_documents": True  # 硅基流动需要明确指定返回文档内容
        }
        
        # 硅基流动支持top_n参数
        if top_n is not None:
            data["top_n"] = top_n
        
        logger.debug(f"🌐 发送请求到硅基流动: {self.endpoint}")
        response = requests.post(self.endpoint, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        logger.debug(f"📨 硅基流动响应: {json.dumps(result, ensure_ascii=False)[:200]}...")
        
        # 解析响应
        rerank_docs = []
        if "results" in result:  # 注意这里！硅基流动的返回结果直接就在result里，没有output这一层嵌套！
            for item in result["results"]:
                # 硅基流动平台返回格式: document.text
                document = item.get("document", {})
                content = document.get("text", "") if isinstance(document, dict) else str(document) # 提取文本内容
                
                rerank_doc = RerankDocument(
                    index=item.get("index", 0),
                    content=content,
                    relevance_score=float(item.get("relevance_score", 0.0))
                )
                rerank_docs.append(rerank_doc)
        else:
            logger.warning(f"硅基流动响应格式异常: {result}")
            return RerankResponse(
                documents=[],
                success=False,
                error_message="响应格式异常"
            )
        
        return RerankResponse(documents=rerank_docs)
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        status = {
            "service": "RerankService",
            "provider": self.provider.value,    
            "model": self.model_name,
            "endpoint": self.endpoint,      # 请求的网址
            "status": "unknown"             # 当前状态：未知
        }
        
        try:
            # 简单的健康检查：用最少的文档进行测试
            test_response = self.rerank_documents(
                query="测试",
                documents=["这是一个测试文档"],
                top_n=1
            )
            
            if test_response.success:
                status["status"] = "healthy"
                status["test_result"] = "ok"
            else:
                status["status"] = "error"
                status["error"] = test_response.error_message
                
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
        
        return status


# 工厂函数，专门负责“生产”重排服务实例
def create_rerank_service() -> Optional[RerankService]:
    """创建重排序服务实例（工厂函数）
    
    Returns:
        RerankService: 重排序服务实例，失败时返回None
    """
    try:
        # 检查配置
        if not Config.RERANK_PROVIDER:  # 厂商未配置
            logger.info("未配置RERANK_PROVIDER，跳过重排序服务初始化")
            return None
        
        if not Config.RERANK_API_KEY:
            logger.warning("缺少RERANK_API_KEY配置")
            return None
            
        if not Config.RERANK_MODEL_NAME:
            logger.warning("缺少RERANK_MODEL_NAME配置")
            return None
        
        # 创建提供商枚举
        try:
            # 配置文件里是字符串，代码里是：“self.provider == RerankProvider.DASHSCOPE”，这是在拿枚举对象和对象比。需要转换：
            #       它把字符串 "dashscope" 扔进了枚举类里。枚举类会去自己的菜单里找，如果找到了，就返回对应的枚举对象赋给provider
            provider = RerankProvider(Config.RERANK_PROVIDER.lower())
        except ValueError:
            logger.error(f"不支持的重排序提供商: {Config.RERANK_PROVIDER}")
            return None
        
        # 创建服务实例
        service = RerankService(
            provider=provider,
            api_key=Config.RERANK_API_KEY,
            model_name=Config.RERANK_MODEL_NAME,
            endpoint=Config.RERANK_ENDPOINT         # 传入网址（可能为空）
        )
        
        logger.info(f"重排序服务创建成功: {provider.value}")
        return service
        
    except Exception as e:
        logger.error(f"创建重排序服务失败: {e}")
        return None


# 全局重排序服务实例，在文件最外层定义一个全局变量，当做专门放重排服务对象的“单间”，初始为空
_rerank_service_instance = None

def get_rerank_service() -> Optional[RerankService]:
    """获取重排序服务实例（单例模式）"""
    global _rerank_service_instance
    
    # 如果单间里是空的（说明还没造过这个对象）
    if _rerank_service_instance is None:
        # 调用上面的工厂函数造一个，塞进单间里
        _rerank_service_instance = create_rerank_service()
    
    return _rerank_service_instance


if __name__ == "__main__":
    # 测试代码
    rerank_service = get_rerank_service()
    if rerank_service:
        # 健康检查
        health = rerank_service.health_check()
        print(f"健康检查: {health}")
        
        # 测试重排序
        test_docs = [
            "苹果是一种水果，含有丰富的维生素",
            "量子计算是计算科学的前沿领域，苹果也在做这方面的研究",
            "苹果是一家科技企业，主要生产手机、电脑等产品"
        ]
        
        result = rerank_service.rerank_documents(
            query="苹果公司",
            documents=test_docs,
            top_n=3
        )
        
        if result.success:
            print("\n重排序结果:")
            for i, doc in enumerate(result.documents):
                print(f"{i+1}. [分数: {doc.relevance_score:.4f}] {doc.content}")
        else:
            print(f"重排序失败: {result.error_message}")
    else:
        print("重排序服务未配置或初始化失败")
