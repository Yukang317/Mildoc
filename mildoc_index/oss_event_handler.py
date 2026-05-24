# 阿里云 OSS 事件处理器 ，负责监听阿里云 OSS 对象存储的事件（如文件上传、删除），并触发相应的处理流程。
#    解决了文档变更的实时感知和处理问题，确保知识库能够及时更新，反映最新的文档状态。
# - 交互关系 ：
#   - 与阿里云 OSS 交互：通过 MNS 消息队列接收事件通知
#   - 与 MinIO 客户端交互：因为兼容阿里云 OSS，所以使用同一套 MinIO 客户端
#   - 与 SimpleObjectParser 交互：解析上传的文档
#   - 与 EmbeddingTool 交互：生成文档片段的向量表示
#   - 与 MilvusAPI 交互：将解析后的文档存储到 Milvus 向量数据库
import json
from datetime import datetime
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable
import os
from dotenv import load_dotenv
from dataclasses import dataclass   # 数据类装饰器

from minio import Minio                             # MinIO客户端，用于与对象存储交互
from mns.account import Account                     # 阿里云消息队列账户
from parser.simple_object_parser import SimpleObjectParser
from embedding import EmbeddingTool
from milvus_api import MilvusAPI, MilvusDocument    # Milvus操作接口
from logger.logging import setup_logging

load_dotenv()

logger = setup_logging()

# Minio 配置，因为兼容阿里云 OSS，所以使用同一套MinIO客户端。
MINIO_BUCKET = os.getenv('MINIO_BUCKET')
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')        # 服务端点
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
MINIO_REGION = os.getenv('MINIO_REGION')
MINIO_USE_VIRTUAL_HOST = os.getenv('MINIO_USE_VIRTUAL_HOST', 'false').lower() == 'true'  # 是否使用虚拟主机名
MINIO_USE_SSL = os.getenv('MINIO_USE_SSL', 'false').lower() == 'true'  # 是否使用 SSL 加密

## 获取MinIO客户端
def _get_minio_client() -> Minio:
    client = Minio(
        endpoint=MINIO_ENDPOINT,      # 【关键】连接 OSS 时，这里是 oss-cn-hangzhou.aliyuncs.com
        access_key=MINIO_ACCESS_KEY,  # 【关键】对应 OSS 的 AccessKey ID
        secret_key=MINIO_SECRET_KEY,  # 【关键】对应 OSS 的 AccessKey Secret
        secure=MINIO_USE_SSL,         # OSS 通常开启 SSL (True)
        region=MINIO_REGION,          # OSS 的区域，如 cn-hangzhou
    )

    if MINIO_USE_VIRTUAL_HOST:
        client.enable_virtual_style_endpoint()  # 启用虚拟样式端点，修改客户端的请求路径生成方式，将桶名称嵌入到主机名中（OSS风格）
        
    return client


## 配置阿里云消息队列
MNS_ACCESS_KEY_ID = os.getenv("MNS_ACCESS_KEY_ID")                  # 从环境变量获取MNS访问密钥ID
MNS_ACCESS_KEY_SECRET = os.getenv("MNS_ACCESS_KEY_SECRET")          # 从环境变量获取MNS访问密钥
MNS_ENDPOINT = os.getenv("MNS_ENDPOINT")                            # 从环境变量获取MNS服务端点
MNS_QUEUE_NAME = os.getenv("MNS_QUEUE_NAME", "mildoc-oss-notify")   # 从环境变量获取MNS队列名称

class OSSEventType(Enum):
    """OSS 事件类型"""
    OBJECT_CREATED = "ObjectCreated"  # 对象创建（上传）
    OBJECT_REMOVED = "ObjectRemoved"  # 对象删除

@dataclass      # 自动生成 __init__ 、 __repr__ 、 __eq__ 等方法
class OSSEvent:
    """OSS 事件数据结构"""
    event_name: str  # 事件名称，如 "ObjectCreated:PutObject"
    event_time: str  # 事件时间
    bucket_name: str  # Bucket 名称
    object_key: str  # 对象键（路径）
    object_size: Optional[int] = None  # 对象大小（字节）
    etag: Optional[str] = None  # 对象 ETag


class OSSEventNotifier:
    """OSS 事件通知管理器"""
    
    def __init__(self):
        """初始化 MNS 队列"""
        if not MNS_ENDPOINT or not MNS_ACCESS_KEY_ID or not MNS_ACCESS_KEY_SECRET:
            logger.error(f"✗ MNS 队列初始化失败: MNS_ENDPOINT or MNS_ACCESS_KEY_ID or MNS_ACCESS_KEY_SECRET is not set")
            raise Exception("MNS_ENDPOINT or MNS_ACCESS_KEY_ID or MNS_ACCESS_KEY_SECRET is not set")
        
            
        mns_account = Account(MNS_ENDPOINT, MNS_ACCESS_KEY_ID, MNS_ACCESS_KEY_SECRET)   # 创建MNS账户
        self.mildoc_queue = mns_account.get_queue(MNS_QUEUE_NAME)  # 获取MNS队列
        logger.info(f"✓ MNS 队列初始化成功 (endpoint: {MNS_ENDPOINT})")
        
    def listen_mns_queue(
        self,
        handler: Callable[[Dict], None],
        poll_interval: int = 5
    ):
        """
        监听 MNS 队列，接收 OSS 事件通知，调用处理函数处理事件。主要在增量更新中使用。
        
        :param handler: 事件处理函数，接收 Dict 对象
        :param poll_interval: 轮询间隔（秒）
        """
        if not self.mildoc_queue:
            logger.error("✗ MNS 客户端未初始化，无法监听队列")
            return
        
        logger.info(f"开始监听 MNS 队列: {MNS_QUEUE_NAME}")
        logger.info("按 Ctrl+C 停止监听")
        
        try:
            
            while True:
                try:
                    # 接收消息（长轮询，等待最多 10 秒）
                    try:
                        msg = self.mildoc_queue.receive_message_with_str_body(wait_seconds=10)  # 接收消息

                        # OSSEventNotifier实例获取到消息内容后，清理MQ中的消息，避免重复处理（同一条消息被多次接受） - MQ和接收者即生产者-消费者模型
                        self.mildoc_queue.delete_message(msg.receipt_handle)
                        logger.info(f"消息已获取并从MQ中清除，避免重复处理: {msg.receipt_handle}")

                    except Exception as e:
                        # 如果MNS队列中没有消息，会抛出异常，**等待指定时间后继续轮询**
                        if "MessageNotExist" in str(e) or "not found" in str(e).lower():
                            time.sleep(poll_interval)
                            continue
                        else:
                            logger.error(f"✗ 接收消息失败: {e}")
                            time.sleep(poll_interval)
                            continue
                    
                    if msg: # 如果获取到消息
                        try:
                            # 解析 OSS 事件
                            # MNS 消息体可能是字符串，需要先解析
                            message_body = msg.message_body # 获取消息体
                            if isinstance(message_body, str):
                                event_data = json.loads(message_body)
                            else:
                                event_data = message_body
                            
                            #logger.info(f"event_data: {event_data}")
                            logger.info(f"数据: {json.dumps(event_data, ensure_ascii=False, indent=2)}")
                            
                            if event_data:  # 如果事件数据存在，即解析成功
                                
                                # 调用处理函数
                                if handler:     # 这里的 handler 就是 self._process_event
                                    handler(event_data)
                                
                                logger.info("  ✓ 消息已处理")  # 完整执行了_process_event(event_data)
                            else:
                                logger.warning(f"⚠ 无法解析事件数据: {message_body}")
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"✗ 解析消息失败: {e}, 消息内容: {msg.message_body}")
                        except Exception as e:
                            logger.error(f"✗ 处理消息失败: {e}")
                    
                    time.sleep(poll_interval)
                    
                except KeyboardInterrupt:
                    logger.info("\n停止监听...")
                    break
                except Exception as e:
                    logger.error(f"✗ 接收消息失败: {e}")
                    time.sleep(poll_interval)
                    
        except Exception as e:
            logger.error(f"✗ 监听队列失败: {e}")
    




class OSSEventHandler:
    """OSS 事件处理器"""
    
    def __init__(self, bucket_name: str = None):
        """
        初始化监听器，MinIO客户端、文档解析器、Milvus API、Embedding工具、OSS事件通知管理器实例
        
        Args:
            bucket_name (str): 要监听的桶名称，默认从环境变量获取
        """
        self.bucket_name = bucket_name or os.getenv("MINIO_BUCKET", "mildoc-yu")
        
        # 初始化各个组件
        self.minio_client = _get_minio_client()
        
        # 初始化解析器
        logger.info("初始化解析器...")
        self.parser: SimpleObjectParser = SimpleObjectParser(minio_client=self.minio_client)
        
        # 初始化Milvus
        logger.info("初始化Milvus...")
        self.milvus_api: MilvusAPI = MilvusAPI()
        
        # 测试embedding工具
        logger.info("测试embedding工具...")
        self.embedding_tool: EmbeddingTool = EmbeddingTool()

        # OSS 事件监听
        logger.info("初始化OSS事件监听...")
        self.oss_event_notifier: OSSEventNotifier = OSSEventNotifier()

        logger.info("所有组件初始化完成！")

    def _process_event(self, event_data: Dict[str, Any]):
        """
        处理单个事件。根据事件类型调用相应的处理方法，并在处理完成后刷新 Milvus 集合
        
        Args:
            event_data (Dict[str, Any]): 事件数据
        """
        try:
            # 提取事件信息
            event_info = self._extract_event_info(event_data)
            if not event_info:
                logger.error("无法提取事件信息，跳过处理")
                return
            
            event_name = event_info.event_name
            timestamp = event_info.event_time
            
            logger.info(f"\n[{timestamp}] 收到事件: {event_name}")
            logger.info(f"对象: {event_info.bucket_name}/{event_info.object_key}")
            
            # 根据事件类型进行处理
            if OSSEventType.OBJECT_CREATED.value in event_name:
                self._handle_object_created(event_info)
            elif OSSEventType.OBJECT_REMOVED.value in event_name:
                self._handle_object_deleted(event_info)
            else:
                logger.error(f"未处理的事件类型: {event_name}")
            
            # 刷新Milvus集合
            self.milvus_api.flush_collection()
            logger.info("Milvus集合刷新完成")
        except Exception as e:
            logger.error(f"处理事件时出错: {e}")
  

    def _extract_event_info(self, event_data: Dict) -> Optional[OSSEvent]:
        """解析 OSS 事件数据"""
        try:
            # OSS 事件通知格式
            events = event_data.get("events", [])
            if not events:
                return None
            
            event = events[0]   # 获取第一个事件
            event_name = event.get("eventName", "")
            event_time = event.get("eventTime", "")
            
            # 获取对象信息和桶信息
            oss_obj = event.get("oss", {}).get("object", {})
            bucket = event.get("oss", {}).get("bucket", {})
            
            return OSSEvent(
                event_name=event_name,
                event_time=event_time,
                bucket_name=bucket.get("name", ""),
                object_key=oss_obj.get("key", ""),
                object_size=oss_obj.get("size"),    # 对象大小
                etag=oss_obj.get("etag"),
            )
        except Exception as e:
            logger.error(f"✗ 解析事件数据失败: {e}")
            return None
    
    def _handle_object_created(self, event_info: OSSEvent):
        """
        处理对象创建事件
        
        Args:
            event_info (Dict[str, Any]): 事件信息
        """
        try:
            bucket_name = event_info.bucket_name
            object_name = event_info.object_key
            
            logger.info(f"\n=== 处理新增对象: {bucket_name}/{object_name} ===")
            logger.info(f"对象大小: {event_info.object_size} 字节")
            
            # 直接调用_process_single_object方法处理
            self._process_single_object(bucket_name, object_name, force_update=True)
            
        except Exception as e:
            logger.error(f"处理对象创建事件失败: {e}")
    
    def _handle_object_deleted(self, event_info: OSSEvent):
        """
        处理对象删除事件
        
        Args:
            event_info (Dict[str, Any]): 事件信息
        """
        try:
            bucket_name = event_info.bucket_name
            object_name = event_info.object_key
            doc_path_name = object_name  # 不再包含bucket_name前缀
            
            logger.info(f"\n=== 处理删除对象: {bucket_name}/{object_name} ===")
            
            # 从Milvus中删除相关记录
            logger.info("从Milvus中查找并删除相关记录...")
            
            # 使用MilvusAPI的删除方法
            if self.milvus_api.delete_existing_document(doc_path_name):
                logger.info(f"成功删除文档记录: {doc_path_name}")
            else:
                logger.error(f"删除文档记录失败: {doc_path_name}")
            
        except Exception as e:
            logger.error(f"处理对象删除事件失败: {e}")
    
    
    def _process_single_object(self, bucket_name: str, object_name: str, force_update: bool = False):
        """
        处理单个对象（用于全量刷新和排查补漏），文档解析、向量生成、存储，连接 OSS 和 Milvus 
        
        Args:
            bucket_name (str): 桶名称
            object_name (str): 对象名称
            force_update (bool): 是否强制更新（True=全量刷新，False=排查补漏）
        
        Returns:
            bool: 处理是否成功
        """
        try:
            doc_path_name = object_name  # 不再包含bucket_name前缀
            
            # 如果是排查补漏模式，先检查是否已存在
            if not force_update:
                if self.milvus_api.check_document_exists(doc_path_name):
                    logger.info(f"  文档已存在，跳过: {object_name}")
                    return True
            
            logger.info(f"  处理文档: {object_name}")
            
            # 解析对象内容
            parse_result: Dict[str, Any] = self.parser.parse_object(bucket_name, object_name)
            
            if 'error' in parse_result:
                logger.error(f"    解析失败: {parse_result['error']}")
                return False
            
            if not parse_result['contents']:
                logger.error(f"    未提取到文本内容，跳过")
                return True
            
            logger.info(f"    解析成功，获得 {len(parse_result['contents'])} 个文本片段")
            
            # 如果是强制更新，先删除已存在的记录
            if force_update:
                self.milvus_api.delete_existing_document(doc_path_name)
            
            # 为每个文本片段生成embedding并存储到Milvus
            success_count = 0
            for i, content in enumerate(parse_result['contents']):
                try:
                    # 生成embedding向量
                    embedding_vector = self.embedding_tool.get_embedding(content)
                    if not embedding_vector:
                        logger.error(f"    片段 {i+1} embedding生成失败，跳过")
                        continue
                    
                    # 准备文档数据
                    doc_data = MilvusDocument(
                        doc_name=parse_result['doc_name'],
                        doc_path_name=parse_result['doc_path_name'],
                        doc_type=parse_result['doc_type'],
                        doc_md5=parse_result['doc_md5'],
                        doc_length=parse_result['doc_length'],
                        content=content,
                        content_vector=embedding_vector,
                        embedding_model=self.embedding_tool.model
                    )
                    
                    # 存储到Milvus（允许重复，因为我们已经处理了去重逻辑）
                    if self.milvus_api.insert_document(doc_data):
                        success_count += 1
                    else:
                        logger.error(f"    片段 {i+1} 存储失败")
                
                except Exception as e:
                    logger.error(f"    处理片段 {i+1} 时出错: {e}")
                    continue
            
            logger.info(f"    完成！成功存储 {success_count}/{len(parse_result['contents'])} 个片段")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"  处理对象失败: {e}")
            return False
    
    def full_update(self):
        """
        模式1：全量刷新 - 遍历Minio桶中的所有数据并更新到Milvus
        """
        logger.info(f"\n=== 模式1：全量刷新 ===")
        logger.info(f"正在遍历桶 '{self.bucket_name}' 中的所有对象...")
        
        try:
            # 获取桶中的所有对象
            objects = self.minio_client.list_objects(self.bucket_name, recursive=True)
            
            total_objects = 0   # 总对象数
            processed_objects = 0   # 已处理对象数
            
            for obj in objects:
                object_name = obj.object_name
                
                # 跳过文件夹
                if object_name.endswith('/'):
                    continue
                
                total_objects += 1

                logger.info(f"\n[{total_objects}] 处理对象: {object_name}")
                
                if self._process_single_object(self.bucket_name, object_name, force_update=True):
                    processed_objects += 1
                    
            self.milvus_api.flush_collection()
            
            logger.info(f"\n=== 全量刷新完成 ===")
            logger.info(f"总对象数: {total_objects}")
            logger.info(f"成功处理: {processed_objects}")
            logger.info(f"失败数量: {total_objects - processed_objects}")
            
        except Exception as e:
            logger.error(f"全量刷新失败: {e}")
    
    def backfill_update(self):
        """
        模式2：排查补漏 - 检查Milvus中不存在的文档并新增
        """
        logger.info(f"\n=== 模式2：排查补漏 ===")
        logger.info(f"正在检查桶 '{self.bucket_name}' 中缺失的文档...")
        
        try:
            # 获取桶中的所有对象
            objects = self.minio_client.list_objects(self.bucket_name, recursive=True)
            
            total_objects = 0   # 总对象数
            new_objects = 0   # 新增对象数
            existing_objects = 0   # 已存在对象数
            
            for obj in objects:
                object_name = obj.object_name
                
                # 跳过文件夹
                if object_name.endswith('/'):
                    continue
                
                total_objects += 1

                logger.info(f"\n[{total_objects}] 检查对象: {object_name}")
                
                # 检查是否已存在，对于已经存在的 99% 的文件，Milvus 只需要查一下元数据（非常快），不需要重新生成向量。
                if self.milvus_api.check_document_exists(object_name):
                    logger.info(f"  已存在，跳过")
                    existing_objects += 1
                else:
                    logger.info(f"  不存在，开始处理...")
                    if self._process_single_object(self.bucket_name, object_name, force_update=False):
                        new_objects += 1
            
            self.milvus_api.flush_collection()
            
            logger.info(f"\n=== 排查补漏完成 ===")
            logger.info(f"总对象数: {total_objects}")
            logger.info(f"已存在: {existing_objects}")
            logger.info(f"新增: {new_objects}")
            logger.info(f"失败数量: {total_objects - existing_objects - new_objects}")
            
        except Exception as e:
            logger.error(f"排查补漏失败: {e}")
    
    # 只处理发生变化的对象，不处理未变化的对象
    def start_listening(self):
        """
        模式3：增量更新 - 根据消息通知进行增量更新
        """
        logger.info(f"\n=== 模式3：增量更新 ===")
        logger.info(f"开始监听桶 '{self.bucket_name}' 的事件...")
        logger.info("按 Ctrl+C 停止监听")
        
        try:
            # 监听桶事件
            self.oss_event_notifier.listen_mns_queue(self._process_event)
                
        except KeyboardInterrupt:
            logger.info("\n监听已停止")
        except Exception as e:
            logger.error(f"监听过程中出错: {e}")
 