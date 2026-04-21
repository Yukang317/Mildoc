# Minio对象存储事件处理器负责监听 MinIO 对象存储的事件（如文件上传、删除），并触发相应的处理流程。
#   - 解决了文档变更的实时感知和处理问题，确保知识库能够及时更新，反映最新的文档状态。
# 交互关系 ：
#   - 与 MinIO 交互：监听对象存储事件
#   - 与 SimpleObjectParser 交互：解析上传的文档
#   - 与 EmbeddingTool 交互：生成文档片段的向量表示
#   - 与 MilvusAPI 交互：将解析后的文档存储到 Milvus 向量数据库
import json
from datetime import datetime
from typing import Dict, Any
import os
from dotenv import load_dotenv

from minio import Minio
from parser.simple_object_parser import SimpleObjectParser
from embedding import EmbeddingTool
from milvus_api import MilvusAPI, MilvusDocument
from logger.logging import setup_logging

load_dotenv()

logger = setup_logging()


# Minio 配置信息。
MINIO_BUCKET = os.getenv('MINIO_BUCKET')            # 桶名称
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')        # 服务端点
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')    # 访问密钥
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')    # 密钥
MINIO_REGION = os.getenv('MINIO_REGION')            # 区域
MINIO_USE_VIRTUAL_HOST = os.getenv('MINIO_USE_VIRTUAL_HOST', 'false').lower() == 'true'   # 是否使用虚拟主机样式的端点
MINIO_USE_SSL = os.getenv('MINIO_USE_SSL', 'false').lower() == 'true'                     # 是否使用SSL

# 创建并返回MinIO客户端实例
def _get_minio_client() -> Minio:
    client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_USE_SSL,
        region=MINIO_REGION,
    )

    # 如果使用虚拟主机
    if MINIO_USE_VIRTUAL_HOST:  
        client.enable_virtual_style_endpoint()  # 启用虚拟主机样式的端点
        
    return client

# Minio事件处理器类
class MinioEventHandler:
    """Minio事件监听器"""
    
    def __init__(self, bucket_name: str = None):
        """
        初始化监听器
        
        Args:
            bucket_name (str): 要监听的桶名称，默认从环境变量获取
        """
        self.bucket_name = bucket_name or os.getenv("MINIO_BUCKET", "mildoc")
        
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

        logger.info("所有组件初始化完成！")
    
    # 接收事件数据
    def _extract_event_info(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从事件数据中提取关键信息
        
        Args:
            event_data (Dict[str, Any]): 事件数据
            
        Returns:
            Dict[str, Any]: 提取的信息
        """
        try:

            #logger.info(f"event_data: {event_data}")
            logger.info(f"数据: {json.dumps(event_data, ensure_ascii=False, indent=2)}")    # 原始数据，非ASCII字符可显示，输出缩进2个空格

            record = event_data.get('Records', [{}])[0]     # 获取第一个记录
            s3_info = record.get('s3', {})  # s3中包含了对象存储相关的信息，节点中拿到以下内容
            
            return {
                'event_name': record.get('eventName', ''),  # 事件名称
                'event_time': record.get('eventTime', ''),  # 事件时间
                'bucket_name': s3_info.get('bucket', {}).get('name', ''),  # 桶名称
                'object_name': s3_info.get('object', {}).get('key', ''),  # 对象名称
                'object_size': s3_info.get('object', {}).get('size', 0),  # 对象大小
                'content_type': s3_info.get('object', {}).get('contentType', ''),   # 对象内容类型
                'etag': s3_info.get('object', {}).get('eTag', ''),  # 对象ETAG
            }
        except Exception as e:
            logger.error(f"提取事件信息失败: {e}")
            return {}
    
    def _handle_object_created(self, event_info: Dict[str, Any]):
        """
        处理对象创建事件
        
        Args:
            event_info (Dict[str, Any]): 事件信息
        """
        try:
            bucket_name = event_info['bucket_name']
            object_name = event_info['object_name']
            
            logger.info(f"\n=== 处理新增对象: {bucket_name}/{object_name} ===")
            logger.info(f"对象大小: {event_info['object_size']} 字节")
            logger.info(f"内容类型: {event_info['content_type']}")
            
            # 直接调用_process_single_object方法（处理单个对象）处理，强制更新
            self._process_single_object(bucket_name, object_name, force_update=True)
            
        except Exception as e:
            logger.error(f"处理对象创建事件失败: {e}")
    
    def _handle_object_deleted(self, event_info: Dict[str, Any]):
        """
        处理对象删除事件
        
        Args:
            event_info (Dict[str, Any]): 事件信息
        """
        try:
            bucket_name = event_info['bucket_name']
            object_name = event_info['object_name']
            doc_path_name = object_name  # 不再包含bucket_name前缀，使用相对路径
            
            logger.info(f"\n=== 处理删除对象: {bucket_name}/{object_name} ===")
            
            # 从Milvus中删除相关记录
            logger.info("从Milvus中查找并删除相关记录...")
            
            # 使用MilvusAPI的删除方法，当 MinIO 中删除对象时，需要同步从 Milvus 向量数据库中删除对应的文档记录
            if self.milvus_api.delete_existing_document(doc_path_name):
                logger.info(f"成功删除文档记录: {doc_path_name}")
            else:
                logger.error(f"删除文档记录失败: {doc_path_name}")
            
        except Exception as e:
            logger.error(f"处理对象删除事件失败: {e}")
    
    def _process_event(self, event_data: Dict[str, Any]):
        """
        处理单个事件，根据事件类型调用相应的处理方法
        
        Args:
            event_data (Dict[str, Any]): 事件数据
        """
        try:
            # 提取事件信息（本文件中的方法）
            event_info = self._extract_event_info(event_data)
            if not event_info:
                logger.error("无法提取事件信息，跳过处理")
                return
            
            event_name = event_info['event_name']
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")    # 获取当前时间
            
            logger.info(f"\n[{timestamp}] 收到事件: {event_name}")
            logger.info(f"对象: {event_info['bucket_name']}/{event_info['object_name']}")
            
            # 根据事件类型进行处理
            if 'ObjectCreated' in event_name:
                self._handle_object_created(event_info)
            elif 'ObjectRemoved' in event_name:
                self._handle_object_deleted(event_info)
            else:
                logger.error(f"未处理的事件类型: {event_name}")
                
        except Exception as e:
            logger.error(f"处理事件时出错: {e}")
    
    def _process_single_object(self, bucket_name: str, object_name: str, force_update: bool = False):
        """
        处理单个对象（用于全量刷新和排查补漏），文档解析、向量生成和存储，连接 MinIO 和 Milvus
        
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
            if not force_update:    # 不是强制更新
                if self.milvus_api.check_document_exists(doc_path_name):
                    logger.info(f"  文档已存在，跳过: {object_name}")
                    return True
            
            logger.info(f"  处理文档: {object_name}")
            
            # 解析对象内容
            parse_result = self.parser.parse_object(bucket_name, object_name)
            
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
            success_count = 0   # 成功数目
            for i, content in enumerate(parse_result['contents']):
                try:
                    # 生成embedding向量
                    embedding_vector = self.embedding_tool.get_embedding(content)
                    if not embedding_vector:
                        logger.error(f"    片段 {i+1} embedding生成失败，跳过")
                        continue
                    
                    # 准备文档数据
                    doc_data = MilvusDocument(  # 创建文档数据
                        doc_name=parse_result['doc_name'],  # 文档名称
                        doc_path_name=parse_result['doc_path_name'],  # 文档路径名称
                        doc_type=parse_result['doc_type'],  # 文档类型
                        doc_md5=parse_result['doc_md5'],
                        doc_length=parse_result['doc_length'],
                        content=content,  # 文本内容
                        content_vector=embedding_vector,    # 文本内容向量
                        embedding_model=self.embedding_tool.model   # 嵌入模型
                    )
                    
                    # 存储到Milvus（允许重复，因为我们已经处理了去重逻辑）
                    #   在 _process_single_object 方法中，有以下去重逻辑：
                    #        1. 排查补漏模式下：先检查文档是否存在，存在则跳过
                    #        2. 强制更新模式下：先删除已存在的记录，再重新插入   
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
            # 获取桶中的所有对象（递归列出）
            objects = self.minio_client.list_objects(self.bucket_name, recursive=True)
            
            total_objects = 0   # 总对象数
            processed_objects = 0   # 已处理对象数
            
            for obj in objects:     # 遍历对象并获取名称
                object_name = obj.object_name
                
                # 跳过文件夹
                if object_name.endswith('/'):
                    continue

                total_objects += 1
                
                logger.info(f"\n[{total_objects}] 处理对象: {object_name}")
                
                if self._process_single_object(self.bucket_name, object_name, force_update=True):
                    processed_objects += 1
            
            # 刷新Milvus集合，保持Milvus中的数据于MinIO一致。将内存中的数据持久化到磁盘
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
            
            total_objects = 0
            new_objects = 0
            existing_objects = 0
            
            for obj in objects:
                object_name = obj.object_name
                
                # 跳过文件夹
                if object_name.endswith('/'):
                    continue
                
                total_objects += 1

                logger.info(f"\n[{total_objects}] 检查对象: {object_name}")
                
                # 检查是否已存在
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
    
    def start_listening(self):
        """
        模式3：增量更新，监听桶事件并实时处理，确保 Milvus 中的数据能够及时反映 MinIO 中的变化。
        """
        logger.info(f"\n=== 模式3：增量更新 ===")
        logger.info(f"开始监听桶 '{self.bucket_name}' 的事件...")
        logger.info("按 Ctrl+C 停止监听")
        
        try:
            # 监听桶事件
            events = self.minio_client.listen_bucket_notification(
                bucket_name=self.bucket_name,
                events=['s3:ObjectCreated:*', 's3:ObjectRemoved:*'] # 监听的事件类型，包括对象创建和删除
            )
            
            for event in events:
                try:
                    if event:
                        # 解析事件数据
                        if isinstance(event, bytes):
                            event_data = json.loads(event.decode('utf-8'))
                        elif isinstance(event, str):
                            event_data = json.loads(event)
                        elif isinstance(event, dict):
                            event_data = event
                        else:
                            logger.error(f"未知的事件数据类型: {type(event)}")
                            continue
                        
                        # 处理事件
                        self._process_event(event_data)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"解析事件数据失败: {e}")
                except Exception as e:
                    logger.error(f"处理事件失败: {e}")
                    
        except KeyboardInterrupt:
            logger.info("\n监听已停止")
        except Exception as e:
            logger.error(f"监听过程中出错: {e}")
 