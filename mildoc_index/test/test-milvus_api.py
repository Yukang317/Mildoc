import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
from unittest.mock import Mock, patch
from mildoc_index.milvus_api import MilvusAPI, MilvusDocument

class TestMilvusAPI:
    """MilvusAPI单元测试"""
    
    @pytest.fixture
    def mock_milvus_client(self):   # 这个fixture使用 patch 来模拟 MilvusClient ，避免实际连接数据库
        """模拟Milvus客户端"""
        # 临时替换MilvusClient类
        with patch('mildoc_index.milvus_api.MilvusClient') as mock_client_class:
            mock_client = Mock()    # 创建一个模拟对象
            mock_client_class.return_value = mock_client    # 当MilvusClient被实例化时，返回这个模拟对象
            
            # 模拟集合检查
            mock_client.has_collection.return_value = True
            
            # 模拟索引检查
            mock_client.list_indexes.return_value = ['vector_index']
            
            # 模拟加载集合
            mock_client.load_collection.return_value = None
            
            # 模拟查询
            mock_client.query.return_value = []
            
            # 模拟删除
            mock_client.delete.return_value = {'delete_count': 1}
            
            # 模拟插入
            mock_client.insert.return_value = {'insert_count': 1}
            
            # 模拟刷新
            mock_client.flush.return_value = None
            
            # 模拟搜索
            mock_client.search.return_value = [[
                Mock(
                    id=1,
                    score=0.95,
                    doc_name='test.pdf',
                    doc_path_name='/test/docs/test.pdf',
                    doc_type='pdf',
                    content='测试内容',
                    embedding_model='text-embedding-ada-002'
                )
            ]]
            
            # 模拟获取集合信息
            mock_client.describe_collection.return_value = {
                'collection_name': 'documents',
                'schema': {'fields': [{'name': 'id'}, {'name': 'content_vector'}]}
            }
            
            yield mock_client
    
    @pytest.fixture
    def milvus_api(self, mock_milvus_client):
        """创建MilvusAPI实例"""
        # 环境变量已在测试环境中配置
        return MilvusAPI()
    
    def test_initialization(self, milvus_api):
        """测试初始化"""
        assert milvus_api is not None
    
    def test_milvus_api_initialization_with_env(self, monkeypatch):
        """测试环境变量配置和初始化"""
        # 设置环境变量
        monkeypatch.setenv('MILVUS_HOST', 'localhost')
        monkeypatch.setenv('MILVUS_PORT', '19530')
        monkeypatch.setenv('MILVUS_USER', '')
        monkeypatch.setenv('MILVUS_PASSWORD', '')
        monkeypatch.setenv('MILVUS_DATABASE', 'default')
        monkeypatch.setenv('MILVUS_COLLECTION', 'documents')
        monkeypatch.setenv('MILVUS_INDEX_NAME', 'vector_index')
        monkeypatch.setenv('MILVUS_VECTOR_DIM', '768')
        
        # 创建实例
        api = MilvusAPI()
        assert api is not None
    
    def test_check_document_exists(self, milvus_api, mock_milvus_client):
        """测试检查文档是否存在"""
        # 测试文档不存在
        mock_milvus_client.query.return_value = []
        assert not milvus_api.check_document_exists('/test/docs/test.pdf')
        
        # 测试文档存在
        mock_milvus_client.query.return_value = [{'id': 1}]
        assert milvus_api.check_document_exists('/test/docs/test.pdf')
    
    def test_delete_existing_document(self, milvus_api, mock_milvus_client):
        """测试删除文档"""
        # 测试空路径
        assert not milvus_api.delete_existing_document('')
        
        # 测试正常删除
        assert milvus_api.delete_existing_document('/test/docs/test.pdf')
    
    def test_insert_document(self, milvus_api, mock_milvus_client):
        """测试插入文档"""
        # 创建测试文档
        test_doc = MilvusDocument(
            doc_name='test.pdf',
            doc_path_name='/test/docs/test.pdf',
            doc_type='pdf',
            doc_md5='test_md5',
            doc_length=1024,
            content='测试内容',
            content_vector=[0.0] * 768,
            embedding_model='text-embedding-ada-002'
        )
        
        assert milvus_api.insert_document(test_doc)
    
    def test_search_similar_documents(self, milvus_api, mock_milvus_client):
        """测试搜索相似文档"""
        query_vector = [0.0] * 768
        results = milvus_api.search_similar_documents(query_vector)
        assert len(results) > 0
    
    def test_flush_collection(self, milvus_api, mock_milvus_client):
        """测试刷新集合"""
        assert milvus_api.flush_collection()
    
    def test_get_collection_info(self, milvus_api, mock_milvus_client):
        """测试获取集合信息"""
        info = milvus_api.get_collection_info()
        assert info is not None
        assert 'collection_name' in info