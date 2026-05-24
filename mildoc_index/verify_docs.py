#!/usr/bin/env python3
"""
Milvus文档验证脚本 - 检查文档是否已成功入库
"""

import sys
import os

def main():
    print("=== Milvus文档验证工具 ===")
    
    # 添加当前目录到路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from milvus_api import MilvusAPI
        
        # 初始化 Milvus API
        print("连接Milvus...")
        api = MilvusAPI()
        print("✅ Milvus连接成功")
        
        # 查询文档
        print("\n查询Milvus中的文档...")
        docs = api.client.query(
            collection_name='mildoc_collection',
            filter='',
            output_fields=['doc_name'],
            limit=50
        )
        
        if not docs:
            print("❌ Milvus中没有文档")
            print("请先运行全量刷新：")
            print("  python main.py --provider oss --mode full-refresh")
            return
        
        # 打印文档列表
        print(f"\n📚 Milvus中的文档 ({len(docs)} 个):")
        print("-" * 60)
        
        doc_names = [doc['doc_name'] for doc in docs]
        for i, doc_name in enumerate(sorted(doc_names), 1):
            print(f"{i:2d}. {doc_name}")
        
        # 检查测试文档是否存在
        print("\n🔍 检查测试文档是否存在:")
        expected_docs = [
            "【全球计算联盟GCC】向量数据库白皮书.pdf",
            "劳动法.pdf", 
            "财务管理文档.pdf",
            "人事管理流程.docx"
        ]
        
        found_count = 0
        for doc in expected_docs:
            if doc in doc_names:
                print(f"  ✅ {doc}")
                found_count += 1
            else:
                print(f"  ❌ {doc}")
        
        print("\n📊 统计结果:")
        print(f"  期望文档数: {len(expected_docs)}")
        print(f"  已入库数: {found_count}")
        
        if found_count == len(expected_docs):
            print("\n🎉 所有测试文档已成功入库！可以开始评估了。")
        else:
            print(f"\n⚠️ 还有 {len(expected_docs) - found_count} 个文档未入库")
            print("请先上传文档到OSS，然后运行：")
            print("  python main.py --provider oss --mode full-refresh")
            
    except Exception as e:
        print(f"\n❌ 出错了: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
