"""快速验证脚本 - 测试Chunk级别指标计算是否正确"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from chunk_metrics import (
    calculate_chunk_recall,
    calculate_chunk_precision,
    calculate_f1,
    calculate_mrr,
    calculate_ndcg_at_k,
    evaluate_chunk_level_metrics
)


def test_basic_chunk_metrics():
    """测试基本的Chunk级别指标计算"""
    print("="*60)
    print("测试1: 基本Chunk级别指标计算")
    print("="*60)
    
    # 模拟数据
    expected_chunks = [
        {'chunk_id': 'chunk_1', 'content': '内容1', 'doc_name': 'doc1.pdf'},
        {'chunk_id': 'chunk_2', 'content': '内容2', 'doc_name': 'doc1.pdf'},
        {'chunk_id': 'chunk_3', 'content': '内容3', 'doc_name': 'doc2.pdf'},
    ]
    
    retrieved_chunks = [
        {'chunk_id': 'chunk_1', 'content': '内容1', 'doc_name': 'doc1.pdf', 'score': 0.9},
        {'chunk_id': 'chunk_4', 'content': '内容4', 'doc_name': 'doc3.pdf', 'score': 0.8},
        {'chunk_id': 'chunk_2', 'content': '内容2', 'doc_name': 'doc1.pdf', 'score': 0.7},
        {'chunk_id': 'chunk_5', 'content': '内容5', 'doc_name': 'doc4.pdf', 'score': 0.6},
        {'chunk_id': 'chunk_3', 'content': '内容3', 'doc_name': 'doc2.pdf', 'score': 0.5},
    ]
    
    # 计算指标
    recall = calculate_chunk_recall(expected_chunks, retrieved_chunks)
    precision = calculate_chunk_precision(expected_chunks, retrieved_chunks)
    f1 = calculate_f1(recall, precision)
    mrr = calculate_mrr(expected_chunks, retrieved_chunks)
    ndcg = calculate_ndcg_at_k(expected_chunks, retrieved_chunks, k=5)
    
    print(f"\n期望的chunks: {len(expected_chunks)}个")
    print(f"检索到的chunks: {len(retrieved_chunks)}个")
    print(f"相关的chunks: 3个 (chunk_1, chunk_2, chunk_3)")
    print(f"\n召回率 (Recall):     {recall:.4f} (期望: 1.0000)")
    print(f"准确率 (Precision):  {precision:.4f} (期望: 0.6000)")
    print(f"F1值:                {f1:.4f} (期望: 0.7500)")
    print(f"MRR:                 {mrr:.4f} (期望: 1.0000, 第一个就是相关的)")
    print(f"NDCG@5:              {ndcg:.4f}")
    
    # 验证结果
    assert abs(recall - 1.0) < 0.01, f"Recall错误: {recall}"
    assert abs(precision - 0.6) < 0.01, f"Precision错误: {precision}"
    assert abs(f1 - 0.75) < 0.01, f"F1错误: {f1}"
    assert abs(mrr - 1.0) < 0.01, f"MRR错误: {mrr}"
    
    print("\n✅ 测试1通过！\n")


def test_partial_recall():
    """测试部分召回的情况"""
    print("="*60)
    print("测试2: 部分召回情况")
    print("="*60)
    
    expected_chunks = [
        {'chunk_id': 'chunk_1', 'content': '内容1', 'doc_name': 'doc1.pdf'},
        {'chunk_id': 'chunk_2', 'content': '内容2', 'doc_name': 'doc1.pdf'},
        {'chunk_id': 'chunk_3', 'content': '内容3', 'doc_name': 'doc2.pdf'},
        {'chunk_id': 'chunk_4', 'content': '内容4', 'doc_name': 'doc2.pdf'},
    ]
    
    retrieved_chunks = [
        {'chunk_id': 'chunk_5', 'content': '无关内容', 'doc_name': 'doc3.pdf', 'score': 0.9},
        {'chunk_id': 'chunk_1', 'content': '内容1', 'doc_name': 'doc1.pdf', 'score': 0.8},
        {'chunk_id': 'chunk_6', 'content': '无关内容', 'doc_name': 'doc4.pdf', 'score': 0.7},
    ]
    
    metrics = evaluate_chunk_level_metrics(expected_chunks, retrieved_chunks)
    
    print(f"\n期望的chunks: {len(expected_chunks)}个")
    print(f"检索到的chunks: {len(retrieved_chunks)}个")
    print(f"相关的chunks: 1个 (只召回了chunk_1)")
    print(f"\n召回率 (Recall):     {metrics['recall']:.4f} (期望: 0.2500)")
    print(f"准确率 (Precision):  {metrics['precision']:.4f} (期望: 0.3333)")
    print(f"F1值:                {metrics['f1']:.4f}")
    print(f"MRR:                 {metrics['mrr']:.4f} (期望: 0.5000, 排名第2)")
    print(f"Recall@1:            {metrics['recall_at_1']:.4f}")
    print(f"Recall@3:            {metrics['recall_at_3']:.4f}")
    print(f"NDCG@5:              {metrics['ndcg_at_5']:.4f}")
    
    assert abs(metrics['recall'] - 0.25) < 0.01, f"Recall错误: {metrics['recall']}"
    assert abs(metrics['precision'] - 0.3333) < 0.01, f"Precision错误: {metrics['precision']}"
    assert abs(metrics['mrr'] - 0.5) < 0.01, f"MRR错误: {metrics['mrr']}"
    
    print("\n✅ 测试2通过！\n")


def test_no_relevant_chunks():
    """测试没有相关chunk被召回的情况"""
    print("="*60)
    print("测试3: 无相关Chunk被召回")
    print("="*60)
    
    expected_chunks = [
        {'chunk_id': 'chunk_1', 'content': '内容1', 'doc_name': 'doc1.pdf'},
    ]
    
    retrieved_chunks = [
        {'chunk_id': 'chunk_99', 'content': '无关内容', 'doc_name': 'doc99.pdf', 'score': 0.9},
    ]
    
    metrics = evaluate_chunk_level_metrics(expected_chunks, retrieved_chunks)
    
    print(f"\n召回率 (Recall):     {metrics['recall']:.4f} (期望: 0.0000)")
    print(f"准确率 (Precision):  {metrics['precision']:.4f} (期望: 0.0000)")
    print(f"F1值:                {metrics['f1']:.4f} (期望: 0.0000)")
    print(f"MRR:                 {metrics['mrr']:.4f} (期望: 0.0000)")
    
    assert metrics['recall'] == 0.0, f"Recall应为0"
    assert metrics['precision'] == 0.0, f"Precision应为0"
    assert metrics['f1'] == 0.0, f"F1应为0"
    assert metrics['mrr'] == 0.0, f"MRR应为0"
    
    print("\n✅ 测试3通过！\n")


def test_data_loader_compatibility():
    """测试数据加载器的兼容性"""
    print("="*60)
    print("测试4: 数据加载器兼容性")
    print("="*60)
    
    from data_loader import TestSample, ChunkInfo
    
    # 测试ChunkInfo
    chunk = ChunkInfo(
        chunk_id="test_chunk_1",
        content="测试内容",
        doc_name="test.pdf"
    )
    print(f"\nChunkInfo创建成功: {chunk.chunk_id}")
    
    # 测试TestSample with expected_chunks
    sample = TestSample(
        question="测试问题",
        ground_truth="标准答案",
        expected_chunks=[chunk],
        metadata={"category": "测试"}
    )
    print(f"TestSample创建成功: {sample.question}")
    print(f"Expected chunks数量: {len(sample.expected_chunks)}")
    
    # 转换为字典
    sample_dict = sample.dict()
    print(f"转换为字典成功，包含字段: {list(sample_dict.keys())}")
    
    print("\n✅ 测试4通过！\n")


if __name__ == "__main__":
    try:
        test_basic_chunk_metrics()
        test_partial_recall()
        test_no_relevant_chunks()
        test_data_loader_compatibility()
        
        print("="*60)
        print("🎉 所有测试通过！Chunk级别指标计算正常工作。")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
