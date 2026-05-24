"""Chunk级别指标计算模块 - 实现真正的文本块级别评估"""

from typing import List, Dict, Any, Optional
from data_loader import ChunkInfo


def normalize_chunk_id(chunk: Dict[str, Any]) -> str:
    """
    标准化Chunk ID，用于比较
    
    Args:
        chunk: Chunk信息字典
        
    Returns:
        标准化的chunk标识
    """
    # 优先使用chunk_id
    if 'chunk_id' in chunk and chunk['chunk_id']:
        return chunk['chunk_id']
    
    # 其次使用内容哈希
    if 'content' in chunk:
        return str(hash(chunk['content']))
    
    # 最后使用文档名+位置
    doc_name = chunk.get('doc_name', '')
    position = chunk.get('position', 0)
    return f"{doc_name}_{position}"


def calculate_chunk_recall(expected_chunks: List[Dict], retrieved_chunks: List[Dict]) -> float:
    """
    计算Chunk级别的召回率
    
    Recall = 检索到的相关chunk数 / 总相关chunk数
    
    Args:
        expected_chunks: 期望的相关chunk列表
        retrieved_chunks: 实际检索到的chunk列表
        
    Returns:
        召回率 (0-1)
    """
    if not expected_chunks:
        return 1.0
    
    if not retrieved_chunks:
        return 0.0
    
    # 构建期望chunk的ID集合
    expected_chunk_ids = set()
    for chunk in expected_chunks:
        chunk_id = normalize_chunk_id(chunk)
        expected_chunk_ids.add(chunk_id)
    
    # 统计检索到的相关chunk数量
    retrieved_relevant_count = 0
    retrieved_chunk_ids = set()
    
    for chunk in retrieved_chunks:
        chunk_id = normalize_chunk_id(chunk)
        # 避免重复计数
        if chunk_id not in retrieved_chunk_ids and chunk_id in expected_chunk_ids:
            retrieved_relevant_count += 1
            retrieved_chunk_ids.add(chunk_id)
    
    return retrieved_relevant_count / len(expected_chunk_ids)


def calculate_chunk_precision(expected_chunks: List[Dict], retrieved_chunks: List[Dict]) -> float:
    """
    计算Chunk级别的准确率
    
    Precision = 检索到的相关chunk数 / 检索到的总chunk数
    
    Args:
        expected_chunks: 期望的相关chunk列表
        retrieved_chunks: 实际检索到的chunk列表
        
    Returns:
        准确率 (0-1)
    """
    if not retrieved_chunks:
        return 0.0
    
    # 构建期望chunk的ID集合
    expected_chunk_ids = set()
    for chunk in expected_chunks:
        chunk_id = normalize_chunk_id(chunk)
        expected_chunk_ids.add(chunk_id)
    
    # 统计检索到的相关chunk数量
    relevant_count = 0
    for chunk in retrieved_chunks:
        chunk_id = normalize_chunk_id(chunk)
        if chunk_id in expected_chunk_ids:
            relevant_count += 1
    
    return relevant_count / len(retrieved_chunks)


def calculate_f1(recall: float, precision: float) -> float:
    """
    计算F1值
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        recall: 召回率
        precision: 准确率
        
    Returns:
        F1值 (0-1)
    """
    if recall + precision == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_recall_at_k(expected_chunks: List[Dict], retrieved_chunks: List[Dict], k: int) -> float:
    """
    计算Recall@K - Top K个结果中的召回率
    
    Args:
        expected_chunks: 期望的相关chunk列表
        retrieved_chunks: 实际检索到的chunk列表（按相关性排序）
        k: Top K
        
    Returns:
        Recall@K (0-1)
    """
    if not expected_chunks:
        return 1.0
    
    if not retrieved_chunks:
        return 0.0
    
    # 只考虑Top K个结果
    top_k_chunks = retrieved_chunks[:k]
    
    # 构建期望chunk的ID集合
    expected_chunk_ids = set()
    for chunk in expected_chunks:
        chunk_id = normalize_chunk_id(chunk)
        expected_chunk_ids.add(chunk_id)
    
    # 统计Top K中检索到的相关chunk数量
    retrieved_relevant_count = 0
    retrieved_chunk_ids = set()
    
    for chunk in top_k_chunks:
        chunk_id = normalize_chunk_id(chunk)
        if chunk_id not in retrieved_chunk_ids and chunk_id in expected_chunk_ids:
            retrieved_relevant_count += 1
            retrieved_chunk_ids.add(chunk_id)
    
    return retrieved_relevant_count / len(expected_chunk_ids)


def calculate_mrr(expected_chunks: List[Dict], retrieved_chunks: List[Dict]) -> float:
    """
    计算平均倒数排名(MRR)
    
    MRR = 1 / rank_of_first_relevant_chunk
    
    Args:
        expected_chunks: 期望的相关chunk列表
        retrieved_chunks: 实际检索到的chunk列表（按相关性排序）
        
    Returns:
        MRR值 (0-1)
    """
    if not expected_chunks or not retrieved_chunks:
        return 0.0
    
    # 构建期望chunk的ID集合
    expected_chunk_ids = set()
    for chunk in expected_chunks:
        chunk_id = normalize_chunk_id(chunk)
        expected_chunk_ids.add(chunk_id)
    
    # 找到第一个相关chunk的排名
    for rank, chunk in enumerate(retrieved_chunks, 1):
        chunk_id = normalize_chunk_id(chunk)
        if chunk_id in expected_chunk_ids:
            return 1.0 / rank
    
    return 0.0


def calculate_ndcg_at_k(expected_chunks: List[Dict], retrieved_chunks: List[Dict], k: int = 5) -> float:
    """
    计算NDCG@K（归一化折损累计增益）
    
    衡量检索结果的排序质量
    
    Args:
        expected_chunks: 期望的相关chunk列表
        retrieved_chunks: 实际检索到的chunk列表（按相关性排序）
        k: Top K
        
    Returns:
        NDCG@K (0-1)
    """
    if not expected_chunks or not retrieved_chunks:
        return 0.0
    
    # 构建期望chunk的ID集合
    expected_chunk_ids = set()
    for chunk in expected_chunks:
        chunk_id = normalize_chunk_id(chunk)
        expected_chunk_ids.add(chunk_id)
    
    # 计算DCG@K
    dcg = 0.0
    for i, chunk in enumerate(retrieved_chunks[:k]):
        chunk_id = normalize_chunk_id(chunk)
        relevance = 1.0 if chunk_id in expected_chunk_ids else 0.0
        dcg += relevance / (i + 1)  # log2(i+2) 简化为 i+1
    
    # 计算IDCG@K（理想情况下的DCG）
    ideal_retrieved = [1.0] * min(len(expected_chunks), k) + [0.0] * max(0, k - len(expected_chunks))
    idcg = sum(rel / (i + 1) for i, rel in enumerate(ideal_retrieved[:k]))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_chunk_level_metrics(expected_chunks: List[Dict], retrieved_chunks: List[Dict]) -> Dict[str, float]:
    """
    综合计算所有Chunk级别指标
    
    Args:
        expected_chunks: 期望的相关chunk列表
        retrieved_chunks: 实际检索到的chunk列表
        
    Returns:
        指标字典
    """
    recall = calculate_chunk_recall(expected_chunks, retrieved_chunks)
    precision = calculate_chunk_precision(expected_chunks, retrieved_chunks)
    f1 = calculate_f1(recall, precision)
    mrr = calculate_mrr(expected_chunks, retrieved_chunks)
    
    # 计算不同K值的Recall@K
    recall_at_1 = calculate_recall_at_k(expected_chunks, retrieved_chunks, 1)
    recall_at_3 = calculate_recall_at_k(expected_chunks, retrieved_chunks, 3)
    recall_at_5 = calculate_recall_at_k(expected_chunks, retrieved_chunks, 5)
    
    # 计算NDCG@5
    ndcg_at_5 = calculate_ndcg_at_k(expected_chunks, retrieved_chunks, 5)
    
    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'mrr': mrr,
        'recall_at_1': recall_at_1,
        'recall_at_3': recall_at_3,
        'recall_at_5': recall_at_5,
        'ndcg_at_5': ndcg_at_5
    }


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    聚合多个查询的指标，计算平均值
    
    Args:
        all_metrics: 每个查询的指标列表
        
    Returns:
        平均指标字典
    """
    if not all_metrics:
        return {}
    
    n = len(all_metrics)
    aggregated = {}
    
    # 获取所有指标键
    keys = all_metrics[0].keys()
    
    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregated[f'avg_{key}'] = sum(values) / len(values)
    
    return aggregated
