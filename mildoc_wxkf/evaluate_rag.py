#!/usr/bin/env python3
"""
在 mildoc_wxkf 目录中直接评估 RAG 系统的脚本
"""

import sys
import os
import json
import time
import argparse
from typing import List, Dict, Any


def evaluate_single_question(question: str) -> Dict:
    """
    评估单个问题
    
    Args:
        question: 测试问题
        
    Returns:
        评估结果字典
    """
    from rag_service import get_rag_service
    
    rag_service = get_rag_service()
    if not rag_service:
        return {
            "question": question,
            "retrieved_docs": [],
            "answer": "",
            "success": False,
            "error_message": "RAG服务初始化失败",
            "latency": 0.0
        }
    
    start_time = time.time()
    
    try:
        response = rag_service.query_service(question)
        
        retrieved_docs = []
        if response.source_documents:
            for doc in response.source_documents:
                retrieved_docs.append({
                    "doc_name": doc.doc_name,
                    "content": doc.content_preview,
                    "score": doc.similarity_score if doc.similarity_score else 0.0
                })
        
        latency = time.time() - start_time
        
        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "answer": response.content,
            "success": response.success,
            "error_message": response.error_message,
            "latency": latency
        }
        
    except Exception as e:
        latency = time.time() - start_time
        return {
            "question": question,
            "retrieved_docs": [],
            "answer": "",
            "success": False,
            "error_message": str(e),
            "latency": latency
        }


def evaluate_multiple_questions(questions: List[str]) -> List[Dict]:
    """
    评估多个问题
    
    Args:
        questions: 测试问题列表
        
    Returns:
        评估结果列表
    """
    results = []
    for question in questions:
        result = evaluate_single_question(question)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="评估 mildoc_wxkf RAG 系统")
    parser.add_argument(
        "--question",
        type=str,
        help="单个测试问题"
    )
    parser.add_argument(
        "--questions",
        type=str,
        help="多个测试问题（JSON数组格式）"
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="输出JSON格式结果"
    )
    
    args = parser.parse_args()
    
    if args.question:
        result = evaluate_single_question(args.question)
        if args.output_json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"问题: {result['question']}")
            print(f"成功: {result['success']}")
            print(f"延迟: {result['latency']:.3f}秒")
            print(f"检索到文档数: {len(result['retrieved_docs'])}")
            for i, doc in enumerate(result['retrieved_docs'], 1):
                print(f"  {i}. {doc['doc_name']} (score: {doc['score']:.3f})")
            if result['answer']:
                print(f"回答: {result['answer'][:100]}...")
    
    elif args.questions:
        try:
            questions = json.loads(args.questions)
            results = evaluate_multiple_questions(questions)
            if args.output_json:
                print(json.dumps(results, ensure_ascii=False))
            else:
                for i, result in enumerate(results, 1):
                    print(f"\n[{i}/{len(results)}]")
                    print(f"问题: {result['question']}")
                    print(f"成功: {result['success']}")
                    print(f"延迟: {result['latency']:.3f}秒")
                    print(f"检索到文档数: {len(result['retrieved_docs'])}")
        except json.JSONDecodeError:
            print("错误：questions参数不是有效的JSON数组")
    
    else:
        # 默认测试
        test_questions = [
            "什么是向量数据库？",
            "Milvus有哪些特点？",
            "如何使用RAG系统？"
        ]
        
        print("=== 评估 mildoc_wxkf RAG 系统 ===")
        print(f"测试问题数量: {len(test_questions)}")
        
        results = evaluate_multiple_questions(test_questions)
        
        print("\n=== 评估结果 ===")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}/{len(results)}]")
            print(f"问题: {result['question']}")
            print(f"成功: {result['success']}")
            print(f"延迟: {result['latency']:.3f}秒")
            print(f"检索到文档数: {len(result['retrieved_docs'])}")
            for j, doc in enumerate(result['retrieved_docs'], 1):
                print(f"  {j}. {doc['doc_name']} (score: {doc['score']:.3f})")
            if result['answer']:
                print(f"回答: {result['answer'][:100]}...")


if __name__ == "__main__":
    main()
