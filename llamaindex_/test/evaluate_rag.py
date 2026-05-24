#!/usr/bin/env python3
"""
RAG系统评估脚本 - 检测召回率和准确率

评估指标：
1. 检索召回率(Recall@k)：衡量检索器能否召回相关文档
2. 检索精确率(Precision@k)：衡量检索结果的准确性
3. 检索F1分数：召回率和精确率的调和平均
4. 回答准确率(Answer Accuracy)：衡量生成回答的正确性
5. 回答忠实度(Faithfulness)：衡量回答是否基于检索到的文档

使用方法：
1. 准备测试数据集(test_cases.json)
2. 运行脚本：python evaluate_rag.py
3. 查看评估报告
"""

import sys
import os

# 添加项目根目录到路径，解决模块导入问题
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
from typing import List, Dict, Any, Tuple
from pathlib import Path
from core.application import RAGApplication
from utils.logger import setup_logger
from llama_index.core.settings import Settings
from llama_index.llms.dashscope import DashScope

# 初始化日志
logger = setup_logger(__name__)

# 配置评估参数
EVAL_CONFIG = {
    "top_k": 5,                    # 检索返回数量
    "rerank_top_k": 3,             # 重排序后保留数量
    "temperature": 0.1,            # LLM温度
    "max_tokens": 500,             # 最大生成token
    "test_data_path": "test_cases.json",  # 测试数据集路径
    "report_path": "evaluation_report.json"  # 评估报告路径
}


class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self):
        self.rag_app = RAGApplication()
        # 确保索引已加载
        if not self.rag_app.ingestion_pipeline.index:
            self.rag_app.ingestion_pipeline.get_documents()
    
    def load_test_cases(self, path: str) -> List[Dict]:
        """加载测试数据集"""
        if not Path(path).exists():
            logger.error(f"测试数据集不存在: {path}")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_recall(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        计算召回率(Recall)
        Recall = |检索到的相关文档| / |所有相关文档|
        """
        if not relevant_docs:
            return 1.0  # 没有相关文档时召回率为1
        
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        # 计算检索到的相关文档数量
        retrieved_relevant = len(retrieved_set & relevant_set)
        
        return retrieved_relevant / len(relevant_set)
    
    def calculate_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        计算精确率(Precision)
        Precision = |检索到的相关文档| / |检索到的所有文档|
        """
        if not retrieved_docs:
            return 0.0
        
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        retrieved_relevant = len(retrieved_set & relevant_set)
        
        return retrieved_relevant / len(retrieved_docs)
    
    def calculate_f1(self, recall: float, precision: float) -> float:
        """计算F1分数"""
        if recall + precision == 0:
            return 0.0
        return 2 * (recall * precision) / (recall + precision)
    
    async def evaluate_retrieval(self, query: str, relevant_docs: List[str]) -> Dict:
        """评估单次检索的召回率和精确率"""
        # 使用工作流进行检索
        if not self.rag_app.workflow:
            self.rag_app._ensure_workflow(streaming=False)
        
        try:
            # 执行工作流获取检索结果
            result = await self.rag_app.workflow.run(query=query, timeout=60.0)
            
            # 提取检索到的文档名
            retrieved_docs = []
            for source in result.get("sources", []):
                metadata = source.get("metadata", {})
                file_name = metadata.get("file_name", "unknown")
                retrieved_docs.append(file_name)
            
            # 计算指标
            recall = self.calculate_recall(retrieved_docs, relevant_docs)
            precision = self.calculate_precision(retrieved_docs, relevant_docs)
            f1 = self.calculate_f1(recall, precision)
            
            return {
                "query": query,
                "retrieved_docs": retrieved_docs,
                "relevant_docs": relevant_docs,
                "recall": recall,
                "precision": precision,
                "f1": f1
            }
        except Exception as e:
            logger.error(f"检索评估失败: {e}")
            return {
                "query": query,
                "error": str(e),
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0
            }
    
    async def evaluate_answer(self, query: str, expected_answer: str, knowledge_bool: bool = True) -> Dict:
        """评估单次回答的质量"""
        try:
            # 获取回答
            answer, sources = await self.rag_app.query_documents(
                session_id="eval_session",
                query=query,
                knowledge_bool=knowledge_bool
            )
            
            # 使用LLM评估回答质量（简化版：对比关键词匹配）
            # 实际项目中可以用更复杂的LLM评估
            accuracy = self.simple_accuracy_check(answer, expected_answer)
            
            # 检查忠实度：回答是否基于检索到的文档
            faithfulness = self.check_faithfulness(answer, sources)
            
            return {
                "query": query,
                "answer": answer,
                "expected_answer": expected_answer,
                "accuracy": accuracy,
                "faithfulness": faithfulness,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"回答评估失败: {e}")
            return {
                "query": query,
                "error": str(e),
                "accuracy": 0.0,
                "faithfulness": 0.0
            }
    
    def simple_accuracy_check(self, answer: str, expected_answer: str) -> float:
        """
        简单的准确率检查：计算关键词匹配率
        实际项目中可以用LLM做更精确的语义匹配
        """
        answer_tokens = set(answer.lower().split())
        expected_tokens = set(expected_answer.lower().split())
        
        if not expected_tokens:
            return 0.0
        
        # 计算匹配的关键词比例
        matched = len(answer_tokens & expected_tokens)
        return matched / len(expected_tokens)
    
    def check_faithfulness(self, answer: str, sources: List[str]) -> float:
        """
        检查回答忠实度：回答内容是否在来源文档中
        """
        if not sources:
            return 0.0
        
        # 提取来源文档的关键词
        source_content = " ".join([str(s.get("content", "")) for s in sources]).lower()
        source_tokens = set(source_content.split())
        
        # 提取回答的关键词
        answer_tokens = set(answer.lower().split())
        
        # 计算回答中有多少内容来自来源
        if not answer_tokens:
            return 0.0
        
        matched = len(answer_tokens & source_tokens)
        return matched / len(answer_tokens)
    
    async def run_evaluation(self, test_cases: List[Dict]) -> Dict:
        """运行完整评估"""
        logger.info(f"开始评估，共 {len(test_cases)} 个测试用例")
        
        retrieval_results = []
        answer_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"处理测试用例 {i}/{len(test_cases)}: {test_case['query'][:30]}...")
            
            # 评估检索
            retrieval_result = await self.evaluate_retrieval(
                test_case["query"],
                test_case.get("relevant_docs", [])
            )
            retrieval_results.append(retrieval_result)
            
            # 评估回答
            answer_result = await self.evaluate_answer(
                test_case["query"],
                test_case.get("expected_answer", ""),
                test_case.get("knowledge_bool", True)
            )
            answer_results.append(answer_result)
        
        # 计算汇总指标
        summary = self.calculate_summary(retrieval_results, answer_results)
        
        # 生成报告
        report = {
            "config": EVAL_CONFIG,
            "summary": summary,
            "retrieval_results": retrieval_results,
            "answer_results": answer_results,
            "test_cases": test_cases
        }
        
        return report
    
    def calculate_summary(self, retrieval_results: List[Dict], answer_results: List[Dict]) -> Dict:
        """计算汇总指标"""
        # 检索指标汇总
        valid_retrieval = [r for r in retrieval_results if "error" not in r]
        avg_recall = sum(r["recall"] for r in valid_retrieval) / len(valid_retrieval) if valid_retrieval else 0.0
        avg_precision = sum(r["precision"] for r in valid_retrieval) / len(valid_retrieval) if valid_retrieval else 0.0
        avg_f1 = sum(r["f1"] for r in valid_retrieval) / len(valid_retrieval) if valid_retrieval else 0.0
        
        # 回答指标汇总
        valid_answer = [a for a in answer_results if "error" not in a]
        avg_accuracy = sum(a["accuracy"] for a in valid_answer) / len(valid_answer) if valid_answer else 0.0
        avg_faithfulness = sum(a["faithfulness"] for a in valid_answer) / len(valid_answer) if valid_answer else 0.0
        
        return {
            "retrieval_metrics": {
                "test_count": len(retrieval_results),
                "valid_count": len(valid_retrieval),
                "avg_recall": avg_recall,
                "avg_precision": avg_precision,
                "avg_f1": avg_f1
            },
            "answer_metrics": {
                "test_count": len(answer_results),
                "valid_count": len(valid_answer),
                "avg_accuracy": avg_accuracy,
                "avg_faithfulness": avg_faithfulness
            }
        }
    
    def print_report(self, report: Dict):
        """打印评估报告"""
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("RAG系统评估报告")
        print("="*60)
        
        print("\n【检索指标】")
        print(f"测试用例数: {summary['retrieval_metrics']['test_count']}")
        print(f"有效用例数: {summary['retrieval_metrics']['valid_count']}")
        print(f"平均召回率(Recall): {summary['retrieval_metrics']['avg_recall']:.4f}")
        print(f"平均精确率(Precision): {summary['retrieval_metrics']['avg_precision']:.4f}")
        print(f"平均F1分数: {summary['retrieval_metrics']['avg_f1']:.4f}")
        
        print("\n【回答指标】")
        print(f"测试用例数: {summary['answer_metrics']['test_count']}")
        print(f"有效用例数: {summary['answer_metrics']['valid_count']}")
        print(f"平均准确率(Accuracy): {summary['answer_metrics']['avg_accuracy']:.4f}")
        print(f"平均忠实度(Faithfulness): {summary['answer_metrics']['avg_faithfulness']:.4f}")
        
        # 打印详细结果
        print("\n【详细结果】")
        for i, (retrieval, answer) in enumerate(
            zip(report["retrieval_results"], report["answer_results"]), 1
        ):
            print(f"\n测试用例 {i}: {retrieval['query']}")
            print(f"  检索召回率: {retrieval['recall']:.4f}")
            print(f"  检索精确率: {retrieval['precision']:.4f}")
            print(f"  回答准确率: {answer['accuracy']:.4f}")
            print(f"  回答忠实度: {answer['faithfulness']:.4f}")
            if "error" in retrieval:
                print(f"  检索错误: {retrieval['error']}")
            if "error" in answer:
                print(f"  回答错误: {answer['error']}")
        
        print("\n" + "="*60)
    
    def save_report(self, report: Dict, path: str):
        """保存评估报告到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"评估报告已保存: {path}")


def generate_test_cases_template(path: str):
    """生成测试数据集模板"""
    template = [
        {
            "query": "无油空气压缩机的功率是多少？",
            "expected_answer": "无油空气压缩机的功率为5.5千瓦",
            "relevant_docs": ["无油空气压缩机.pdf"],
            "knowledge_bool": True
        },
        {
            "query": "产品的保修期限是多久？",
            "expected_answer": "产品保修期为一年",
            "relevant_docs": ["产品手册.pdf"],
            "knowledge_bool": True
        },
        {
            "query": "如何安装设备？",
            "expected_answer": "安装步骤：1. 开箱检查...",
            "relevant_docs": ["安装指南.pdf"],
            "knowledge_bool": True
        },
        {
            "query": "介绍一下公司的发展历程",
            "expected_answer": "公司成立于2000年...",
            "relevant_docs": ["公司介绍.pdf"],
            "knowledge_bool": True
        },
        {
            "query": "什么是人工智能？",
            "expected_answer": "人工智能是...",
            "relevant_docs": [],  # 知识库外问题
            "knowledge_bool": False
        }
    ]
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    logger.info(f"测试数据集模板已生成: {path}")


async def main():
    """主函数"""
    # 检查测试数据集是否存在
    if not Path(EVAL_CONFIG["test_data_path"]).exists():
        logger.warning("测试数据集不存在，生成模板")
        generate_test_cases_template(EVAL_CONFIG["test_data_path"])
        print(f"请先填充 {EVAL_CONFIG['test_data_path']} 中的测试用例，然后重新运行")
        return
    
    # 加载测试数据集
    evaluator = RAGEvaluator()
    test_cases = evaluator.load_test_cases(EVAL_CONFIG["test_data_path"])
    
    if not test_cases:
        logger.error("没有可用的测试用例")
        return
    
    # 运行评估
    report = await evaluator.run_evaluation(test_cases)
    
    # 打印报告
    evaluator.print_report(report)
    
    # 保存报告
    evaluator.save_report(report, EVAL_CONFIG["report_path"])


if __name__ == "__main__":
    # 设置LLM（需要配置API密钥）
    from config.settings import Settings as AppSettings
    Settings.llm = DashScope(
        api_key=AppSettings.API_KEY,
        api_base=AppSettings.API_BASE_URL,
        model_name=AppSettings.MODEL,
        temperature=EVAL_CONFIG["temperature"]
    )
    
    # 运行评估
    asyncio.run(main())
