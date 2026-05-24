"""RAG系统评估模块主入口 - 基于RAGAS框架的Chunk级别评估"""

import argparse
import sys
from data_loader import DataLoader, create_sample_dataset
from config import settings



def run_ragas_evaluation(args):
    """执行RAGAS自动化评估流程"""
    from ragas_evaluator import RagasEvaluator
    
    evaluator = RagasEvaluator()
    
    if args.action == "generate":
        # 生成测试集
        print("=== RAGAS测试集生成 ===")
        
        # 选择文档来源
        if args.source == "oss":
            print("从OSS加载文档...")
            documents = evaluator.load_documents_from_oss()
        else:
            print("从本地加载文档...")
            documents = evaluator.load_documents_from_local()
        
        if not documents:
            print("错误：未加载到任何文档")
            sys.exit(1)
        
        # 生成测试集
        test_size = args.test_size if args.test_size else 10
        testset = evaluator.generate_testset(documents, test_size=test_size)
        
        if testset:
            evaluator.save_testset(testset, args.output or "ragas_generated_testset.json")
            print("\n测试集生成完成！")
    
    elif args.action == "evaluate":
        # 执行评估
        print("=== RAGAS评估 ===")
        
        if not args.dataset:
            print("错误：请指定测试集文件（--dataset 参数）")
            sys.exit(1)
        
        testset = evaluator.load_testset(args.dataset)
        if not testset:
            print(f"错误：无法加载测试集 '{args.dataset}'")
            sys.exit(1)
        
        result = evaluator.evaluate(testset)
        evaluator.print_evaluation_report(result)
    
    elif args.action == "full":
        # 完整流程：生成测试集 + 评估
        print("=== RAGAS完整评估流程 ===")
        
        # 加载文档
        if args.source == "oss":
            print("从OSS加载文档...")
            documents = evaluator.load_documents_from_oss()
        else:
            print("从本地加载文档...")
            documents = evaluator.load_documents_from_local()
        
        if not documents:
            print("错误：未加载到任何文档")
            sys.exit(1)
        
        # 生成测试集
        test_size = args.test_size if args.test_size else 10
        print(f"\n生成 {test_size} 个测试样本...")
        testset = evaluator.generate_testset(documents, test_size=test_size)
        
        if not testset:
            print("错误：测试集生成失败")
            sys.exit(1)
        
        output_filename = args.output or "ragas_generated_testset.json"
        evaluator.save_testset(testset, output_filename)
        
        # 执行评估
        print("\n执行评估...")
        result = evaluator.evaluate(testset)
        evaluator.print_evaluation_report(result)
        
        print("\n=== 完整流程完成 ===")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAG系统性能评估工具 - 基于RAGAS框架的Chunk级别评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

【RAGAS自动化评估模式】
  # 从本地文档生成测试集（10个样本）
  python main.py --action generate --source local --test-size 10
  
  # 从OSS文档生成测试集
  python main.py --action generate --source oss --output my_testset.json
  
  # 使用已有测试集执行RAGAS评估
  python main.py --action evaluate --dataset ragas_generated_testset.json
  
  # 完整流程：生成测试集 + 执行评估
  python main.py --action full --source local --test-size 10

指标说明:
  - Context Recall: 检索器是否能召回完整的相关信息（Chunk级别）
  - Context Precision: 检索到的上下文是否都与问题相关（Chunk级别）
  - Faithfulness: 回答是否基于检索内容，有无幻觉
  - Answer Relevancy: 回答是否直接回答了问题
  - Answer Correctness: 回答的事实正确性
        """
    )
    
    # 通用参数
    parser.add_argument(
        "--dataset",
        type=str,
        help="测试数据集文件名（JSON格式）"
    )
    
    parser.add_argument(
        "--source",
        choices=["local", "oss"],
        default="local",
        help="数据源：local(本地) 或 oss(阿里云OSS)"
    )
    
    parser.add_argument(
        "--action",
        choices=["generate", "evaluate", "full"],
        required=True,
        help="RAGAS操作类型: generate(生成测试集), evaluate(执行评估), full(完整流程)"
    )
    
    parser.add_argument(
        "--test-size",
        type=int,
        default=10,
        help="生成测试集的样本数量（默认10）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="生成的测试集保存文件名"
    )
    
    args = parser.parse_args()
    
    # 执行RAGAS评估
    run_ragas_evaluation(args)


if __name__ == "__main__":
    main()