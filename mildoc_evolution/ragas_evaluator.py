"""RAGAS评估器模块 - 基于RAGAS框架实现Chunk级别自动化RAG系统评估"""

import os
import json
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

# ✅ 修复：延迟导入 RAGAS，避免模块加载时阻塞
RAGAS_AVAILABLE = False
Dataset = None  # type: ignore

def _check_ragas_available():
    """检查 RAGAS 是否可用（延迟导入）"""
    global RAGAS_AVAILABLE, Dataset
    if RAGAS_AVAILABLE:
        return True
    
    try:
        from ragas import evaluate
        from ragas.metrics import (
            context_recall,
            context_precision,
            faithfulness,
            answer_relevancy,
            answer_correctness
        )
        from ragas.testset import TestsetGenerator  # RAGAS 0.4.x 的新路径
        from datasets import Dataset as RagasDataset
        
        # 将导入的模块保存到全局变量
        globals()['evaluate'] = evaluate
        globals()['context_recall'] = context_recall
        globals()['context_precision'] = context_precision
        globals()['faithfulness'] = faithfulness
        globals()['answer_relevancy'] = answer_relevancy
        globals()['answer_correctness'] = answer_correctness
        globals()['TestsetGenerator'] = TestsetGenerator
        Dataset = RagasDataset
        RAGAS_AVAILABLE = True
        return True
        
    except ImportError as e:
        print(f"警告：未安装ragas库或导入失败: {e}")
        print("请运行 `uv pip install ragas` 安装")
        return False

try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("警告：未安装langchain相关库")

from config import settings
from data_loader import ChunkInfo
try:
    from document_converter import DocumentConverter
    CONVERTER_AVAILABLE = True
except ImportError:
    CONVERTER_AVAILABLE = False


class RagasEvaluator:
    """RAGAS评估器 - 支持测试集自动生成和多维度指标评估"""

    def __init__(self):
        self.generator = None
        self.llm = None
        self.embeddings = None
        self._init_ragas()

    def _init_ragas(self):
        """初始化RAGAS所需的LLM和Embedding"""
        # ✅ 修复：延迟导入 RAGAS
        if not _check_ragas_available():
            print("RAGAS不可用，请安装依赖")
            return
        
        if not settings.validate_ragas_config():
            print("RAGAS配置不完整，跳过初始化")
            return

        try:
            # 设置环境变量（LangChain需要）
            import os
            os.environ['OPENAI_API_KEY'] = settings.LLM_API_KEY
            os.environ['OPENAI_BASE_URL'] = settings.LLM_BASE_URL
            
            # 配置LLM
            self.llm = ChatOpenAI(
                model_name=settings.LLM_MODEL_NAME,
                openai_api_key=settings.LLM_API_KEY,
                openai_api_base=settings.LLM_BASE_URL,
                temperature=0.0
            )

            # 配置Embedding
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                openai_api_key=settings.EMBEDDING_API_KEY,
                openai_api_base=settings.EMBEDDING_BASE_URL
            )

            # 初始化测试集生成器（RAGAS 0.4.x API）
            self.generator = TestsetGenerator.from_langchain(
                llm=self.llm,
                embedding_model=self.embeddings
            )

            print("✅ RAGAS评估器初始化成功")

        except Exception as e:
            print(f"RAGAS初始化失败: {e}")
            import traceback
            traceback.print_exc()

    def load_documents_from_local(self, docs_dir: Optional[str] = None) -> List:
        """
        从本地目录加载文档
        
        Args:
            docs_dir: 文档目录路径，默认为配置中的LOCAL_DOCUMENTS_DIR
        
        Returns:
            文档列表
        """
        if not LANGCHAIN_AVAILABLE:
            print("Langchain不可用，无法加载文档")
            return []

        target_dir = Path(docs_dir) if docs_dir else settings.LOCAL_DOCUMENTS_DIR
        
        if not target_dir.exists():
            print(f"文档目录不存在: {target_dir}")
            return []

        try:
            loader = DirectoryLoader(
                str(target_dir),
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            print(f"已加载 {len(documents)} 个文档")
            return documents
        except Exception as e:
            print(f"加载文档失败: {e}")
            return []

    def load_documents_from_oss(self) -> List:
        """
        从OSS加载文档到本地，自动转换PDF/DOCX为txt格式
        
        Returns:
            文档列表
        """
        if not settings.validate_oss_config():
            print("OSS配置不完整")
            return []

        try:
            import oss2
            auth = oss2.Auth(settings.OSS_ACCESS_KEY_ID, settings.OSS_ACCESS_KEY_SECRET)
            bucket = oss2.Bucket(auth, settings.OSS_ENDPOINT, settings.OSS_BUCKET_NAME)

            # 初始化文档转换器（如果需要）
            converter = None
            if CONVERTER_AVAILABLE:
                converter = DocumentConverter()
            
            # 下载所有文档到本地（支持多种格式）
            docs_downloaded = []
            text_extensions = ('.txt', '.md', '.markdown')  # 文本格式
            convertible_extensions = ('.pdf', '.docx', '.doc', '.xlsx', '.pptx')  # 可转换格式
            all_supported = text_extensions + convertible_extensions
            
            print(f"开始从 OSS 扫描文档...")
            print(f"Bucket: {settings.OSS_BUCKET_NAME}")
            print(f"支持格式: {', '.join(all_supported)}")
            if converter:
                print(f"🔄 文档转换器已就绪，将自动转换PDF/DOCX等格式")
            else:
                print(f"⚠️  文档转换器不可用，仅处理文本格式")
            
            # 先尝试从 documents/ 目录查找，再查根目录
            doc_prefixes = [
                settings.OSS_DOCUMENTS_PREFIX,  # documents/
                ''  # 根目录
            ]
            
            for prefix in doc_prefixes:
                print(f"\n扫描前缀: '{prefix}'")
                file_count = 0
                convert_count = 0
                
                for obj in oss2.ObjectIterator(bucket, prefix=prefix):
                    # 检查是否为支持的格式
                    if not any(obj.key.lower().endswith(ext) for ext in all_supported):
                        continue
                    
                    # 构造本地路径
                    relative_path = obj.key.replace(prefix, '') if prefix else obj.key
                    local_path = settings.LOCAL_DOCUMENTS_DIR / relative_path
                    
                    # 如果是文本格式，直接下载
                    if any(obj.key.lower().endswith(ext) for ext in text_extensions):
                        # 如果文件已存在且大小相同，跳过
                        if local_path.exists() and local_path.stat().st_size == obj.size:
                            print(f"  ⏭️  跳过(已存在): {obj.key}")
                            docs_downloaded.append(str(local_path))
                            continue
                        
                        # 下载文件
                        try:
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            bucket.get_object_to_file(obj.key, str(local_path))
                            docs_downloaded.append(str(local_path))
                            file_count += 1
                            print(f"  ✅ 下载: {obj.key} ({obj.size} bytes)")
                        except Exception as e:
                            print(f"  ❌ 下载失败 {obj.key}: {e}")
                    
                    # 如果是可转换格式，下载后转换
                    elif converter and any(obj.key.lower().endswith(ext) for ext in convertible_extensions):
                        # 先下载到临时位置
                        temp_path = local_path
                        try:
                            temp_path.parent.mkdir(parents=True, exist_ok=True)
                            bucket.get_object_to_file(obj.key, str(temp_path))
                            
                            # 转换为txt
                            txt_path = converter.convert_to_txt(
                                str(temp_path), 
                                str(settings.LOCAL_DOCUMENTS_DIR)
                            )
                            
                            if txt_path:
                                docs_downloaded.append(txt_path)
                                file_count += 1
                                convert_count += 1
                                print(f"  ✅ 转换: {obj.key} -> {Path(txt_path).name}")
                            else:
                                print(f"  ⚠️  转换失败: {obj.key}")
                        except Exception as e:
                            print(f"  ❌ 处理失败 {obj.key}: {e}")
                    else:
                        print(f"  ⚠️  跳过(不支持或无转换器): {obj.key}")
                
                if file_count > 0:
                    print(f"从前缀 '{prefix}' 处理了 {file_count} 个文件（转换 {convert_count} 个）")
                    break  # 找到文件后停止
            
            if not docs_downloaded:
                print("⚠️  OSS上未找到支持的文档格式")
                return []
            
            print(f"\n✅ 总共处理了 {len(docs_downloaded)} 个文档")

            # 加载到Langchain
            return self.load_documents_from_local(str(settings.LOCAL_DOCUMENTS_DIR))

        except Exception as e:
            print(f"从OSS加载文档失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def generate_testset(self, documents: List, test_size: int = 10) -> Optional["Dataset"]:
        """
        使用RAGAS自动生成测试集
        
        Args:
            documents: Langchain文档列表
            test_size: 生成的测试样本数量
        
        Returns:
            RAGAS测试集（Dataset格式）
        """
        if not self.generator or not documents:
            print("无法生成测试集：生成器未初始化或文档为空")
            return None

        try:
            print(f"开始生成 {test_size} 个测试样本...")
            start_time = time.time()

            # RAGAS 0.4.x API: 使用 testset_size 和 query_distribution
            testset = self.generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=test_size,  # 注意：不是 test_size
                query_distribution={      # 注意：不是 distributions
                    "simple": 0.4,       # 简单问题
                    "reasoning": 0.3,    # 推理问题
                    "multi_context": 0.3 # 多上下文问题
                }
            )

            elapsed = time.time() - start_time
            print(f"测试集生成完成，耗时 {elapsed:.2f} 秒")
            return testset

        except Exception as e:
            print(f"生成测试集失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_testset(self, testset, filename: str):
        """
        保存测试集到本地
        
        Args:
            testset: RAGAS测试集
            filename: 保存文件名
        """
        if not testset:
            print("测试集为空")
            return

        try:
            # 转换为DataFrame并保存
            df = testset.to_pandas()
            output_path = settings.LOCAL_TEST_DATA_DIR / filename
            df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            print(f"测试集已保存到: {output_path}")
        except Exception as e:
            print(f"保存测试集失败: {e}")

    def load_testset(self, filename: str) -> Optional["Dataset"]:
        """
        从本地加载测试集（支持两种格式）
        
        Args:
            filename: 测试集文件名
        
        Returns:
            RAGAS测试集（Dataset格式）
        """
        try:
            file_path = settings.LOCAL_TEST_DATA_DIR / filename
            if not file_path.exists():
                print(f"测试集文件不存在: {file_path}")
                return None

            import pandas as pd
            import json
            
            # 先读取JSON判断格式
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # 格式1：自定义格式（有 samples 字段）
            if 'samples' in raw_data:
                print(f"检测到自定义测试集格式，转换中...")
                samples = raw_data['samples']
                
                # 提取需要的字段
                questions = [s.get('question', '') for s in samples]
                ground_truths = [s.get('ground_truth', '') for s in samples]
                
                # 构建 contexts（从 expected_chunks 中提取）
                contexts_list = []
                expected_chunks_list = []
                
                for sample in samples:
                    # 从 expected_chunks 提取 contexts
                    if 'expected_chunks' in sample and sample['expected_chunks']:
                        chunks = sample['expected_chunks']
                        ctx = [chunk.get('content', '') for chunk in chunks if chunk.get('content')]
                        contexts_list.append(ctx)
                        expected_chunks_list.append(chunks)
                    else:
                        contexts_list.append([])
                        expected_chunks_list.append([])
                
                # 创建 DataFrame
                df = pd.DataFrame({
                    'question': questions,
                    'ground_truth': ground_truths,
                    'contexts': contexts_list,
                    'expected_chunks': expected_chunks_list
                })
                
                dataset = Dataset.from_pandas(df)
                print(f"✅ 已加载测试集: {filename}，共 {len(dataset)} 个样本")
                return dataset
            
            # 格式2：RAGAS Dataset 格式（直接是 records）
            else:
                df = pd.read_json(file_path, orient='records')
                dataset = Dataset.from_pandas(df)
                print(f"✅ 已加载测试集: {filename}，共 {len(dataset)} 个样本")
                return dataset
                
        except Exception as e:
            print(f"加载测试集失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_rag_query(self, question: str) -> Dict[str, Any]:
        """
        调用RAG系统获取回答和上下文（支持Chunk级别信息）
        
        Args:
            question: 用户问题
        
        Returns:
            包含answer、contexts和retrieved_chunks的字典
        """
        try:
            # ✅ 修复：使用 importlib 动态加载，避免 sys.path 冲突
            import sys
            import importlib.util
            from pathlib import Path
            
            # 动态加载 rag_service 模块（避免与 mildoc_evolution.config 冲突）
            wxkf_dir = Path(__file__).parent.parent / "mildoc_wxkf"
            rag_service_path = wxkf_dir / "rag_service.py"
            
            if not rag_service_path.exists():
                print(f"❌ rag_service.py 不存在: {rag_service_path}")
                return {"answer": "", "contexts": [], "retrieved_chunks": []}
            
            # ⚠️ 关键修复：临时添加 mildoc_wxkf 到 sys.path，并保存原始状态
            original_path = sys.path.copy()
            original_modules = {k: v for k, v in sys.modules.items() if k in ['config', 'rerank_service', 'rag_service']}
            
            try:
                print("🔄 正在加载 RAG 服务模块...")
                
                # 临时移除 mildoc_evolution 目录，防止 config 冲突
                evolution_dir = str(Path(__file__).parent)
                if evolution_dir in sys.path:
                    sys.path.remove(evolution_dir)
                
                # 将 mildoc_wxkf 添加到最前面
                if str(wxkf_dir) not in sys.path:
                    sys.path.insert(0, str(wxkf_dir))
                
                # 清除可能已缓存的错误模块
                for mod_name in ['config', 'rerank_service', 'rag_service']:
                    if mod_name in sys.modules:
                        del sys.modules[mod_name]
                
                # 现在可以安全导入
                from rag_service import get_rag_service
                
                print("🔄 正在初始化 RAG 服务（可能需要连接 Milvus）...")
                rag_service = get_rag_service()
                
            except Exception as e:
                print(f"❌ RAG 服务加载或初始化失败: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return {"answer": "", "contexts": [], "retrieved_chunks": []}
            finally:
                # 恢复原始 sys.path
                sys.path[:] = original_path
                
                # 恢复原始模块缓存（可选，避免污染全局命名空间）
                for mod_name in ['config', 'rerank_service', 'rag_service']:
                    if mod_name in sys.modules:
                        del sys.modules[mod_name]
                    if mod_name in original_modules:
                        sys.modules[mod_name] = original_modules[mod_name]
            
            if not rag_service:
                print("❌ RAG服务初始化失败")
                return {"answer": "", "contexts": [], "retrieved_chunks": []}
            
            # 调用RAG服务
            response = rag_service.query_service(question)
            
            if not response.success:
                print(f"⚠️  RAG查询失败: {response.error_message}")
                return {"answer": "", "contexts": [], "retrieved_chunks": []}
            
            # 提取上下文（检索到的文档内容）
            contexts = []
            retrieved_chunks = []
            
            if response.source_documents:
                for idx, doc in enumerate(response.source_documents):
                    content = doc.content_preview if hasattr(doc, 'content_preview') else str(doc)
                    doc_name = doc.doc_name if hasattr(doc, 'doc_name') else f"doc_{idx}"
                    score = doc.similarity_score if hasattr(doc, 'similarity_score') else 0.0
                    
                    if content:
                        contexts.append(content)
                        
                        # 构建Chunk信息（如果RAG系统返回了chunk_id）
                        chunk_info = {
                            'chunk_id': getattr(doc, 'chunk_id', f"{doc_name}_chunk_{idx}"),
                            'content': content,
                            'doc_name': doc_name,
                            'score': score,
                            'metadata': {
                                'position': idx,
                                'relevance_score': score
                            }
                        }
                        retrieved_chunks.append(chunk_info)
            
            return {
                "answer": response.content,
                "contexts": contexts,
                "retrieved_chunks": retrieved_chunks
            }

        except ImportError as e:
            print(f"❌ 导入RAG服务模块失败: {e}")
            print(f"   请确保 mildoc_wxkf 目录存在且包含 rag_service.py")
            return {"answer": "", "contexts": [], "retrieved_chunks": []}
        except Exception as e:
            print(f"❌ 调用RAG系统失败: {e}")
            import traceback
            traceback.print_exc()
            return {"answer": "", "contexts": [], "retrieved_chunks": []}

    def evaluate(self, testset: "Dataset") -> Dict[str, Any]:
        """
        使用RAGAS评估RAG系统性能（支持Chunk级别分析）
        
        Args:
            testset: RAGAS测试集
        
        Returns:
            评估结果字典
        """
        if not _check_ragas_available() or not testset:
            print("无法执行评估：RAGAS不可用或测试集为空")
            return {}

        print("\n=== 开始RAGAS评估 ===")
        print(f"测试样本数: {len(testset)}")

        # 运行RAG系统获取所有回答
        questions = testset["question"]
        answers = []
        contexts = []  # ⚠️ 关键修复：从RAG检索结果中提取真实contexts
        all_retrieved_chunks = []  # 保存每个问题的检索结果
        failed_samples = []  # 记录失败的样本

        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] 处理问题: {question[:50]}...")
            try:
                result = self.run_rag_query(question)
                
                # 检查是否成功获取回答
                if not result.get("answer") or not result.get("contexts"):
                    print(f"  ⚠️  警告：样本 {i} 未获取到有效回答或上下文，跳过")
                    failed_samples.append(i)
                    continue
                
                answers.append(result["answer"])
                contexts.append(result["contexts"])  # ✅ 使用真实的检索上下文
                all_retrieved_chunks.append(result["retrieved_chunks"])
            except Exception as e:
                print(f"  ❌ 错误：样本 {i} 处理失败: {e}")
                failed_samples.append(i)
                continue

        # 如果有失败样本，发出警告
        if failed_samples:
            print(f"\n⚠️  警告：{len(failed_samples)} 个样本处理失败，将从评估中排除")
            print(f"   失败样本索引: {failed_samples}")
        
        # 如果没有有效样本，直接返回
        if not answers:
            print("❌ 错误：没有有效的评估样本")
            return {}

        # 构建评估数据集（只包含成功的样本）
        eval_data = {
            "question": [questions[i] for i in range(len(questions)) if i not in failed_samples],
            "answer": answers,
            "contexts": contexts,
            "ground_truth": [testset["ground_truth"][i] for i in range(len(testset["ground_truth"])) if i not in failed_samples]
        }

        eval_dataset = Dataset.from_dict(eval_data)

        # 执行RAGAS评估（传入自定义LLM配置）
        start_time = time.time()
        try:
            scores = evaluate(
                dataset=eval_dataset,
                metrics=[
                    context_recall,      # 上下文召回率
                    context_precision,   # 上下文精确率
                    faithfulness,        # 忠实度（无幻觉）
                    answer_relevancy,    # 答案相关性
                    answer_correctness   # 答案正确性
                ],
                llm=self.llm,  # 使用我们配置的LLM（qwen-plus）
                embeddings=self.embeddings  # 使用我们配置的Embedding
            )
        except Exception as e:
            print(f"评估失败: {e}")
            return {}

        elapsed = time.time() - start_time
        print(f"RAGAS评估完成，耗时 {elapsed:.2f} 秒")

        # 转换结果为字典（RAGAS 0.4.x API）
        try:
            # 尝试使用 to_dict() 方法
            result_dict = scores.to_dict()
            if not isinstance(result_dict, dict):
                raise ValueError("to_dict() 返回的不是字典类型")
        except (AttributeError, ValueError):
            # RAGAS 0.4.x 可能使用不同的方法
            try:
                # 尝试转换为 pandas 然后转字典
                import pandas as pd
                df = scores.to_pandas()
                # 计算每个指标的平均值
                result_dict = {}
                for col in df.columns:
                    if col != 'question':  # 跳过问题列
                        try:
                            result_dict[col] = float(df[col].mean())
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                print(f"警告：无法转换评估结果: {e}")
                # 最后的备选方案：手动构建字典
                result_dict = {}
                try:
                    for metric_name in scores.scores.keys():
                        result_dict[metric_name] = float(scores.scores[metric_name].mean())
                except Exception:
                    result_dict = {"error": "无法提取评估结果"}
        
        # 添加Chunk级别指标（如果有expected_chunks标注）
        chunk_metrics_summary = self._calculate_chunk_metrics_if_available(testset, all_retrieved_chunks)
        if chunk_metrics_summary:
            result_dict['chunk_level_metrics'] = chunk_metrics_summary

        # ✅ 修复：处理NaN值，转换为None以便JSON序列化
        def sanitize_for_json(obj):
            """递归清理NaN和Inf值"""
            import math
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None  # JSON不支持NaN/Inf，转为null
                return obj
            elif isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            return obj
        
        result_dict = sanitize_for_json(result_dict)

        # 保存评估结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = settings.RAGAS_OUTPUT_DIR / f"ragas_evaluation_result_{timestamp}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2, default=str)
            print(f"✅ 评估结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存评估结果失败: {e}")
            import traceback
            traceback.print_exc()

        return result_dict

    def _calculate_chunk_metrics_if_available(self, testset, all_retrieved_chunks: List[List[Dict]]) -> Optional[Dict]:
        """
        如果测试集包含expected_chunks标注，计算Chunk级别指标
        
        Args:
            testset: 测试集
            all_retrieved_chunks: 所有问题的检索结果
            
        Returns:
            Chunk级别指标汇总，如果没有expected_chunks则返回None
        """
        try:
            from chunk_metrics import evaluate_chunk_level_metrics, aggregate_metrics
            
            # 检查是否有expected_chunks字段
            if 'expected_chunks' not in testset.column_names:
                print("ℹ️  测试集不包含expected_chunks字段，跳过Chunk级别指标计算")
                return None
            
            expected_chunks_list = testset['expected_chunks']
            
            # 过滤出有expected_chunks标注的样本
            valid_samples = []
            skipped_count = 0
            
            for i, (expected_chunks, retrieved_chunks) in enumerate(zip(expected_chunks_list, all_retrieved_chunks)):
                if expected_chunks and len(expected_chunks) > 0:
                    # 将ChunkInfo对象转换为字典（兼容Pydantic V2）
                    if hasattr(expected_chunks[0], 'model_dump'):
                        expected_chunks_dict = [chunk.model_dump() for chunk in expected_chunks]
                    elif hasattr(expected_chunks[0], 'dict'):
                        expected_chunks_dict = [chunk.dict() for chunk in expected_chunks]
                    else:
                        expected_chunks_dict = expected_chunks
                    
                    try:
                        metrics = evaluate_chunk_level_metrics(expected_chunks_dict, retrieved_chunks)
                        valid_samples.append(metrics)
                    except Exception as e:
                        print(f"  ⚠️  样本 {i} 的Chunk指标计算失败: {e}")
                        skipped_count += 1
                else:
                    skipped_count += 1
            
            if not valid_samples:
                print(f"⚠️  没有有效样本包含expected_chunks标注（跳过{skipped_count}个样本），跳过Chunk级别指标计算")
                return None
            
            if skipped_count > 0:
                print(f"ℹ️  共跳过 {skipped_count} 个无标注或计算失败的样本")
            
            # 聚合指标
            aggregated = aggregate_metrics(valid_samples)
            print(f"✅ Chunk级别指标计算完成（基于{len(valid_samples)}个有效样本）")
            
            return aggregated
            
        except ImportError as e:
            print(f"⚠️  无法导入chunk_metrics模块: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Chunk级别指标计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_evaluation_report(self, result: Dict[str, Any]):
        """
        打印评估报告
        
        Args:
            result: 评估结果字典
        """
        if not result:
            print("评估结果为空")
            return

        print("\n" + "="*60)
        print("         RAGAS 评估报告 (Chunk级别)")
        print("="*60)
        
        # 打印RAGAS指标得分
        print("\n【RAGAS 核心指标】")
        if 'context_recall' in result and result['context_recall'] is not None:
            print(f"  上下文召回率 (Context Recall):     {result['context_recall']:.4f}")
        else:
            print("  上下文召回率 (Context Recall):     N/A (评估失败)")
        
        if 'context_precision' in result and result['context_precision'] is not None:
            print(f"  上下文精确率 (Context Precision):  {result['context_precision']:.4f}")
        else:
            print("  上下文精确率 (Context Precision):  N/A (评估失败)")
        
        if 'faithfulness' in result and result['faithfulness'] is not None:
            print(f"  忠实度 (Faithfulness):             {result['faithfulness']:.4f}")
        else:
            print("  忠实度 (Faithfulness):             N/A (评估失败)")
        
        if 'answer_relevancy' in result and result['answer_relevancy'] is not None:
            print(f"  答案相关性 (Answer Relevancy):     {result['answer_relevancy']:.4f}")
        else:
            print("  答案相关性 (Answer Relevancy):     N/A (评估失败)")
        
        if 'answer_correctness' in result and result['answer_correctness'] is not None:
            print(f"  答案正确性 (Answer Correctness):   {result['answer_correctness']:.4f}")
        else:
            print("  答案正确性 (Answer Correctness):   N/A (评估失败)")
        
        # 打印Chunk级别指标（如果有）
        if 'chunk_level_metrics' in result:
            print("\n【Chunk 级别指标】")
            chunk_metrics = result['chunk_level_metrics']
            if 'avg_recall' in chunk_metrics and chunk_metrics['avg_recall'] is not None:
                print(f"  平均召回率 (Avg Recall):         {chunk_metrics['avg_recall']:.4f}")
            else:
                print("  平均召回率 (Avg Recall):         N/A")
            
            if 'avg_precision' in chunk_metrics and chunk_metrics['avg_precision'] is not None:
                print(f"  平均准确率 (Avg Precision):      {chunk_metrics['avg_precision']:.4f}")
            else:
                print("  平均准确率 (Avg Precision):      N/A")
            
            if 'avg_f1' in chunk_metrics and chunk_metrics['avg_f1'] is not None:
                print(f"  平均F1值 (Avg F1):               {chunk_metrics['avg_f1']:.4f}")
            else:
                print("  平均F1值 (Avg F1):               N/A")
            
            if 'avg_mrr' in chunk_metrics and chunk_metrics['avg_mrr'] is not None:
                print(f"  平均MRR (Avg MRR):               {chunk_metrics['avg_mrr']:.4f}")
            else:
                print("  平均MRR (Avg MRR):               N/A")
            
            if 'avg_recall_at_1' in chunk_metrics and chunk_metrics['avg_recall_at_1'] is not None:
                print(f"  Recall@1:                        {chunk_metrics['avg_recall_at_1']:.4f}")
            else:
                print("  Recall@1:                        N/A")
            
            if 'avg_recall_at_3' in chunk_metrics and chunk_metrics['avg_recall_at_3'] is not None:
                print(f"  Recall@3:                        {chunk_metrics['avg_recall_at_3']:.4f}")
            else:
                print("  Recall@3:                        N/A")
            
            if 'avg_recall_at_5' in chunk_metrics and chunk_metrics['avg_recall_at_5'] is not None:
                print(f"  Recall@5:                        {chunk_metrics['avg_recall_at_5']:.4f}")
            else:
                print("  Recall@5:                        N/A")
            
            if 'avg_ndcg_at_5' in chunk_metrics and chunk_metrics['avg_ndcg_at_5'] is not None:
                print(f"  NDCG@5:                          {chunk_metrics['avg_ndcg_at_5']:.4f}")
            else:
                print("  NDCG@5:                          N/A")
        
        print("\n" + "="*60)
        print("指标说明:")
        print("  【RAGAS指标】")
        print("    - Context Recall: 检索器是否能召回完整的相关信息")
        print("    - Context Precision: 检索到的上下文是否都与问题相关")
        print("    - Faithfulness: 回答是否基于检索内容，有无幻觉")
        print("    - Answer Relevancy: 回答是否直接回答了问题")
        print("    - Answer Correctness: 回答的事实正确性")
        print("\n  【Chunk级别指标】")
        print("    - Recall/Precision/F1: 真正的文本块级别检索质量")
        print("    - MRR: 第一个相关chunk的排名倒数")
        print("    - Recall@K: Top K个结果中的召回率")
        print("    - NDCG@5: 前5个结果的排序质量")
        print("="*60)


def main():
    """演示RAGAS评估流程"""
    evaluator = RagasEvaluator()

    # 示例1: 从本地加载文档并生成测试集
    print("=== 示例1: 生成测试集 ===")
    documents = evaluator.load_documents_from_local()
    if documents:
        testset = evaluator.generate_testset(documents, test_size=10)
        if testset:
            evaluator.save_testset(testset, "ragas_generated_testset.json")

    # 示例2: 加载已有测试集并评估
    print("\n=== 示例2: 执行评估 ===")
    testset = evaluator.load_testset("ragas_generated_testset.json")
    if testset:
        result = evaluator.evaluate(testset)
        evaluator.print_evaluation_report(result)


if __name__ == "__main__":
    main()