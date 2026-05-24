# RAGAS Chunk级别评估系统 - 使用指南

> **版本**: 5.2 (支持阿里云通义千问 + Chunk级别评估)  
> **最后更新**: 2026-04-23

---

## 🚀 快速开始（3步）

### Step 1: 环境准备

```bash
cd /root/mildoc_202601/mildoc_evolution
uv sync  # 安装依赖
```

### Step 2: 配置 `.env` 文件

```bash
# OSS配置
OSS_ACCESS_KEY_ID=your_ak
OSS_ACCESS_KEY_SECRET=your_sk
OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com
OSS_BUCKET_NAME=mildoc-yu

# LLM配置（关键：必须是阿里云模型）
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-plus              # ✅ 推荐

# Embedding配置
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL_NAME=text-embedding-v4
```

### Step 3: 执行评估

```bash
# 方式1：评估现有测试集（推荐）
uv run python3 main.py --action evaluate --dataset ragas_format_testset_with_chunks.json

# 方式2：从本地文档生成测试集并评估
uv run python3 main.py --action full --source local --test-size 5

# 方式3：从OSS文档生成测试集并评估（自动转换PDF/DOCX）
uv run python3 main.py --action full --source oss --test-size 5
```

---

## ✨ 核心特性

### 🔬 真正的Chunk级别评估

**传统方法的问题**：
```python
# ❌ 旧方法：文档级别（只要文档中有任意chunk被召回就算成功）
if doc_name in expected_docs:
    relevant_count += 1  # 每个文档只计1次，不精确
```

**我们的改进**：
```python
# ✅ 新方法：真正的Chunk级别（精确到每个文本块）
if chunk_id in expected_chunk_ids:
    relevant_count += 1  # 每个chunk独立计数，更精确
```

### 📊 双维度评估体系

| 维度 | 指标 | 说明 |
|------|------|------|
| **RAGAS指标** | Context Recall/Precision | LLM评估的语义相关性 |
| (LLM评判) | Faithfulness | 回答是否基于检索内容（无幻觉） |
| | Answer Relevancy/Correctness | 答案质量和正确性 |
| **Chunk级别指标** | Recall/Precision/F1 | 精确的文本块检索质量 |
| (精确计算) | MRR | 第一个相关chunk的排名 |
| | Recall@K | Top K结果的查全率 |
| | NDCG@K | 排序质量综合指标 |

### 🎯 智能文档加载

- ✅ 支持 OSS 根目录和 `documents/` 前缀
- ✅ 自动识别 `.txt`, `.md`, `.pdf`, `.docx` 等格式
- ✅ PDF/DOCX 自动转换为 TXT（集成项目解析器）
- ✅ 增量下载（跳过已存在且未变化的文件）

---

## 📖 完整评估流程

### 📂 系统架构

```
mildoc_evolution/
├── 📄 核心代码
│   ├── main.py                  # 主入口
│   ├── ragas_evaluator.py       # RAGAS评估器（支持Chunk级）
│   ├── chunk_metrics.py         # Chunk级别指标计算
│   ├── data_loader.py           # 数据加载
│   └── config.py                # 配置管理
│
├── 📊 测试数据
│   ├── test_data/
│   │   ├── ragas_format_testset.json                    # 文档级标注示例
│   │   └── ragas_format_testset_with_chunks.json        # ✨ Chunk级标注示例
│   └── documents/               # 本地文档目录（从OSS下载）
│
├── 📈 输出
│   ├── ragas_output/            # RAGAS评估结果JSON
│   └── reports/                 # 传统报告（保留但不再使用）
│
└── 📝 文档
    └── USAGE_GUIDE.md           # 本文件
```

### 🔄 数据流

```
OSS/本地文档 
    ↓
load_documents_from_oss/local()
    ↓
Langchain Document Loader
    ↓
TestsetGenerator.generate_with_langchain_docs()
    ↓
测试集 (Dataset)
    ↓
run_rag_query() → mildoc_wxkf/rag_service.py
    ↓
RAGAS evaluate(llm=self.llm) + Chunk Metrics
    ↓
评估报告 (JSON + 控制台输出)
```

### 📋 详细步骤

#### Step 1: 文档准备

**OSS方式**（推荐，自动转换PDF/DOCX）：
```bash
uv run python3 main.py --action load --source oss
```

系统会自动：
- 扫描 OSS 根目录和 `documents/` 前缀
- 下载 `.txt`, `.md` 文件
- 转换 `.pdf`, `.docx` 为 `.txt`（使用项目解析器）
- 保存到 `documents/` 目录

**本地方式**：
```bash
# 将文档复制到 documents/ 目录
cp /path/to/your/*.txt documents/
```

#### Step 2: 准备测试集

使用现有的测试集：
```bash
ls test_data/
# ragas_format_testset.json                  # 简单格式
# ragas_format_testset_with_chunks.json      # Chunk级别标注
```

或手动编辑添加新问题：
```json
{
    "samples": [
        // ... 现有样本 ...
        
        // 添加新样本
        {
            "question": "你的新问题",
            "expected_doc_source": "相关文档.pdf",
            "ground_truth": "标准答案",
            "metadata": {"category": "分类", "difficulty": "难度"}
        }
    ]
}
```

#### Step 3: 执行评估

```bash
# 方式1：评估现有测试集（推荐）
uv run python3 main.py --action evaluate --dataset ragas_format_testset_with_chunks.json

# 方式2：完整流程（生成测试集 + 评估，耗时较长）
uv run python3 main.py --action full --source local --test-size 5
```

**过程**：
1. 对每个问题调用 RAG 系统（`mildoc_wxkf/rag_service.py`）
2. 收集检索到的 chunks 和生成的答案
3. 计算 RAGAS 指标（LLM评估，使用 qwen-plus）
4. 计算 Chunk 级别指标（精确计算）
5. 保存结果

#### Step 4: 查看结果

**控制台输出**：
```
============================================================
         RAGAS 评估报告 (Chunk级别)
============================================================

【RAGAS 核心指标】
  上下文召回率 (Context Recall):     0.8500
  上下文精确率 (Context Precision):  0.7200
  忠实度 (Faithfulness):             0.9100
  答案相关性 (Answer Relevancy):     0.8800
  答案正确性 (Answer Correctness):   0.8200

【Chunk 级别指标】
  平均召回率 (Avg Recall):         0.6667
  平均准确率 (Avg Precision):      0.5000
  平均F1值 (Avg F1):               0.5714
  平均MRR (Avg MRR):               0.7500
  Recall@1:                        0.3333
  Recall@3:                        0.6667
  Recall@5:                        0.6667
  NDCG@5:                          0.6200

============================================================
```

**JSON文件**：
- 位置：`ragas_output/ragas_evaluation_result_YYYYMMDD_HHMMSS.json`
- 包含完整的指标数据和每个样本的详细结果

---

## 📝 测试集编写规范

### 格式 1：简单格式（适合快速测试）

```json
{
    "name": "基础测试集",
    "samples": [
        {
            "question": "怀孕7个月可以加班吗？",
            "ground_truth": "不合法，劳动法规定不得安排怀孕7个月以上女职工加班",
            "expected_doc_source": "劳动法.pdf"
        }
    ]
}
```

**特点**：
- ✅ 简单易写
- ✅ 只需指定文档名
- ⚠️ 只能计算 RAGAS 核心指标，不能计算 Chunk 级别指标

### 格式 2：Chunk 级别标注（推荐，更精确）

```json
{
    "name": "Chunk级别测试集",
    "samples": [
        {
            "question": "怀孕7个月可以加班吗？",
            "expected_doc_source": "劳动法.pdf",
            
            // ⚠️ 关键：expected_chunks 必须与 Milvus 中的真实 chunk 匹配
            "expected_chunks": [
                {
                    "chunk_id": "劳动法.pdf_chunk_15",  // 格式：文件名_chunk_序号
                    "content": "第六十一条 对怀孕七个月以上的女职工，不得安排其延长工作时间和夜班劳动。",
                    "doc_name": "劳动法.pdf"
                },
                {
                    "chunk_id": "劳动法.pdf_chunk_16",
                    "content": "用人单位违反本法规定，延长劳动者工作时间的，由劳动行政部门给予警告...",
                    "doc_name": "劳动法.pdf"
                }
            ],
            
            "ground_truth": "不合法，根据《劳动法》第六十一条...",
            "metadata": {"category": "法律应用", "difficulty": "中等"}
        }
    ]
}
```

**特点**：
- ✅ 精确到每个文本块
- ✅ 可以计算所有指标（包括 Recall@K, MRR, NDCG）
- ⚠️ 需要知道真实的 chunk ID

### 🎯 如何让 expected_chunks 匹配 Milvus？

#### 步骤 1：了解 Milvus 的 Chunk ID 格式

Milvus 中的 chunk ID 通常格式为：
```
{doc_name}_chunk_{index}
```

例如：
- `劳动法.pdf_chunk_0`
- `劳动法.pdf_chunk_1`
- `人事管理流程.docx_chunk_5`

#### 步骤 2：查看实际的 Chunk ID

运行以下命令查询 Milvus：

```bash
cd /root/mildoc_202601/mildoc_index
uv run python3 -c "
from milvus_api import MilvusAPI
api = MilvusAPI()
results = api.search('怀孕', top_k=5)
for r in results:
    print(f'Chunk ID: {r.id}')
    print(f'Doc Name: {r.metadata.get(\"doc_name\")}')
    print(f'Content: {r.page_content[:100]}')
    print('---')
"
```

#### 步骤 3：编写测试集

根据实际的 chunk ID 编写：

```json
{
    "question": "怀孕7个月可以加班吗？",
    "expected_chunks": [
        {
            "chunk_id": "劳动法.pdf_chunk_15",  // ⚠️ 必须是真实的 ID
            "content": "第六十一条 对怀孕七个月以上的女职工...",
            "doc_name": "劳动法.pdf"
        }
    ]
}
```

### 💡 简化方案

如果你不确定 chunk ID，可以**只使用文档级标注**：

```json
{
    "question": "怀孕7个月可以加班吗？",
    "expected_doc_source": "劳动法.pdf",  // 只需要指定文档名
    "ground_truth": "不合法..."
}
```

这样只会计算 RAGAS 核心指标，Chunk 级别指标会显示为 0。

---

## 📈 指标解读

### RAGAS 核心指标（0-1分，越高越好）

| 指标 | 优秀 | 良好 | 需改进 | 说明 |
|------|------|------|--------|------|
| **Context Recall** | >0.85 | 0.70-0.85 | <0.70 | 检索器能否找到所有相关信息 |
| **Context Precision** | >0.85 | 0.70-0.85 | <0.70 | 检索到的内容是否都相关 |
| **Faithfulness** | >0.90 | 0.80-0.90 | <0.80 | 回答是否有幻觉 |
| **Answer Relevancy** | >0.85 | 0.70-0.85 | <0.70 | 回答是否切题 |
| **Answer Correctness** | >0.85 | 0.70-0.85 | <0.70 | 回答的事实正确性 |

### Chunk 级别指标

| 指标 | 优秀 | 良好 | 需改进 | 说明 |
|------|------|------|--------|------|
| **Recall** | >0.80 | 0.60-0.80 | <0.60 | 召回的相关chunk比例 |
| **Precision** | >0.80 | 0.60-0.80 | <0.60 | 检索结果的准确度 |
| **F1** | >0.75 | 0.55-0.75 | <0.55 | Recall和Precision的平衡 |
| **MRR** | >0.80 | 0.60-0.80 | <0.60 | 第一个相关chunk的排名 |
| **Recall@3** | >0.70 | 0.50-0.70 | <0.50 | Top 3的查全率 |
| **NDCG@5** | >0.80 | 0.60-0.80 | <0.60 | 前5个结果的排序质量 |

### 🛠️ 优化建议

**如果 Recall 低**：
- 增加检索的 top_k 值
- 优化 embedding 模型
- 改进文档分块策略

**如果 Precision 低**：
- 添加重排序（Rerank）步骤
- 调整相似度阈值
- 优化查询改写

**如果 Faithfulness 低**：
- 改进 prompt，强调"基于检索内容回答"
- 限制生成温度（temperature）
- 添加引用来源要求

**如果 MRR 低**：
- 优化向量检索算法
- 添加关键词检索（混合搜索）
- 改进查询理解

---

## 🔧 常见问题与解决

### Q1: RAGAS 指标都是 NaN

**现象**：
```
上下文召回率 (Context Recall):     nan
忠实度 (Faithfulness):             nan
```

**原因**：LLM 配置错误，RAGAS 尝试使用 `gpt-4o-mini` 但阿里云不支持。

**解决**：
1. 检查 `.env` 中 `LLM_MODEL_NAME=qwen-plus`
2. 确保 `ragas_evaluator.py` 的 `evaluate()` 传入了 `llm=self.llm`（✅ 已修复）

### Q2: Chunk 级别指标都是 0

**现象**：
```
平均召回率 (Avg Recall):         0.0000
平均MRR (Avg MRR):               0.0000
```

**原因**：`expected_chunks` 中的 `chunk_id` 与 Milvus 中的实际 ID 不匹配。

**解决**：
1. 查询 Milvus 获取真实的 chunk ID
2. 或者改用 `expected_doc_source`（文档级标注）

### Q3: 如何验证 RAG 系统是否正常？

先测试单个问题：

```bash
cd /root/mildoc_202601/mildoc_wxkf
uv run python3 evaluate_rag.py --question "怀孕7个月可以加班吗？" --output-json
```

如果能返回答案，说明 RAG 系统正常，可以进行 RAGAS 评估。

### Q4: 评估太慢怎么办？

**优化方法**：
1. 减少测试集大小（3-5个问题即可）
2. 使用 `qwen-turbo` 代替 `qwen-plus`（更快但稍弱）
3. 跳过 Chunk 级别指标（只用简单格式测试集）

### Q5: 为什么我的 PDF/DOCX 文档没有被加载？

**A**: RAGAS 测试集生成器只支持文本格式。系统会自动转换：

```bash
# 从 OSS 加载时自动转换
uv run python3 main.py --action load --source oss

# 或手动转换
uv run python3 document_converter.py document.pdf
```

### Q6: 依赖被 `uv sync` 卸载

**现象**：
```bash
uv pip install ragas  # 安装成功
uv sync               # 又卸载了
```

**原因**：根目录 `pyproject.toml` 未包含 ragas 依赖。

**解决**：已在 `/root/mildoc_202601/pyproject.toml` 中添加依赖。

### Q7: OSS 端点配置错误

**现象**：
```
SSL: CERTIFICATE_VERIFY_FAILED
Hostname mismatch
```

**原因**：使用了 MNS 端点而非 OSS 端点。

**解决**：
```bash
# ❌ 错误
OSS_ENDPOINT=https://1636250688038240.mns.cn-hangzhou.aliyuncs.com

# ✅ 正确
OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com
```

### Q8: PDF/DOCX 解析失败

**现象**：
```
PDF解析失败: a bytes-like object is required, not 'str'
```

**原因**：解析器的 `parse()` 方法期望 bytes 数据，但传入了文件路径字符串。

**解决**：已在 `document_converter.py` 中修复，现在正确读取文件为 bytes。

---

## 🔍 技术细节

### RAGAS LLM 配置详解

#### 为什么需要配置 LLM？

RAGAS 在评估时需要使用 LLM 来判断答案质量（如忠实度、相关性等）。默认使用 OpenAI 的 `gpt-4o-mini`，但阿里云 API 不支持这个模型，会导致 **404 错误**。

#### 配置位置

**① 环境变量（`.env` 文件）**：
```bash
LLM_MODEL_NAME=qwen-plus              # ✅ 必须是阿里云支持的模型
```

**支持的阿里云模型**：
- ✅ `qwen-plus`（推荐，性能好）
- ✅ `qwen-turbo`（速度快）
- ✅ `qwen-max`（最强，但贵）
- ❌ `gpt-4o-mini`（OpenAI 模型，不支持）

**② 代码配置（`ragas_evaluator.py`）**：
```python
# 初始化 LLM
self.llm = ChatOpenAI(
    model_name=settings.LLM_MODEL_NAME,      # 从 .env 读取 qwen-plus
    openai_api_key=settings.LLM_API_KEY,
    openai_api_base=settings.LLM_BASE_URL,
    temperature=0.0
)

# 执行评估时传入 LLM（关键！）
scores = evaluate(
    dataset=eval_dataset,
    metrics=[context_recall, context_precision, ...],
    llm=self.llm,              # ⚠️ 必须传入自定义 LLM
    embeddings=self.embeddings  # ⚠️ 必须传入自定义 Embedding
)
```

### 关键修复说明

#### 1. NaN 值问题修复

**问题**：`answer_relevancy` 和 `answer_correctness` 指标返回 NaN

**原因**：测试集加载时 `contexts` 字段为空列表

**解决**：从 RAG 检索结果中提取真实 contexts，并添加失败样本追踪机制

#### 2. 解耦 subprocess 调用

**问题**：硬编码绝对路径导致模块间强耦合

**解决**：改为直接导入 `mildoc_wxkf.rag_service` 模块，动态路径计算

#### 3. 完善异常处理

**改进**：
- ✅ 失败样本追踪与报告
- ✅ NaN/Inf 值清理（JSON序列化前）
- ✅ 详细的错误日志和堆栈跟踪
- ✅ sys.path 污染修复（使用上下文管理器）

---

## 📞 技术支持

### 关键文件

- **配置文件**：`.env`
- **评估器**：`ragas_evaluator.py`
- **测试集**：`test_data/ragas_format_testset_with_chunks.json`
- **主入口**：`main.py`
- **文档转换器**：`document_converter.py`
- **Chunk指标计算**：`chunk_metrics.py`

### 验证命令

```bash
# 运行单元测试
uv run python3 test_chunk_metrics.py

# 检查依赖
uv pip list | grep -E "ragas|datasets|oss2|rapidfuzz"

# 测试OSS连接
uv run python3 -c "from ragas_evaluator import RagasEvaluator; e=RagasEvaluator(); docs=e.load_documents_from_oss(); print(f'Loaded {len(docs)} docs')"

# 测试RAG服务
cd ../mildoc_wxkf && uv run python3 evaluate_rag.py --question "测试" --output-json
```

### 注意事项

1. **依赖要求**：确保已安装 ragas, langchain, datasets 等依赖
2. **环境变量**：`.env` 文件中必须配置完整的 LLM 和 Embedding 参数
3. **mildoc_wxkf 模块**：确保该目录存在且 `rag_service.py` 可正常导入
4. **Milvus 数据库**：确保正在运行并可连接
5. **Chunk 级别指标**：当前测试集可能不包含 `expected_chunks` 字段，Chunk 级别指标将显示为 `null`（这是预期行为）

---

## 🎯 总结

**三步完成 RAGAS 评估**：

1. ✅ **配置 LLM**：`.env` 中设置 `LLM_MODEL_NAME=qwen-plus`
2. ✅ **准备测试集**：编写 JSON 格式的测试问题
3. ✅ **执行评估**：`uv run python3 main.py --action evaluate --dataset xxx.json`

**关键点**：
- RAGAS 评估时必须传入 `llm=self.llm`（✅ 已修复）
- 测试集的 `expected_chunks` 需要匹配真实的 chunk ID
- 如果不确定 chunk ID，使用 `expected_doc_source` 即可
- OSS 加载支持自动转换 PDF/DOCX（✅ 已实现）
- 完善的异常处理和失败样本追踪（✅ 已实现）

**预期指标范围**：

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| Context Recall | >0.85 | 0.70-0.85 | <0.70 |
| Faithfulness | >0.90 | 0.80-0.90 | <0.80 |
| Chunk Recall | >0.80 | 0.60-0.80 | <0.60 |
| MRR | >0.80 | 0.60-0.80 | <0.60 |

祝评估顺利！🚀
