#!/bin/bash

# 测试运行脚本
echo "=== 开始运行Milvus API测试 ==="

# 确保在项目根目录
cd /root/mildoc_202601 || exit 1

# 安装测试依赖
echo "安装测试依赖..."
uv add pytest pytest-cov

# 运行测试并生成覆盖率报告
echo "运行测试..."
uv run pytest test-milvus_api.py -v --cov=mildoc_index.milvus_api --cov-report=term-missing

echo "=== 测试完成 ==="