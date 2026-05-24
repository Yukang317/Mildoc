import sys
import subprocess
import re

# 确保支持 DataFrame.to_markdown（需要 tabulate）
def ensure_tabulate():
    try:
        import tabulate  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])

ensure_tabulate()

import tabula
import pandas as pd
from PyPDF2 import PdfReader

def get_number_of_pages(file_path):
    pdf = PdfReader(file_path)
    return len(pdf.pages)

file_path = "RAG_test_document.pdf"
total_pages = get_number_of_pages(file_path)

# 读第1页（带表头）
dfs_first = tabula.read_pdf(
    file_path, pages=[1], lattice=True,
    pandas_options={'dtype': str}
)
if not dfs_first or dfs_first[0].empty:
    raise RuntimeError("第1页未解析出表格，请检查 lattice/area/columns 参数或切换 stream=True")

first_df = dfs_first[0]
column_names = list(first_df.columns)

# 读剩余页（不把首行当表头）
dfs_rest = tabula.read_pdf(
    file_path, pages=list(range(2, total_pages + 1)), lattice=True,
    pandas_options={'header': None, 'dtype': str}
)

rest_frames = []
if dfs_rest:
    for df in dfs_rest:
        if df is None or df.empty:
            continue
        # 列数对齐
        if len(df.columns) < len(column_names):
            for _ in range(len(column_names) - len(df.columns)):
                df[_] = pd.NA
        elif len(df.columns) > len(column_names):
            df = df.iloc[:, :len(column_names)]
        df.columns = column_names
        rest_frames.append(df)

# 合并
frames = [first_df] + rest_frames
result = pd.concat(frames, ignore_index=True)


# === 新增：基础清洗函数 ===
def clean_cell(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # 避免破坏 MD 管道表格
    s = s.replace("|", "｜")
    # 合并多余空白
    s = re.sub(r"\s+", " ", s)
    return s

# 对合并后的整个表做清洗
result = result.applymap(clean_cell)

# === 新增：规范表头（把被拆的列名修正为标准版）===
# 你可按实际表头替换为最终想要的列名
standard_cols = ["申请编号","申请部门","采购类别","预算金额(元)","审批状态","负责人","申请日期"]

# 1) 先把列数量对齐到标准列数
if result.shape[1] < len(standard_cols):
    for _ in range(len(standard_cols) - result.shape[1]):
        result[_] = ""
    result = result.reindex(columns=list(range(len(standard_cols))))
elif result.shape[1] > len(standard_cols):
    result = result.iloc[:, :len(standard_cols)]

# 2) 赋标准列名（覆盖原始被拆/错误列名）
result.columns = standard_cols

# === 新增：合并“被拆成两行”的记录（尤其是申请日期列）===
merged_rows = []
i = 0
n = len(result)

# 判断某行是否“基本为空行”（核心字段全空）
def is_blank_core_row(row):
    core = [row["申请编号"], row["申请部门"], row["采购类别"]]
    return all(c == "" for c in core)



while i < n:
    row = result.iloc[i].copy()

    # 规则：申请日期如以 "-" 结尾（YYYY- 或 YYYY-MM-），且下一行该列是 1-2 位数字，则合并
    date_val = row["申请日期"]
    if re.match(r"^\d{4}-(\d{2}-)?$", date_val) and i + 1 < n:
        next_row = result.iloc[i+1]
        next_date = clean_cell(next_row["申请日期"])
        if re.match(r"^\d{1,2}$", next_date):
            # 合并日期
            row["申请日期"] = f"{date_val}{next_date.zfill(2)}"
            # 若当前行某些列为空而下一行有内容，顺带合并（防止列被拆到下一行）
            for col in standard_cols:
                if col == "申请日期":
                    continue
                if row[col] == "" and clean_cell(next_row[col]) != "":
                    row[col] = clean_cell(next_row[col])
            merged_rows.append(row)
            i += 2
            continue


    # 如果本行几乎是空且下一行是有效数据，可跳过本行（防残片进入）
    if is_blank_core_row(row) and i + 1 < n and not is_blank_core_row(result.iloc[i+1]):
        i += 1
        continue

    merged_rows.append(row)
    i += 1

result = pd.DataFrame(merged_rows, columns=standard_cols).reset_index(drop=True)

# 导出为 Markdown 表格
md_path = "tabula_merged.md"
with open(md_path, "w", encoding="utf-8") as f:
    # 可选：先写一个标题
    f.write("# 采购明细汇总\n\n")
    # 写入 Markdown 表格
    f.write(result.to_markdown(index=False))
    f.write("\n")

print(result.head())
print(f"已导出 Markdown 到: {md_path}（总行数: {len(result)}，总列数: {result.shape[1]}）")
