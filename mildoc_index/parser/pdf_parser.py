# 提供内存中的字节流处理，用于将二进制数据转换为文件对象
from io import BytesIO
# PDF 文档读取库，用于解析 PDF 文件内容
from PyPDF2 import PdfReader    

import sys, os

# 确保可以导入项目中的其他模块（向上两级目录获取当前文件“__file__”的绝对路径）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser.document_parser import DocumentParser
from logger.logging import setup_logging

logger = setup_logging()

class PDFParser(DocumentParser):
    """PDF文档解析器"""
    
    def parse(self, data: bytes) -> str:    # 接收 PDF 二进制数据，返回文本内容
        """解析PDF文档"""
        try:
            reader = PdfReader(BytesIO(data))   # 二进制数据转为文件对象后被读取 PDF 内容
            text_content = ""
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()  # 从 PDF 页面提取文本内容，返回字符串
                if page_text:
                    text_content += page_text + "\n"
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"PDF解析失败: {e}")
            return ""
    
    def supports(self, content_type: str) -> bool:
        """检查是否支持PDF"""
        # 支持两种常见的 PDF 内容类型表示
        return content_type.lower() in ['application/pdf', 'pdf']


if __name__ == "__main__":
    parser = PDFParser()

    file_path = ["../data/pdf1.pdf", "../data/pdf2.pdf", "../data/pdf3.pdf", "../data/pdf4.pdf"]
    for path in file_path:
        with open(path, "rb") as f:
            data = f.read()
            result = parser.parse(data)
            print("\n\n")
            print(f"#################   文件路径: {path}    #################")
            print(result)



# - 继承自 DocumentParser 抽象基类，实现了标准接口
# - 被文档处理模块调用，用于解析上传的 PDF 文件
# - 为 embedding.py 模块提供标准化的文本输入，用于向量化处理