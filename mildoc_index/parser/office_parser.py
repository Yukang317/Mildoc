import os, sys
from io import BytesIO
from markitdown import MarkItDown   # 用于解析 Office 文档（Word、Excel、PowerPoint）为 Markdown 格式


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser.document_parser import DocumentParser
from logger.logging import setup_logging

logger = setup_logging()




class OfficeParser(DocumentParser):
    """Office文档解析器，使用markitdown"""
    
    def __init__(self):
        """初始化markitdown实例"""
        self.markitdown = MarkItDown(enable_plugins=False)  # 禁用插件，提高性能
    
    def parse(self, data: bytes) -> str:
        """解析Office文档"""
        try:
            # 使用BytesIO创建文件类对象，返回一个 BytesID 类的实例，是一个内存中的二进制流对象
            file_stream = BytesIO(data)
            
            # 使用markitdown的convert_stream方法解析 - 用于将文件流转换为结构化对象，自动检测文件类型，包含文本内容、元数据等信息
            result = self.markitdown.convert_stream(file_stream)
            
            if result and hasattr(result, 'text_content'):
                return result.text_content.strip()
            else:
                logger.error("markitdown解析结果为空或格式异常")
                return ""
                
        except Exception as e:
            logger.error(f"Office文档解析失败: {e}")
            return ""
    
    def supports(self, content_type: str) -> bool:
        """检查是否支持Office文档格式"""
        supported_types = [
            # Word文档
            'application/msword',  # .doc
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx
            
            # Excel文档
            'application/vnd.ms-excel',  # .xls
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # .xlsx
            
            # PowerPoint文档
            'application/vnd.ms-powerpoint',  # .ppt
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',  # .pptx
            
            # PDF (markitdown也支持PDF)
            'application/pdf',
        ]
        
        return content_type.lower() in [t.lower() for t in supported_types]


if __name__ == "__main__":
    parser = OfficeParser()
    with open("../data/msdoc3.docx", "rb") as f:
        data = f.read()
    result = parser.parse(data)
    print(result[:2000])