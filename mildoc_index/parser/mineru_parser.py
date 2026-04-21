# 专门使用 MinerU 库和 VLM（视觉语言模型）技术来解析 PDF 文档，处理图表、表格、图片中的文字。
import os, sys
from document_parser import DocumentParser
from dotenv import load_dotenv
# 导入MinerU相关模块
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2   # PDF 字节转换工具
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze  # VLM 文档分析工具
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make  # 生成内容
from mineru.utils.enum_class import MakeMode  # 内容生成模式枚举

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger.logging import setup_logging

load_dotenv()

logger = setup_logging()

class MinerUParser(DocumentParser):
    """MinerU文档解析器，专门用于解析PDF文档"""
    
    def __init__(self):
        """
        初始化MinerU解析器
        
        Args:
            server_url (str): MinerU服务器地址，默认从环境变量或使用默认值
        """
        self.server_url = os.getenv("MINERU_SERVER_URL")  # 获取 MinerU 服务器地址
        if not self.server_url:
            raise ValueError("MINERU_SERVER_URL 环境变量未设置")

        logger.info(f"MinerU服务器地址: {self.server_url}")
    
    def parse(self, data: bytes) -> str:
        """解析PDF文档"""
        try:
            logger.info("正在使用MinerU解析PDF文档...")
            
            # 转换PDF字节数据
            logger.info("正在转换PDF字节数据...")
            # 转换 PDF 二进制数据为字节流，用于 VLM 分析
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(data, 0, None)
            
            # 使用VLM分析PDF
            logger.info("正在分析PDF文档...")
            # middle_json 存储 VLM 分析 PDF 后的详细结果
            middle_json, infer_result = vlm_doc_analyze(
                pdf_bytes,  # 转换后的 PDF 字节数据
                backend="http-client",  # 使用HTTP客户端后端
                server_url=self.server_url, 
                image_writer=None  # 不写入图像
            )
            
            # 获取PDF信息
            pdf_info = middle_json["pdf_info"]  # 从分析结果中提取 PDF 信息
            
            # 生成Markdown内容 - MakeMode.MM_MD
            logger.info("正在生成Markdown内容...") 
            # 根据 pdf_info 中的结构化信息，按照指定的模式（这里是 Markdown）生成格式化的内容
            md_content_str = vlm_union_make(pdf_info, MakeMode.MM_MD, None)
            
            logger.info(f"MinerU解析完成，生成了{len(md_content_str)}字符的Markdown内容")
            return md_content_str.strip()
            
        except ImportError as e:
            logger.error(f"MinerU模块导入失败: {e}")
            logger.error("请确保已正确安装MinerU相关依赖，包括MinerU CLI和MinerU VLM后端")
            logger.error("请参考MinerU官方文档进行安装和配置")
            logger.error("例如：https://github.com/MinerU/MinerU")
            logger.error("https://github.com/MinerU/MinerU/tree/main/MinerU-CLI")
            return ""
        except Exception as e:
            logger.error(f"MinerU解析失败: {e}")
            return ""
    
    def supports(self, content_type: str) -> bool:
        """检查是否支持PDF格式"""
        return content_type.lower() in ['application/pdf', 'pdf']


if __name__ == "__main__":
    parser = MinerUParser()
    with open("../data/pdf3.pdf", "rb") as f:
        data = f.read()
    result = parser.parse(data)
    print("-"*100)
    print(result)