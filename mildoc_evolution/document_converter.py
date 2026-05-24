"""文档转换工具 - 将PDF/DOCX等格式转换为txt以便RAGAS评估使用"""

import os
import sys
from pathlib import Path
from typing import List, Optional

# ✅ 修复：使用上下文管理器避免全局sys.path污染
class MildocIndexImporter:
    """临时导入mildoc_index模块的上下文管理器"""
    
    def __init__(self):
        self.mildoc_index_path = Path(__file__).parent.parent / "mildoc_index"
        self.original_path = None
    
    def __enter__(self):
        self.original_path = sys.path.copy()
        if str(self.mildoc_index_path) not in sys.path:
            sys.path.insert(0, str(self.mildoc_index_path))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始sys.path
        sys.path[:] = self.original_path

try:
    with MildocIndexImporter():
        from parser.office_parser import OfficeParser
        from parser.pdf_parser import PDFParser
        PARSERS_AVAILABLE = True
except ImportError as e:
    print(f"警告：无法导入解析器: {e}")
    PARSERS_AVAILABLE = False


class DocumentConverter:
    """文档转换器 - 支持多种格式转txt"""
    
    def __init__(self):
        if PARSERS_AVAILABLE:
            self.office_parser = OfficeParser()
            self.pdf_parser = PDFParser()
            print("✅ 文档解析器初始化成功")
        else:
            print("⚠️  解析器不可用，将跳过格式转换")
    
    def convert_to_txt(self, file_path: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        将文档转换为txt格式
        
        Args:
            file_path: 输入文件路径
            output_dir: 输出目录，默认为输入文件同目录
            
        Returns:
            转换后的txt文件路径，失败返回None
        """
        input_path = Path(file_path)
        
        if not input_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return None
        
        # 确定输出目录
        if output_dir:
            output_path = Path(output_dir) / f"{input_path.stem}.txt"
        else:
            output_path = input_path.with_suffix('.txt')
        
        # 如果已存在且更新，跳过
        if output_path.exists() and output_path.stat().st_mtime >= input_path.stat().st_mtime:
            print(f"⏭️  跳过(已存在): {output_path.name}")
            return str(output_path)
        
        try:
            text_content = ""
            
            # 根据扩展名选择解析器
            ext = input_path.suffix.lower()
            
            if ext in ['.pdf']:
                print(f"📄 解析PDF: {input_path.name}")
                text_content = self._parse_pdf(str(input_path))
                
            elif ext in ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt']:
                print(f"📄 解析Office文档: {input_path.name}")
                text_content = self._parse_office(str(input_path))
                
            elif ext in ['.txt', '.md', '.markdown']:
                # 文本文件直接读取
                with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
                print(f"📄 读取文本文件: {input_path.name}")
                
            else:
                print(f"⚠️  不支持的格式: {ext}")
                return None
            
            if not text_content or len(text_content.strip()) == 0:
                print(f"⚠️  解析结果为空: {input_path.name}")
                return None
            
            # 保存为txt
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            print(f"✅ 转换成功: {input_path.name} -> {output_path.name} ({len(text_content)} chars)")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 转换失败 {input_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_pdf(self, file_path: str) -> str:
        """解析PDF文件"""
        try:
            # 读取为二进制数据
            with open(file_path, 'rb') as f:
                data = f.read()
            
            result = self.pdf_parser.parse(data)
            return result if result else ""
        except Exception as e:
            print(f"PDF解析错误: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _parse_office(self, file_path: str) -> str:
        """解析Office文档"""
        try:
            # 读取为二进制数据
            with open(file_path, 'rb') as f:
                data = f.read()
            
            result = self.office_parser.parse(data)
            return result if result else ""
        except Exception as e:
            print(f"Office文档解析错误: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def batch_convert(self, input_dir: str, output_dir: Optional[str] = None, 
                     extensions: List[str] = None) -> List[str]:
        """
        批量转换目录下的所有文档
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录，默认为输入目录
            extensions: 要转换的文件扩展名列表
            
        Returns:
            成功转换的文件列表
        """
        if extensions is None:
            extensions = ['.pdf', '.docx', '.doc', '.xlsx', '.pptx']
        
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ 目录不存在: {input_dir}")
            return []
        
        converted_files = []
        
        for ext in extensions:
            for file in input_path.glob(f'*{ext}'):
                result = self.convert_to_txt(str(file), output_dir)
                if result:
                    converted_files.append(result)
        
        print(f"\n📊 批量转换完成: {len(converted_files)}/{sum(1 for ext in extensions for _ in input_path.glob(f'*{ext}'))} 个文件")
        return converted_files


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='文档转换工具')
    parser.add_argument('input', help='输入文件或目录')
    parser.add_argument('-o', '--output', help='输出目录（可选）')
    parser.add_argument('--batch', action='store_true', help='批量转换目录')
    
    args = parser.parse_args()
    
    converter = DocumentConverter()
    
    if args.batch:
        # 批量转换
        converted = converter.batch_convert(args.input, args.output)
        print(f"\n✅ 共转换 {len(converted)} 个文件")
    else:
        # 单个文件转换
        result = converter.convert_to_txt(args.input, args.output)
        if result:
            print(f"\n✅ 转换成功: {result}")
        else:
            print("\n❌ 转换失败")
            sys.exit(1)


if __name__ == "__main__":
    main()
