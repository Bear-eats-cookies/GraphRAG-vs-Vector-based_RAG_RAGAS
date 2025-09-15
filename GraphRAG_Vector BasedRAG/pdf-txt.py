# -*- coding: utf-8 -*-            
# @Time : 2025/9/5 19:19
#  : 殷韵智
# @FileName: pdf-txt.py
# @Software: PyCharm
# 将 PDF 文件转换为 TXT 文件
import pdfplumber
"""
为什么要转换
    1、文本提取的便利性（
    PDF 文件通常包含复杂的格式（如表格、图像、页眉页脚），直接处理可能需要专门的解析工具（如 PyPDF2 或 pdfplumber），
        而这些工具可能无法完全准确提取结构化文本。
    转换为 TXT 文件可以提取纯文本内容，去除格式干扰，便于后续处理（如分词、实体提取或向量嵌入）。
    ）
    2、统一数据格式：
    项目同时实现了 GraphRAG 和 Vector-based RAG，为保证公平比较，两种方法需要使用相同的输入数据。
    TXT 文件是一种通用的中间格式，适合两种方法的预处理：
        GraphRAG：将 TXT 文件输入到索引流程，生成知识图谱（存储为 Parquet 和 LanceDB）。
        Vector-based RAG：将 TXT 文件分块后生成嵌入，存储到 FAISS 向量数据库。
    同时：txt数据处理更简单，减少了因格式问题导致的信息丢失或解析错误。
"""
def pdf_to_txt(pdf_path, txt_path):
    """将 PDF 文件转换为 TXT 文件"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        print(f"PDF 已转换为 TXT: {txt_path}")
    except Exception as e:
        print(f"转换失败: {e}")

# 示例使用
pdf_path = "unh_earnings_report.pdf"
txt_path = "ragtest/ragtest/input/unh_data.txt"
pdf_to_txt(pdf_path, txt_path)
