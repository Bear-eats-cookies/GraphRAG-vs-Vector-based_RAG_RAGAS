# -*- coding: utf-8 -*-
# @Time : 2025/9/5 16:24
# @Author : 殷韵智
# @FileName: vector_rag.py
# @Software: PyCharm
# Vector-based RAG 实现

"""
Vector-based RAG (Retrieval-Augmented Generation) 系统

这个脚本实现了基于向量相似性的检索增强生成系统，主要功能包括：
1. 文档加载和预处理（分块）
2. 文档向量化和索引构建
3. 基于语义相似性的文档检索
4. 检索结果重排序优化
5. 结合检索内容生成准确回答

与 GraphRAG 的区别：
- GraphRAG：基于知识图谱的结构化检索，适合复杂关系推理
- Vector RAG：基于语义相似性的文档检索，适合直接信息查找

主要组件：
- 文档分块器：将长文档切分为可管理的片段
- 向量存储：使用 FAISS 进行高效的相似性搜索
- 重排序器：使用 FlashrankRerank 优化检索结果排序
- 生成器：使用 GPT 模型基于检索内容生成回答

技术架构：
- 数据层：原始文档的分块处理
- 嵌入层：文档向量化和相似性计算
- 存储层：FAISS 向量数据库
- 检索层：MMR 算法和重排序优化
- 生成层：基于上下文的 LLM 回答生成
"""
"""
代码功能总结
这个 Vector-based RAG（基于向量的检索增强生成）系统的主要作用包括：

🎯 核心功能
文档预处理 - 加载文本文件并智能分块，保持上下文连续性
向量化索引 - 使用 OpenAI 嵌入模型将文档转换为语义向量
高效检索 - 基于 FAISS 向量数据库进行快速相似性搜索
结果优化 - 使用 MMR 算法和 FlashrankRerank 提升检索质量
智能问答 - 结合检索内容使用 GPT 模型生成准确回答
🔧 技术架构
数据层：文档分块和预处理
嵌入层：text-embedding-3-small 模型进行向量化
存储层：FAISS 向量数据库支持高效搜索
检索层：MMR 算法 + FlashrankRerank 重排序
生成层：GPT-4o-mini 基于上下文生成回答
🚀 工作流程
加载原始文档并分割为 700 字符的块（50 字符重叠）
使用嵌入模型将所有文档块转换为向量
构建 FAISS 向量数据库并创建检索器
接收用户查询并转换为向量表示
使用余弦相似度搜索最相关的文档块
应用 MMR 算法和重排序优化结果
选择前 3 个最相关文档作为上下文
使用 LLM 基于检索内容生成回答
💡 系统优势
语义理解：不依赖关键词匹配，理解查询的真实意图
快速检索：FAISS 提供毫秒级的向量搜索性能
精确匹配：重排序机制进一步提升相关性
上下文保持：文档分块重叠确保信息完整性
扩展性强：可处理大规模文档集合
这个系统特别适合处理基于文档的问答任务，如财务报告分析、技术文档查询、知识库检索等场景。

"""
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage

# ==================== 配置部分 ====================
# 配置 OpenAI API 环境变量
# 注意：在生产环境中应该使用更安全的方式管理 API 密钥
os.environ["OPENAI_API_KEY"] = 'OPENAI_API_KEY'
os.environ["OPENAI_API_BASE"] = 'OPENAI_API_BASE'

# ==================== 模型初始化 ====================
# 初始化大语言模型 (LLM)
# 用于基于检索到的文档内容生成最终回答
llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"],
    model="gpt-4o-mini"                     # 使用 GPT-4o-mini 模型进行文本生成
)

# 初始化嵌入模型
# 用于将文档和查询转换为向量表示，支持语义相似性计算
embedding_model = OpenAIEmbeddings(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"],
    model="text-embedding-3-small"          # 使用 text-embedding-3-small 进行向量化
)

# ==================== 文档加载与预处理 ====================
# 加载原始文档
# TextLoader 用于读取纯文本文件，支持指定编码格式
loader = TextLoader("ragtest/ragtest/input/unh_data.txt", encoding='utf-8')
pages = loader.load()

# 文档分块处理
# RecursiveCharacterTextSplitter 递归地将长文档分割为较小的块
# 这样做的目的：
# 1. 确保每个块的大小适合嵌入模型处理
# 2. 提高检索的精确性（避免无关信息干扰）
# 3. 控制生成时的上下文长度
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,                         # 每个文档块的最大字符数
    chunk_overlap=50,                       # 相邻块之间的重叠字符数（保持上下文连续性）
    length_function=len,                    # 使用字符长度作为分割依据
    is_separator_regex=False                # 分隔符不使用正则表达式
)

# 执行文档分块
docs = text_splitter.split_documents(pages)
print(f"📄 文档已分割为 {len(docs)} 个块")

# ==================== 向量数据库构建 ====================
# 使用 FAISS 构建向量数据库
# FAISS (Facebook AI Similarity Search) 是一个高效的相似性搜索库
# 功能：将文档块转换为向量并建立索引，支持快速相似性搜索
faiss_db = FAISS.from_documents(
    docs,                                   # 输入的文档块列表
    embedding_model,                        # 嵌入模型，用于向量化文档
    distance_strategy=DistanceStrategy.COSINE  # 使用余弦相似度作为距离度量
)

# 创建检索器
# 检索器负责根据查询找到最相关的文档块
retriever = faiss_db.as_retriever(
    search_kwargs={"k": 10},                # 检索前 10 个最相关的文档块
    search_type="mmr"                       # 使用 MMR (Maximal Marginal Relevance) 算法
)                                          # MMR 在相关性和多样性之间取得平衡，避免检索结果过于相似

print("🔍 FAISS 向量数据库构建完成")

# ==================== 检索结果重排序 ====================
# 配置 FlashrankRerank 重排序器
# FlashrankRerank 是一个轻量级的重排序模型，用于优化检索结果的排序
# 作用：在初步检索结果的基础上，进一步提升最相关文档的排名
try:
    import flashrank
    from flashrank import Ranker
    
    # 初始化 Flashrank 排序器
    ranker = Ranker()
    compressor = FlashrankRerank(model=ranker)
    
    print("✅ Flashrank 重排序器已启用")
except ImportError:
    # 如果 Flashrank 未安装，则使用默认检索器
    print("⚠️ Flashrank 未安装，使用默认检索器")
    compression_retriever = retriever
else:
    # 创建上下文压缩检索器
    # 结合基础检索器和重排序器，提供更精确的检索结果
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,         # 重排序压缩器
        base_retriever=retriever            # 基础检索器
    )
    print("✅ 上下文压缩检索器已配置")

# ==================== 查询执行函数 ====================
def run_vector_rag_query(query):
    """
    运行基于向量的 RAG 查询，返回生成的回答
    
    Args:
        query (str): 用户的查询问题
    
    Returns:
        str: 基于检索文档生成的回答
    
    工作流程：
    1. 将查询转换为向量表示
    2. 在向量数据库中搜索最相关的文档块
    3. 使用重排序器优化检索结果
    4. 选择前3个最相关的文档作为上下文
    5. 构建包含查询和文档的提示
    6. 使用 LLM 基于检索内容生成回答
    
    技术特点：
    - 语义搜索：基于向量相似性而非关键词匹配
    - 上下文优化：通过重排序提升相关性
    - 多文档融合：整合多个相关文档片段
    - 智能生成：LLM 基于上下文进行推理和回答
    """
    # 步骤1-3：检索相关文档（已由 compression_retriever 完成）
    # 这一步包括：查询向量化 → 相似性搜索 → 重排序优化
    docs = compression_retriever.invoke(query)
    
    # 步骤4：选择前3个最相关的文档并组合为上下文
    # 使用分隔符连接多个文档，便于 LLM 理解文档边界
    base_document = "\n---------".join([d.page_content for d in docs[:3]])
    
    # 步骤5：构建对话消息
    messages = [
        # 系统消息：指导 LLM 如何使用提供的文档
        SystemMessage(content="Given document, use these documents to answer query"),
        # 用户消息：包含查询和检索到的相关文档
        HumanMessage(content=f"Query: {query}\nDocument: {base_document}")
    ]
    
    # 步骤6：使用 LLM 生成回答
    ai_msg = llm.invoke(messages)
    return ai_msg.content

# ==================== 演示和测试 ====================
if __name__ == "__main__":
    """
    Vector-based RAG 系统演示
    
    功能说明：
    - 展示如何使用向量 RAG 系统回答基于文档的问题
    - 示例查询涉及财务数据的具体分析
    - 演示系统如何从大量文档中精确检索相关信息
    
    系统优势：
    1. 快速检索：基于向量相似性的高效搜索
    2. 精确匹配：语义理解能力强，不依赖关键词匹配
    3. 上下文保持：通过文档分块和重叠保持信息完整性
    4. 结果优化：重排序机制提升检索质量
    5. 扩展性强：可处理大规模文档集合
    
    适用场景：
    - 文档问答系统
    - 知识库检索
    - 技术文档查询
    - 法律条文检索
    - 学术论文分析
    """
    
    # 示例查询：关于 UnitedHealth Group 财务变化的复杂问题
    query = "What factors contributed to the changes in UnitedHealth Group's revenues and net earnings from 2023 to 2024, and how did these changes affect the earnings per share attributable to UnitedHealth Group common shareholders?"
    
    print("🚀 Vector-based RAG 系统启动")
    print("🔍 开始执行查询...")
    print(f"📝 查询问题: {query}")
    print("\n" + "="*80)
    
    # 执行查询并获取回答
    response = run_vector_rag_query(query)
    
    print(f"🤖 Vector-based RAG 回答:")
    print(f"{response}")
    print("\n" + "="*80)
    print("✅ 查询完成")