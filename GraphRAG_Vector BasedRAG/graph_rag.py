# -*- coding: utf-8 -*-
# @Time : 2025/9/5 16:24
# @Author : 殷韵智
# @FileName: graph_rag.py
# @Software: PyCharm
# GraphRAG 实现：基于知识图谱的检索增强生成

"""
GraphRAG (Graph Retrieval-Augmented Generation) 实现

这个脚本实现了基于知识图谱的检索增强生成系统，主要功能包括：
1. 加载和处理知识图谱数据（实体、关系、社区报告等）
2. 生成实体的语义嵌入向量
3. 构建向量数据库用于相似性搜索
4. 实现本地搜索引擎，结合图谱结构和语义搜索
5. 提供异步查询接口，支持复杂问题的回答

主要组件：
- 实体（Entities）：知识图谱中的节点，包含描述和嵌入向量
- 关系（Relationships）：实体间的连接关系
- 社区报告（Community Reports）：图谱中社区的摘要信息
- 文本单元（Text Units）：原始文档的分块文本
- 向量存储：用于语义相似性搜索的向量数据库
"""

"""
代码功能总结
这个 GraphRAG（图检索增强生成）系统的主要作用包括：

🎯 核心功能
知识图谱构建与加载 - 从预处理的数据文件中加载实体、关系、社区报告等结构化知识
语义嵌入生成 - 将实体描述转换为高维向量，支持语义相似性搜索
向量数据库管理 - 使用 LanceDB 存储和检索嵌入向量
混合上下文构建 - 结合图谱结构和语义搜索构建查询上下文
智能问答系统 - 基于知识图谱回答复杂的多实体、多关系问题
🔧 技术架构
数据层：Parquet 文件存储的结构化知识图谱数据
嵌入层：OpenAI text-embedding-3-small 模型生成语义向量
存储层：LanceDB 向量数据库支持高效相似性搜索
推理层：GPT-4o-mini 模型基于图谱上下文生成回答
搜索层：LocalSearch 引擎整合多种信息源
🚀 工作流程
加载预处理的知识图谱数据（实体、关系、社区等）
为每个实体生成语义嵌入向量
构建向量数据库用于快速相似性搜索
接收用户查询并转换为向量表示
搜索相关实体、关系和社区信息
构建包含相关信息的上下文
使用大语言模型基于上下文生成准确回答
这个系统特别适合处理需要整合多个知识点、涉及复杂关系推理的问题，比如财务分析、企业关系分析等场景。

"""

import os
import pandas as pd
import tiktoken
import asyncio
from graphrag.query.indexer_adapters import (
    read_indexer_entities, read_indexer_relationships,
    read_indexer_reports, read_indexer_text_units
)
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from openai import OpenAI
import shutil

# ==================== 配置部分 ====================
# 配置 OpenAI API 环境变量
# 注意：在生产环境中应该使用更安全的方式管理 API 密钥
os.environ["OPENAI_API_KEY"] = 'sk-Sy5Z7b2m4sIDaZtFKZAIkQhAwFEn7adveD2ZaByDYdZMsn4V'
os.environ["OPENAI_API_BASE"] = 'https://api.agicto.cn/v1'

# ==================== 模型初始化 ====================
# 初始化大语言模型 (LLM)
# 用于生成最终的回答和推理
llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    api_base=os.environ["OPENAI_API_BASE"],
    api_type="openai_chat",
    model="gpt-4o-mini",                    # 使用 GPT-4o-mini 模型
    deployment_name="gpt-4o-mini",
    max_retries=20                          # 最大重试次数，提高稳定性
)

# 初始化文本嵌入模型
# 用于将文本转换为向量表示，支持语义相似性搜索
text_embedder = OpenAIEmbedding(
    api_key=os.environ["OPENAI_API_KEY"],
    api_base=os.environ["OPENAI_API_BASE"],
    api_type="openai_embedding",
    model="text-embedding-3-small",         # 使用 text-embedding-3-small 模型
    deployment_name="text-embedding-3-small",
    max_retries=20
)

# ==================== 数据路径配置 ====================
# 定义输入数据目录和各种数据表的名称
INPUT_DIR = "ragtest/ragtest/output"        # GraphRAG 处理后的输出目录
ENTITY_TABLE = "entities"                   # 实体表名
ENTITY_EMBEDDING_TABLE = "entities"        # 实体嵌入表名
RELATIONSHIP_TABLE = "relationships"       # 关系表名
COMMUNITY_REPORT_TABLE = "community_reports"  # 社区报告表名
TEXT_UNIT_TABLE = "text_units"             # 文本单元表名
LANCEDB_URI = f"{INPUT_DIR}/lancedb"       # LanceDB 向量数据库路径
COMMUNITY_LEVEL = 1                        # 社区层级，用于分层图谱分析

# ==================== 数据加载 ====================
# 从 Parquet 文件加载各种预处理的数据
# Parquet 是一种高效的列式存储格式，适合大数据处理
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")              # 实体数据
report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")    # 社区报告数据
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")  # 实体嵌入数据
relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")  # 关系数据
text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")       # 文本单元数据

# ==================== 数据预处理 ====================
# 将社区信息合并到实体数据中
# 这样每个实体都知道自己属于哪个社区和层级
if 'community' in report_df.columns:
    entity_df = entity_df.merge(report_df[['community']], left_index=True, right_index=True, how='left')
if 'level' in report_df.columns:
    entity_df = entity_df.merge(report_df[['level']], left_index=True, right_index=True, how='left')

# 读取并构建社区报告对象
# 社区报告包含了图谱中各个社区的摘要信息
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

# 处理实体嵌入数据
# 移除 degree 列以避免数据结构冲突（degree 列可能导致处理错误）
entity_embedding_df_no_degree = entity_embedding_df.drop(columns=['degree'], errors='ignore')
# 构建实体对象列表，每个实体包含其属性和嵌入向量
entities = read_indexer_entities(entity_df, entity_embedding_df_no_degree, COMMUNITY_LEVEL)

# ==================== 嵌入向量生成 ====================
# 初始化 OpenAI 客户端，用于直接调用嵌入 API
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"]
)

def get_embedding(text, model="text-embedding-3-small"):
    """
    使用 OpenAI API 获取文本的嵌入向量
    
    Args:
        text (str): 需要转换为向量的文本
        model (str): 使用的嵌入模型名称
    
    Returns:
        list: 文本的嵌入向量（1536维）
    
    功能说明：
    - 将文本转换为高维向量表示
    - 向量能够捕捉文本的语义信息
    - 用于后续的相似性搜索和匹配
    """
    if not text or text.strip() == "":
        return [0.0] * 1536  # 空文本返回零向量，1536是模型的默认维度

    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"获取嵌入向量时出错: {e}")
        return [0.0] * 1536  # 出错时返回零向量

# ==================== 实体嵌入向量生成 ====================
# 为每个实体生成语义嵌入向量
# 这是 GraphRAG 的核心步骤之一，将实体的文本描述转换为向量表示
for i, entity in enumerate(entities):
    # 检查实体是否已有嵌入向量
    if getattr(entity, 'description_embedding', None) is None:
        # 获取实体的描述和标题
        description = getattr(entity, 'description', '')
        title = getattr(entity, 'title', '无标题')
        
        # 组合标题和描述作为嵌入的输入文本
        # 格式："标题: 描述" 或仅 "标题"（如果没有描述）
        text_to_embed = f"{title}: {description}" if description else title
        
        print(f"正在为实体 '{title}' 生成嵌入向量 ({i + 1}/{len(entities)})...")
        
        # 生成嵌入向量并赋值给实体
        embedding = get_embedding(text_to_embed)
        entity.description_embedding = embedding
        
        print(f"✅ 完成 (维度: {len(embedding)})")

# ==================== 其他数据结构构建 ====================
# 构建关系对象列表，包含实体间的连接信息
relationships = read_indexer_relationships(relationship_df)

# 构建文本单元对象列表，包含原始文档的分块信息
text_units = read_indexer_text_units(text_unit_df)

# ==================== 向量数据库设置 ====================
# 初始化 LanceDB 向量存储
# LanceDB 是一个高性能的向量数据库，专门用于存储和搜索嵌入向量
description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
description_embedding_store.connect(db_uri=LANCEDB_URI)

# 清理旧的数据库文件，确保使用最新的嵌入向量
lance_db_path = f"{INPUT_DIR}/lancedb/entity_description_embeddings.lance"
if os.path.exists(lance_db_path):
    shutil.rmtree(lance_db_path)
    print("已清理旧的 LanceDB 数据库")

# 将实体的语义嵌入向量存储到向量数据库中
# 这样可以快速进行相似性搜索，找到与查询最相关的实体
store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)
print("✅ 成功存储实体语义嵌入!")

# ==================== 搜索上下文构建器 ====================
# 初始化 token 编码器，用于计算文本长度和管理上下文窗口
token_encoder = tiktoken.get_encoding("cl100k_base")

# 创建混合上下文构建器
# 这是 GraphRAG 的核心组件，负责为查询构建相关的上下文信息
context_builder = LocalSearchMixedContext(
    community_reports=reports,                          # 社区报告，提供高层次的摘要信息
    text_units=text_units,                             # 文本单元，提供原始文档片段
    entities=entities,                                 # 实体列表，提供结构化的知识点
    relationships=relationships,                       # 关系列表，提供实体间的连接信息
    covariates=None,                                  # 协变量（暂未使用）
    entity_text_embeddings=description_embedding_store, # 实体嵌入向量存储
    embedding_vectorstore_key="id",                   # 向量存储的键名
    text_embedder=text_embedder,                      # 文本嵌入器
    token_encoder=token_encoder                       # token 编码器
)

# ==================== 搜索引擎配置 ====================
# 本地搜索上下文参数配置
# 这些参数控制如何从知识图谱中选择和组合相关信息
local_context_params = {
    "text_unit_prop": 0.2,                    # 文本单元在上下文中的比例（20%）
    "community_prop": 0.8,                    # 社区报告在上下文中的比例（80%）
    "conversation_history_max_turns": 10,     # 对话历史的最大轮数
    "conversation_history_user_turns_only": True,  # 是否只保留用户的对话轮次
    "top_k_mapped_entities": 40,              # 选择的相关实体数量
    "top_k_relationships": 40,                # 选择的相关关系数量
    "include_entity_rank": True,              # 是否包含实体排名信息
    "include_relationship_weight": True,      # 是否包含关系权重信息
    "include_community_rank": False,          # 是否包含社区排名信息
    "return_candidate_context": False,        # 是否返回候选上下文
    "max_tokens": 12000                       # 上下文的最大 token 数量
}

# 大语言模型参数配置
llm_params = {
    "max_tokens": 2000,                       # 生成回答的最大 token 数量
    "temperature": 0.0                        # 温度参数，0.0 表示确定性输出
}

# 初始化本地搜索引擎
# 这是 GraphRAG 系统的主要接口，整合了所有组件
search_engine = LocalSearch(
    llm=llm,                                  # 大语言模型
    context_builder=context_builder,          # 上下文构建器
    token_encoder=token_encoder,              # token 编码器
    llm_params=llm_params,                    # LLM 参数
    context_builder_params=local_context_params,  # 上下文构建参数
    response_type="prioritized list"          # 响应类型：优先级列表
)

# ==================== 查询接口 ====================
async def run_graphrag_query(query):
    """
    运行 GraphRAG 查询，返回基于知识图谱的回答
    
    Args:
        query (str): 用户的查询问题
    
    Returns:
        str: GraphRAG 系统生成的回答
    
    工作流程：
    1. 接收用户查询
    2. 使用嵌入模型将查询转换为向量
    3. 在向量数据库中搜索相关实体
    4. 基于图谱结构找到相关的关系和社区
    5. 构建包含相关信息的上下文
    6. 使用 LLM 基于上下文生成回答
    """
    result = await search_engine.asearch(query)
    return result.response

# ==================== 演示功能 ====================
async def graphrag_query_demo():
    """
    GraphRAG 查询演示函数
    
    功能说明：
    - 展示如何使用 GraphRAG 系统回答复杂问题
    - 示例查询涉及多个实体和时间维度的分析
    - 演示系统如何整合图谱中的多种信息源
    """
    # 示例查询：关于 UnitedHealth Group 财务变化的复杂问题
    query = "What factors contributed to the changes in UnitedHealth Group's revenues and net earnings from 2023 to 2024, and how did these changes affect the earnings per share attributable to UnitedHealth Group common shareholders?"
    
    # 执行查询并获取回答
    response = await run_graphrag_query(query)
    print(f"GraphRAG 回答: {response}")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    """
    程序主入口
    
    执行流程：
    1. 加载和预处理知识图谱数据
    2. 生成实体嵌入向量
    3. 构建向量数据库
    4. 初始化搜索引擎
    5. 运行演示查询
    
    注意：整个过程是异步的，适合处理大规模数据和复杂查询
    """
    asyncio.run(graphrag_query_demo())