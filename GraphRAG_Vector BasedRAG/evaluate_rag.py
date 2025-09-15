# -*- coding: utf-8 -*-
# @Time : 2025/9/9 11:30
# @Author : 殷韵智
# @FileName: evaluate_rag.py
# @Software: PyCharm
# RAG 评估脚本：比较 GraphRAG 和 Vector-based RAG，并生成适合处理情况分析

"""
RAG 系统评估与比较工具

这个脚本实现了对两种 RAG 系统的全面评估和比较，主要功能包括：
1. 自动化测试数据生成和查询执行
2. 使用 RAGAS 框架进行多维度评估
3. 生成详细的性能比较报告
4. 智能分析各系统的适用场景
5. 可视化评估结果对比

评估维度：
- 答案正确性 (Answer Correctness)：回答的准确性和完整性
- 语义相似性 (Semantic Similarity)：回答与参考答案的语义相似度
- 答案相关性 (Answer Relevancy)：回答与问题的相关程度
- 忠实度 (Faithfulness)：回答对检索内容的忠实程度

系统比较：
- GraphRAG：基于知识图谱的结构化检索，适合复杂关系推理
- Vector RAG：基于向量相似性的文档检索，适合直接信息查找

输出结果：
- CSV 格式的详细评估数据
- 可视化对比图表
- 智能场景适用性分析报告
"""

import pandas as pd
import asyncio
import os
import matplotlib.pyplot as plt
from ragas import evaluate, EvaluationDataset
from ragas.metrics import AnswerCorrectness, SemanticSimilarity, AnswerRelevancy, Faithfulness
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

# 导入 graph_rag 和 vector_rag 的函数
from graph_rag import run_graphrag_query
from vector_rag import run_vector_rag_query

# ==================== 可视化配置 ====================
# 配置 matplotlib 支持中文显示
plt.rcParams["font.family"] = ["SimHei"]        # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False      # 解决负号显示问题

# ==================== 环境配置 ====================
# 配置 OpenAI API 环境变量
os.environ["OPENAI_API_KEY"] = 'OPENAI_API_KEY'
os.environ["OPENAI_API_BASE"] = 'OPENAI_API_BASE'

# ==================== 模型初始化 ====================
# 初始化用于评估的大语言模型
# RAGAS 框架需要 LLM 来评估回答质量
llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"],
    model="gpt-4o-mini"
)

# 初始化嵌入模型
# 用于计算语义相似性等指标
embedding_model = OpenAIEmbeddings(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"],
    model="text-embedding-3-small"
)

# 初始化 OpenAI 客户端
# 用于生成智能分析报告
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"]
)

# ==================== 数据预处理 ====================
# 加载文本单元数据作为默认上下文
# 这些数据来自 GraphRAG 的预处理结果
parquet_file = "ragtest/ragtest/output/text_units.parquet"
if not os.path.exists(parquet_file):
    raise FileNotFoundError(f"文件 {parquet_file} 不存在于 {os.getcwd()}")

text_units_df = pd.read_parquet(parquet_file)
# 提取前3个文本单元作为默认上下文
# 如果文件结构不同，可能需要调整列名
default_context = " ".join(text_units_df['text'].head(3).tolist()) if 'text' in text_units_df.columns else ""

# ==================== 测试数据集 ====================
# 定义测试问题集合
# 每个问题都有对应的参考答案和检索上下文
# flag 字段指定使用哪种 RAG 系统进行测试
test_data = [
    {
        "user_input": "How did the change in claims submission patterns and accelerated claims payment timeframes affect the days claims payable in the third quarter of 2024, and how did this factor into the company's cash flows and return on equity?",
        "reference": "In Q3 2024, UnitedHealth Group's days claims payable increased to 47.4 from 45.2 in Q2, due to changes in claims submission patterns and accelerated payment timeframes. This increase supported care providers but reduced cash flows from operations to $7.6 billion. The return on equity was stable at 24.5%, reflecting operational efficiency despite higher claim payouts.",
        "retrieved_contexts": default_context or "Days claims payable in Q3 2024 was 47.4, up from 45.2 in Q2 2024, driven by accelerated claims payment timeframes. Cash flows from operations were $7.6 billion, impacted by higher payouts. Return on equity remained at 24.5%.",
        "flag": "vector_rag"  # 使用 Vector RAG 测试
    },
    {
        "user_input": "How did the change in claims submission patterns and accelerated claims payment timeframes affect the days claims payable in the third quarter of 2024, and how did this factor into the company's cash flows and return on equity?",
        "reference": "In Q3 2024, UnitedHealth Group's days claims payable increased to 47.4 from 45.2 in Q2, due to changes in claims submission patterns and accelerated payment timeframes. This increase supported care providers but reduced cash flows from operations to $7.6 billion. The return on equity was stable at 24.5%, reflecting operational efficiency despite higher claim payouts.",
        "retrieved_contexts": default_context or "UnitedHealth Group's days claims payable rose to 47.4 in Q3 2024 from 45.2 in Q2, due to faster claims processing. Cash flows from operations were $7.6 billion, affected by increased payouts. Return on equity held steady at 24.5%.",
        "flag": "graphrag"  # 使用 GraphRAG 测试
    },
    {
        "user_input": "How did the revenues from different business segments of UnitedHealth Group and Optum change from 2023 to 2024, and what were the impacts of cyberattack business disruption?",
        "reference": "From 2023 to 2024, UnitedHealth Group's revenues grew by 8% to $400.3 billion, with UnitedHealthcare increasing due to Medicare Advantage growth and Optum Health growing from value-based care. Optum Insight saw a 1% revenue decline due to a cyberattack, costing $867 million. The cyberattack caused $0.75 per share in business disruption impacts in 2024.",
        "retrieved_contexts": default_context or "UnitedHealth Group's 2024 revenues grew 8% to $400.3 billion, driven by UnitedHealthcare and Optum Health growth. Optum Insight revenues fell 1% due to a cyberattack, with $867 million in losses. Cyberattack business disruption impacts were $0.75 per share in 2024.",
        "flag": "vector_rag"
    },
    {
        "user_input": "How did the revenues from different business segments of UnitedHealth Group and Optum change from 2023 to 2024, and what were the impacts of cyberattack business disruption?",
        "reference": "From 2023 to 2024, UnitedHealth Group's revenues grew by 8% to $400.3 billion, with UnitedHealthcare increasing due to Medicare Advantage growth and Optum Health growing from value-based care. Optum Insight saw a 1% revenue decline due to a cyberattack, costing $867 million. The cyberattack caused $0.75 per share in business disruption impacts in 2024.",
        "retrieved_contexts": default_context or "UnitedHealth Group's revenues increased 8% to $400.3 billion in 2024, led by UnitedHealthcare and Optum Health. Optum Insight revenues dropped 1% due to cyberattack disruptions, costing $867 million. Business disruption impacts were $0.75 per share.",
        "flag": "graphrag"
    },
    {
        "user_input": "How did the South American impacts and cyberattack direct response costs affect the net earnings and net margin attributable to UnitedHealth Group common shareholders in the third quarter and first nine months of 2024?",
        "reference": "In Q3 2024, UnitedHealth Group's net earnings were $6.055 billion, with a net margin of 6.0%, impacted by $0.28 per share in cyberattack direct response costs and South American operation losses. For the first nine months, net earnings were $15.50-$15.75 per share, reflecting $7 billion in South American losses and $1.90-$2.05 per share in cyberattack costs.",
        "retrieved_contexts": default_context or "Q3 2024 net earnings were $6.055 billion, with a 6.0% net margin, affected by $0.28 per share cyberattack costs and South American losses. For the first nine months, net earnings were $15.50-$15.75 per share, including $7 billion South American losses and $1.90-$2.05 per share cyberattack impacts.",
        "flag": "vector_rag"
    },
    {
        "user_input": "How did the South American impacts and cyberattack direct response costs affect the net earnings and net margin attributable to UnitedHealth Group common shareholders in the third quarter and first nine months of 2024?",
        "reference": "In Q3 2024, UnitedHealth Group's net earnings were $6.055 billion, with a net margin of 6.0%, impacted by $0.28 per share in cyberattack direct response costs and South American operation losses. For the first nine months, net earnings were $15.50-$15.75 per share, reflecting $7 billion in South American losses and $1.90-$2.05 per share in cyberattack costs.",
        "retrieved_contexts": default_context or "Net earnings in Q3 2024 were $6.055 billion, with a 6.0% net margin, reduced by $0.28 per share cyberattack costs and South American impacts. For the first nine months, net earnings were $15.50-$15.75 per share, affected by $7 billion South American losses and $1.90-$2.05 per share cyberattack costs.",
        "flag": "graphrag"
    },
    {
        "user_input": "What factors contributed to the changes in UnitedHealth Group's revenues and net earnings from 2023 to 2024, and how did these changes affect the earnings per share attributable to UnitedHealth Group common shareholders?",
        "reference": "From 2023 to 2024, UnitedHealth Group's revenues increased by 8% to $400.3 billion due to higher premiums and service expansions. Net earnings dropped to $14.4 billion from $22 billion, impacted by a $7 billion South American loss and $1.90-$2.05 per share cyberattack costs. Earnings per share rose 8% to $15.50-$15.75 due to share repurchasing.",
        "retrieved_contexts": default_context or "UnitedHealth Group's 2024 revenues grew 8% to $400.3 billion, driven by premiums and service growth. Net earnings fell to $14.4 billion due to South American losses and cyberattack costs. Earnings per share increased 8% to $15.50-$15.75 with share buybacks.",
        "flag": "vector_rag"
    },
    {
        "user_input": "What factors contributed to the changes in UnitedHealth Group's revenues and net earnings from 2023 to 2024, and how did these changes affect the earnings per share attributable to UnitedHealth Group common shareholders?",
        "reference": "From 2023 to 2024, UnitedHealth Group's revenues increased by 8% to $400.3 billion due to higher premiums and service expansions. Net earnings dropped to $14.4 billion from $22 billion, impacted by a $7 billion South American loss and $1.90-$2.05 per share cyberattack costs. Earnings per share rose 8% to $15.50-$15.75 due to share repurchasing.",
        "retrieved_contexts": default_context or "UnitedHealth Group's revenues grew 8% to $400.3 billion in 2024, led by premium increases and service expansions. Net earnings decreased to $14.4 billion due to $7 billion South American losses and cyberattack impacts. Earnings per share rose 8% to $15.50-$15.75.",
        "flag": "graphrag"
    }
]

# ==================== 智能分析功能 ====================
def analyze_rag_suitability(rag_df, graph_df):
    """
    使用大模型分析 Vector-based RAG 和 GraphRAG 的适合场景
    
    Args:
        rag_df (pd.DataFrame): Vector RAG 的评估结果
        graph_df (pd.DataFrame): GraphRAG 的评估结果
    
    Returns:
        str: 智能生成的场景适用性分析报告
    
    功能说明：
    - 基于 RAGAS 评估指标计算平均分数
    - 使用 GPT 模型分析各系统的优势和适用场景
    - 生成结构化的分析报告
    - 为用户选择合适的 RAG 系统提供指导
    """
    metrics = ['answer_correctness', 'semantic_similarity', 'answer_relevancy', 'faithfulness']
    
    # 计算各系统在不同指标上的平均表现
    rag_means = rag_df[metrics].mean() if not rag_df.empty else pd.Series(0, index=metrics)
    graph_means = graph_df[metrics].mean() if not graph_df.empty else pd.Series(0, index=metrics)

    # 构造分析提示
    # 提供详细的评估数据供 GPT 分析
    prompt = f"""
    基于以下 RAGAS 评估结果，分析 Vector-based RAG 和 GraphRAG 分别适合处理哪些情况：

    Vector-based RAG 平均分数：
    - 答案正确性: {rag_means.get('answer_correctness', 0):.2f}
    - 语义相似性: {rag_means.get('semantic_similarity', 0):.2f}
    - 答案相关性: {rag_means.get('answer_relevancy', 0):.2f}
    - 忠实度: {rag_means.get('faithfulness', 0):.2f}

    GraphRAG 平均分数：
    - 答案正确性: {graph_means.get('answer_correctness', 0):.2f}
    - 语义相似性: {graph_means.get('semantic_similarity', 0):.2f}
    - 答案相关性: {graph_means.get('answer_relevancy', 0):.2f}
    - 忠实度: {graph_means.get('faithfulness', 0):.2f}

    请以中文提供分析，列出每种 RAG 方法适合的场景（至少 3 种），并简要说明原因。返回结果格式如下：
    ### Vector-based RAG 适合场景
    - 场景 1: 说明
    - 场景 2: 说明
    - 场景 3: 说明

    ### GraphRAG 适合场景
    - 场景 1: 说明
    - 场景 2: 说明
    - 场景 3: 说明
    """

    try:
        # 调用 GPT 模型生成分析报告
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个专业的 RAG 系统分析师，能够根据评估结果分析适合场景。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3  # 较低的温度确保分析结果的一致性
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"生成适合场景分析失败: {e}")
        return "无法生成分析结果"

# ==================== 查询执行功能 ====================
async def get_responses(test_data):
    """
    执行测试查询并收集两种 RAG 系统的响应
    
    Args:
        test_data (list): 测试数据集合
    
    Returns:
        tuple: (vector_rag_data, graph_rag_data) 两个系统的响应数据
    
    功能说明：
    - 根据 flag 字段分别调用不同的 RAG 系统
    - 处理查询异常并记录错误信息
    - 收集完整的查询-响应对用于后续评估
    - 支持异步执行提高效率
    """
    vector_rag_data = []
    graph_rag_data = []

    for item in test_data:
        query = item["user_input"]
        reference = item["reference"]
        contexts = [item["retrieved_contexts"]]

        # 执行 Vector-based RAG 查询
        if item["flag"] == "vector_rag":
            try:
                print(f"🔍 执行 Vector RAG 查询: {query[:50]}...")
                response = run_vector_rag_query(query)
                vector_rag_data.append({
                    "user_input": query,
                    "reference": reference,
                    "retrieved_contexts": contexts,
                    "response": response
                })
                print("✅ Vector RAG 查询完成")
            except Exception as e:
                print(f"❌ Vector RAG 查询失败: {e}")
                vector_rag_data.append({
                    "user_input": query,
                    "reference": reference,
                    "retrieved_contexts": contexts,
                    "response": "查询失败"
                })

        # 执行 GraphRAG 查询
        if item["flag"] == "graphrag":
            try:
                print(f"🔍 执行 GraphRAG 查询: {query[:50]}...")
                response = await run_graphrag_query(query)
                graph_rag_data.append({
                    "user_input": query,
                    "reference": reference,
                    "retrieved_contexts": contexts,
                    "response": response
                })
                print("✅ GraphRAG 查询完成")
            except Exception as e:
                print(f"❌ GraphRAG 查询失败: {e}")
                graph_rag_data.append({
                    "user_input": query,
                    "reference": reference,
                    "retrieved_contexts": contexts,
                    "response": "查询失败"
                })

    return vector_rag_data, graph_rag_data

# ==================== 主评估函数 ====================
async def main():
    """
    主评估流程
    
    执行步骤：
    1. 检查必需的输入文件
    2. 执行测试查询收集响应
    3. 使用 RAGAS 框架进行多维度评估
    4. 生成评估报告和可视化图表
    5. 进行智能场景适用性分析
    6. 保存所有结果到文件
    
    输出文件：
    - vector_rag_results.csv: Vector RAG 详细评估结果
    - graph_rag_results.csv: GraphRAG 详细评估结果
    - rag_suitability_analysis.txt: 智能场景分析报告
    - rag_comparison.png: 可视化对比图表
    """
    print("🚀 RAG 系统评估开始")
    
    # ==================== 文件检查 ====================
    # 检查所有必需的输入文件是否存在
    required_files = [
        "ragtest/ragtest/output/entities.parquet",          # GraphRAG 实体数据
        "ragtest/ragtest/output/relationships.parquet",     # GraphRAG 关系数据
        "ragtest/ragtest/output/community_reports.parquet", # GraphRAG 社区报告
        "ragtest/ragtest/output/text_units.parquet",        # GraphRAG 文本单元
        "ragtest/ragtest/input/unh_data.txt"                # Vector RAG 原始文档
    ]
    
    print("📋 检查必需文件...")
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"文件 {file} 不存在于 {os.getcwd()}")
    print("✅ 所有必需文件检查完成")

    # ==================== 查询执行 ====================
    print("\n🔄 开始执行测试查询...")
    vector_rag_data, graph_rag_data = await get_responses(test_data)
    print(f"📊 收集到 {len(vector_rag_data)} 个 Vector RAG 响应")
    print(f"📊 收集到 {len(graph_rag_data)} 个 GraphRAG 响应")

    # ==================== RAGAS 评估 ====================
    print("\n📈 开始 RAGAS 评估...")
    
    # 评估 Vector-based RAG
    try:
        print("🔍 评估 Vector-based RAG...")
        rag_eval_dataset = EvaluationDataset.from_dict(vector_rag_data)
        rag_results = evaluate(
            dataset=rag_eval_dataset,
            metrics=[
                AnswerCorrectness(),    # 答案正确性
                SemanticSimilarity(),   # 语义相似性
                AnswerRelevancy(),      # 答案相关性
                Faithfulness()          # 忠实度
            ],
            llm=llm,
            embeddings=embedding_model
        )
        rag_df = rag_results.to_pandas()
        print("✅ Vector RAG 评估完成")
    except Exception as e:
        print(f"❌ Vector RAG 评估失败: {e}")
        rag_df = pd.DataFrame()

    # 评估 GraphRAG
    try:
        print("🔍 评估 GraphRAG...")
        graph_eval_dataset = EvaluationDataset.from_dict(graph_rag_data)
        graph_results = evaluate(
            dataset=graph_eval_dataset,
            metrics=[
                AnswerCorrectness(),    # 答案正确性
                SemanticSimilarity(),   # 语义相似性
                AnswerRelevancy(),      # 答案相关性
                Faithfulness()          # 忠实度
            ],
            llm=llm,
            embeddings=embedding_model
        )
        graph_df = graph_results.to_pandas()
        print("✅ GraphRAG 评估完成")
    except Exception as e:
        print(f"❌ GraphRAG 评估失败: {e}")
        graph_df = pd.DataFrame()

    # ==================== 结果保存 ====================
    print("\n💾 保存评估结果...")
    if not rag_df.empty:
        rag_df.to_csv("vector_rag_results.csv", index=False)
        print("✅ Vector RAG 结果已保存到 vector_rag_results.csv")
    if not graph_df.empty:
        graph_df.to_csv("graph_rag_results.csv", index=False)
        print("✅ GraphRAG 结果已保存到 graph_rag_results.csv")

    # ==================== 结果输出 ====================
    print("\n📊 评估结果摘要:")
    print("=" * 60)
    print("Vector-based RAG 评估结果:")
    if not rag_df.empty:
        print(rag_df.describe())
    else:
        print("无评估数据")
    
    print("\n" + "=" * 60)
    print("GraphRAG 评估结果:")
    if not graph_df.empty:
        print(graph_df.describe())
    else:
        print("无评估数据")

    # ==================== 智能分析 ====================
    print("\n🧠 生成智能场景分析...")
    suitability_analysis = analyze_rag_suitability(rag_df, graph_df)
    print("\n📋 适合场景分析:")
    print("=" * 60)
    print(suitability_analysis)

    # 保存分析结果
    with open("rag_suitability_analysis.txt", "w", encoding="utf-8") as f:
        f.write(suitability_analysis)
    print("✅ 场景分析已保存到 rag_suitability_analysis.txt")

    # ==================== 可视化对比 ====================
    if not rag_df.empty and not graph_df.empty:
        print("\n📊 生成可视化对比图...")
        
        # 计算各指标的平均分数
        metrics = ['answer_correctness', 'semantic_similarity', 'answer_relevancy', 'faithfulness']
        rag_means = rag_df[metrics].mean()
        graph_means = graph_df[metrics].mean()

        # 创建对比柱状图
        plt.figure(figsize=(12, 8))
        bar_width = 0.35
        index = range(len(metrics))
        
        # 绘制柱状图
        plt.bar(index, rag_means, bar_width, 
                label='Vector-based RAG', color='#1f77b4', alpha=0.8)
        plt.bar([i + bar_width for i in index], graph_means, bar_width, 
                label='GraphRAG', color='#ff7f0e', alpha=0.8)
        
        # 添加数值标签
        for i, v in enumerate(rag_means):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        for i, v in enumerate(graph_means):
            plt.text(i + bar_width, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 图表配置
        plt.xlabel('评估指标', fontsize=12)
        plt.ylabel('分数', fontsize=12)
        plt.title('Vector-based RAG 与 GraphRAG 评估结果比较', fontsize=14, fontweight='bold')
        plt.xticks([i + bar_width / 2 for i in index], 
                  ['答案正确性', '语义相似性', '答案相关性', '忠实度'])
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('rag_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ 对比图表已保存到 rag_comparison.png")
        plt.show()
    else:
        print("⚠️ 缺少评估数据，无法生成可视化对比")

    print("\n🎉 RAG 系统评估完成！")
    print("📁 输出文件:")
    print("  - vector_rag_results.csv: Vector RAG 详细评估结果")
    print("  - graph_rag_results.csv: GraphRAG 详细评估结果")
    print("  - rag_suitability_analysis.txt: 智能场景分析报告")
    print("  - rag_comparison.png: 可视化对比图表")

# ==================== 程序入口 ====================
if __name__ == "__main__":
    """
    程序主入口
    
    功能说明：
    - 解决 Jupyter 环境中的异步上下文问题
    - 启动完整的 RAG 系统评估流程
    - 处理异步操作和错误管理
    
    使用场景：
    - RAG 系统性能对比分析
    - 系统选型决策支持
    - 模型优化效果评估
    - 学术研究和技术报告
    """
    import nest_asyncio

    # 解决在 Jupyter 等环境中的异步上下文问题
    nest_asyncio.apply()
    
    # 启动主评估流程
    asyncio.run(main())