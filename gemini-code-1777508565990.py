import os
import operator
from typing import Annotated, List, TypedDict

# 确保安装了：pip install langgraph langchain_openai
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ==========================================
# 1. 定义状态 (State)
# ==========================================
class MarketState(TypedDict):
    query: str                       # 用户输入的调研目标
    search_results: List[str]        # 抓取的原始信息
    consumer_insights: str           # 消费者洞察分析
    competitor_roadmap: str          # 竞品路线图推测
    final_swot_report: str           # 最终 SWOT 报告
    steps: Annotated[List[str], operator.add]  # 记录 Agent 执行轨迹

# ==========================================
# 2. 初始化大模型
# ==========================================
# 这里建议使用具备强推理能力的模型，如 GPT-4o
llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.2,
    # base_url="如果你使用代理/中转，请在此配置"
)

# ==========================================
# 3. 定义各个 Agent (Nodes) 的逻辑
# ==========================================

def scout_agent(state: MarketState):
    """搜索与调度智能体：模拟多源数据抓取"""
    print("--- [Agent 1: Scout] 正在检索市场情报与竞品动态... ---")
    query = state['query']
    
    # 模拟 CoT 拆解过程
    # 在生产环境中，此处会调用 Tavily, Google Search 或 Selenium 爬虫
    mock_data = [
        f"社交媒体：用户对{query}的评价中，30%提到续航痛点，50%关注外观设计。",
        f"行业报告：竞品A在上周发布的专利显示其正在布局固态电池技术。",
        f"电商评论：同类产品在亚马逊上的退货率主要集中在‘操作复杂’这一维度。"
    ]
    
    return {
        "search_results": mock_data,
        "steps": ["Scout Agent: 完成了多渠道异构数据的实时抓取"]
    }

def analyst_agent(state: MarketState):
    """分析智能体：深度语义挖掘与趋势识别"""
    print("--- [Agent 2: Analyst] 正在进行消费者情感分析与痛点提取... ---")
    raw_data = "\n".join(state['search_results'])
    
    prompt = f"""
    你是一名资深市场分析师。请分析以下原始情报，并提取出 3 个‘消费者未被满足的核心需求’：
    ---
    {raw_data}
    ---
    请以结构化方式输出洞察。
    """
    response = llm.invoke(prompt)
    
    return {
        "consumer_insights": response.content,
        "steps": ["Analyst Agent: 完成了对原始数据的语义对齐与痛点建模"]
    }

def strategist_agent(state: MarketState):
    """策略生成智能体：长链推理生成 SWOT 与建议"""
    print("--- [Agent 3: Strategist] 正在基于 CoT 生成最终战略报告... ---")
    
    context = f"""
    原始调研目标: {state['query']}
    消费者洞察: {state['consumer_insights']}
    """
    
    prompt = f"""
    你是一名顶级商业战略顾问。请基于以下分析结果，利用思维链（Chain-of-Thought）方法，
    生成一份详细的 SWOT 报告，并给出 3 条可落地的行动建议。
    ---
    {context}
    ---
    要求：逻辑严密，建议具有前瞻性。
    """
    response = llm.invoke(prompt)
    
    return {
        "final_swot_report": response.content,
        "steps": ["Strategist Agent: 完成了从数据到战略决策的长链推演"]
    }

# ==========================================
# 4. 构建 LangGraph 工作流图 (The Graph)
# ==========================================

def create_market_intelligence_app():
    # 初始化图逻辑
    workflow = StateGraph(MarketState)

    # 添加节点
    workflow.add_node("scout", scout_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("strategist", strategist_agent)

    # 设置执行路径 (Scout -> Analyst -> Strategist)
    workflow.set_entry_point("scout")
    workflow.add_edge("scout", "analyst")
    workflow.add_edge("analyst", "strategist")
    workflow.add_edge("strategist", END)

    # 编译
    return workflow.compile()

# ==========================================
# 5. 执行入口
# ==========================================
if __name__ == "__main__":
    # 请确保已设置环境变量，或在此处直接硬编码（不推荐）
    # os.environ["OPENAI_API_KEY"] = "sk-xxxx"

    app = create_market_intelligence_app()
    
    # 模拟输入
    initial_input = {
        "query": "高端智能家居领域的扫地机器人赛道"
    }
    
    print("\n>>> 启动多 Agent 协作工作流...\n")
    
    # 运行流式输出
    final_output = None
    for output in app.stream(initial_input):
        for key, value in output.items():
            if 'steps' in value:
                print(f"当前进度: {value['steps'][-1]}")
            final_output = value # 捕获最后的输出状态

    print("\n" + "="*50)
    print("🚀 自动化市场调研报告")
    print("="*50)
    # 由于是流式，我们从最后的 output 中获取最终报告
    # 在实际复杂场景下，可以用 app.invoke() 直接获取最终 state
    final_state = app.invoke(initial_input)
    print(final_state['final_swot_report'])
    print("\n" + "="*50)