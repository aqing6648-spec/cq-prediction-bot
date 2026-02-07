import streamlit as st
import pandas as pd
from openai import OpenAI

# ================= 1. 基础配置 =================
st.set_page_config(page_title="高考志愿AI预测 (极速版)", page_icon="🚀", layout="wide")

# API Key 配置
try:
    MY_API_KEY = st.secrets["MY_API_KEY"]
except:
    MY_API_KEY = "sk-你的Key" # 本地测试用

BASE_URL = "https://api.siliconflow.cn/v1"
MY_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

# ================= 2. 加载数据 (只读 score.csv) =================
@st.cache_data
def load_data():
    try:
        # 读取分数表
        df = pd.read_csv("score.csv")
        return df
    except Exception as e:
        return None

df = load_data()

# ================= 3. 侧边栏: 考生信息录入 =================
with st.sidebar:
    st.header("🎯 考生档案")
    
    # 核心输入
    my_score = st.number_input("你的高考分数", min_value=0, max_value=750, value=520, step=1)
    
    st.divider()
    
    st.subheader("🔍 筛选条件")
    # 关键词筛选
    target_subject = st.text_input("想学的专业 (空则不限)", "计算机")
    target_city = st.text_input("想去的城市/省份 (空则不限)", "")
    
    # 风险偏好
    risk_option = st.radio(
        "推荐策略", 
        ["全部", "🟢 保底 (分差 > 15分)", "🔵 稳妥 (分差 5~15分)", "🟡 冲刺 (分差 -10~5分)"],
        index=0
    )

# ================= 4. 主界面逻辑 =================
st.title("🚀 高考志愿 AI 预测系统")
st.caption(f"当前参考数据：2024年重庆物理类录取分数线")

if df is not None:
    # --- 核心算法区 ---
    
    # 1. 初步筛选 (专业 & 城市)
    result = df.copy()
    if target_subject:
        result = result[result["专业"].str.contains(target_subject, na=False)]
    if target_city:
        result = result[result["城市"].str.contains(target_city, na=False) | result["省份"].str.contains(target_city, na=False)]
    
    # 2. 计算分差 (关键算法)
    # 逻辑：你的分 - 去年最低分
    # 正数越多越稳，负数代表要冲
    result["分差"] = my_score - result["最低分"]
    
    # 3. 打标签函数
    def get_tag(diff):
        if diff >= 15: return "🟢 保底"
        if diff >= 5: return "🔵 稳妥"
        if diff >= -10: return "🟡 冲刺"
        return "🔴 风险" # 分差 < -10

    result["录取概率"] = result["分差"].apply(get_tag)
    
    # 4. 根据侧边栏选择进行过滤
    if "保底" in risk_option:
        result = result[result["分差"] >= 15]
    elif "稳妥" in risk_option:
        result = result[(result["分差"] >= 5) & (result["分差"] < 15)]
    elif "冲刺" in risk_option:
        result = result[(result["分差"] >= -10) & (result["分差"] < 5)]
        
    # 5. 排序优化
    # 我们按“分差绝对值”排序，优先展示那些“分差最小”（最匹配）的学校
    # 这样用户第一眼看到的不是那种高出100分的烂学校，而是刚刚好的学校
    result["匹配度"] = result["分差"].abs()
    result = result.sort_values("匹配度")
    
    # --- 展示区 ---
    st.subheader(f"为你找到 {len(result)} 个方案")
    
    # 只展示核心列
    cols = ["学校", "专业", "最低分", "分差", "录取概率", "城市", "985", "211", "选科"]
    st.dataframe(
        result[cols].head(100), # 只展示前100个防止卡顿
        use_container_width=True,
        hide_index=True
    )
    
    # ================= 5. AI 分析师 (RAG) =================
    st.divider()
    
    # 只有当筛选出结果时才显示 AI 按钮
    if not result.empty:
        if st.button("🤖 呼叫 AI 帮我分析前 5 个志愿"):
            
            # 获取前5个数据
            top_5_data = result[cols].head(5).to_csv(index=False)
            
            prompt = f"""
            我是考生，分数 {my_score}。
            我筛选出的意向专业是：{target_subject if target_subject else "不限"}。
            
            系统算法推荐了以下 5 个最匹配的学校（基于2024年数据）：
            {top_5_data}
            
            请你作为资深高考志愿填报专家，帮我深度分析：
            1. 【性价比分析】：哪个学校虽然分不高，但是是985/211或者有特色？
            2. 【风险提示】：对于标记为"冲刺"的学校，我有多大概率滑档？
            3. 【最终建议】：如果是你，你会优先把哪个填在第一个位置？为什么？
            
            请用表格+加粗重点的方式回答。
            """
            
            client = OpenAI(api_key=MY_API_KEY, base_url=BASE_URL)
            
            with st.chat_message("assistant"):
                with st.spinner("AI 正在对比学校实力与录取概率..."):
                    stream = client.chat.completions.create(
                        model=MY_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True
                    )
                    st.write_stream(stream) # 开启流式输出，看起来更酷
    else:
        st.warning("⚠️ 没有找到符合条件的学校，请尝试降低分数或清空筛选条件。")

else:
    st.error("❌ 未找到 score.csv 文件，请检查文件是否在当前目录下！")