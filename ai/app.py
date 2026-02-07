import streamlit as st
from openai import OpenAI
import os
# ================= 配置区域 =================
# 还是原来的配方
MY_API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://api.siliconflow.cn/v1"
MY_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# ===========================================

# 1. 网页标题设置
st.set_page_config(page_title="重庆专升本AI助手", page_icon="🎓")
st.title("🎓 重庆专升本 AI 咨询助手")
st.caption("基于 DeepSeek R1 + 内部绝密知识库")

# 2. 初始化 AI 客户端
# 使用 @st.cache_resource 确保每次刷新网页不用重新连接，提高速度
@st.cache_resource
def get_client():
    return OpenAI(api_key=MY_API_KEY, base_url=BASE_URL)

client = get_client()

# 3. 读取知识库函数
def get_knowledge():
    try:
        with open("knowledge.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "暂无数据"

# 4. 核心：管理聊天记录 (Session State)
# Streamlit 每次你点按钮都会重跑一遍代码，所以要用 Session State 记住之前的聊天
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 开场白
    st.session_state.messages.append({"role": "assistant", "content": "你好！我是你的专属升学顾问。关于重庆专升本，你想知道什么？"})

# 5. 把聊天记录画在网页上
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 6. 处理用户输入
# 当用户在输入框回车时...
if user_input := st.chat_input("请输入你的问题（例如：重邮软件工程多少分？）"):
    
    # a. 显示用户的话
    with st.chat_message("user"):
        st.write(user_input)
    # b. 记入历史
    st.session_state.messages.append({"role": "user", "content": user_input})

    # c. 呼叫 AI (带 RAG)
    with st.chat_message("assistant"):
        with st.spinner("AI 正在查阅内部资料..."):
            try:
                # 构造 Prompt
                context = get_knowledge()
                system_prompt = f"""
                你是一个升学顾问。请严格基于以下资料回答：
                === 资料 ===
                {context}
                ============
                """
                
                # 发送请求（把所有历史记录发过去，让它有上下文）
                # 这里我们在历史记录前临时插一个 system prompt
                messages_to_send = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                
                response = client.chat.completions.create(
                    model=MY_MODEL,
                    messages=messages_to_send,
                    stream=False
                )
                
                ai_reply = response.choices[0].message.content
                st.write(ai_reply) # 把回答写在网页上
                
                # d. 记入历史
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                st.error(f"出错了: {e}")