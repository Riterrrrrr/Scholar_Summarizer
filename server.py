# backend/server.py

# 头部引用需要确保包含这些
import os
import uvicorn
import trafilatura
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai # Google 官方最新 SDK
from google.genai import types
import time

# ================= 配置区域 =================

# 1. API 配置
# 请去 https://aistudio.google.com/ 获取 API Key
GOOGLE_API_KEY = "AIzaSyCLNsWRfOjc2cZnjTSi5tv3fbRKHjB6TsU"  # 付费 Key，请自行更换
# GOOGLE_API_KEY = "AIzaSyDFiq3gsNqG_8QK8NqALsKrsYRs8woCUq0"  # 免费 Key，请自行更换

# 2. 模型选择
# 2025年此时，建议查看 Google AI Studio 模型列表。
# 可能是 "gemini-3.0-pro", "gemini-2.0-flash", 或 "gemini-1.5-pro"
MODEL_NAME = "gemini-2.5-flash" 

# ================= 初始化 =================

app = FastAPI(title="Scholar Summarizer Backend")

# 配置 CORS (跨域资源共享)
# 允许来自浏览器插件的请求访问此服务器
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中建议限制为插件的 ID，但在本地开发 "*" 没问题
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 Gemini 客户端
client = genai.Client(api_key=GOOGLE_API_KEY)

# 定义前端传来的数据格式
class URLRequest(BaseModel):
    url: str

# ================= 核心工具函数 =================

def fetch_clean_content(url: str):
    """
    抓取并清洗网页内容
    """
    print(f"[*] 收到抓取请求: {url}")
    try:
        # trafilatura 自动处理 User-Agent 和 Cookie 穿透
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None, "无法连接到该网页 (Network Error)"
        
        # 提取正文，去除导航栏、表格和评论
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        
        if not text:
            return None, "网页内容提取为空 (可能是纯图片或 SPA 动态加载)"
            
        # 截取前 4000 字符 (足够覆盖简介，且节省 Token)
        return text[:4000], None
        
    except Exception as e:
        return None, f"抓取异常: {str(e)}"

def generate_summary_gemini(text_content: str):
    """
    V5.0: 通用学术总结框架 (The Universal Academic Profiler)
    旨在兼容各种写作风格（显性声明、隐性描述、愿景驱动型）
    """
    
    # System Instruction: 定义任务的本质是“信息蒸馏”而非简单的“总结”
    system_instruction = """
    You are an expert Academic Research Analyst.
    Your objective is to distill a scholar's profile into a precise statement of their *active* research contribution.
    You must distinguish between "what they research" (High Value) and "who they are/what they teach" (Low Value).
    """

    # User Prompt: 通用处理协议
    prompt_content = f"""
    Please analyze the text using the following **Universal Research Extraction Protocol**:

    ### Phase 1: Signal Detection (The Funnel)
    Scan the text and extract research topics based on the following hierarchy. **Prioritize the highest level found.**

    * **Level 1: Explicit Temporality (The "Now" Signal)**
        * Look for: "Current research", "Recent projects", "Working on", "In progress", "Latest work".
        * *Action:* If found, this is the primary source.
    
    * **Level 2: Active Investigation (The "Doing" Signal)**
        * Look for strong verbs indicating active inquiry: "Investigates", "Examines", "Conducts research on", "Explores", "Analyzes".
        * *Action:* Use this if Level 1 is missing. This captures the scholar's ongoing work.
    
    * **Level 3: Intent & Impact (The "Goal" Signal)**
        * Look for teleological markers: "Aiming to", "Seeks to", "Goal is to", "Dedicated to improving", "Address the problem of".
        * *Action:* Use this to identify the *purpose* or *application* of their work (often the most descriptive part).

    ### Phase 2: Noise Cancellation (Universal Filters)
    Strictly IGNORE the following categories unless they are the *object* of research:
    * **Pedagogy:** "Taught courses in...", "Engages students...", "Teaching interests".
    * **Biography:** "Earned PhD from...", "Joined faculty in...", "Director of...".
    * **General Affiliation:** Do not assume research topics based solely on the Department name (e.g., "Social Work", "Computer Science") or broad disciplinary labels (e.g., "Law", "Sociology") without specific context.

    ### Phase 3: Synthesis & Compression
    * **Format:** Create a single, high-density English phrase.
    * **Structure:** Start with a dynamic component (Verb/Noun) + Specific Topic + Context/Population.
    * **Faithfulness:** Retain specific terminology (e.g., "incarcerated women", "machine learning fairness") rather than generalizing (e.g., "vulnerable groups", "AI ethics").
    
    ### Constraints
    * **Language:** English ONLY.
    * **Length:** Strictly **10 to 15 words**.

    ### Input Text:
    {text_content}

    ### Output:
    Provide ONLY the final English summary from Phase 3.
    """

    # === 核心修改：重试循环配置 ===
    current_max_tokens = 2048  # 初始 Token 额度
    max_retries = 3            # 最多尝试 3 次
    
    # 安全设置 (保持全开)
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]

    for attempt in range(max_retries):
        try:
            print(f"[*] 第 {attempt + 1} 次尝试，Max Tokens: {current_max_tokens}")

            response = client.models.generate_content(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    max_output_tokens=current_max_tokens, # <--- 动态变量
                    safety_settings=safety_settings,
                ),
                contents=[prompt_content]
            )

            # 1. 检查是否因为 Token 不够而截断
            # 注意：不同版本的 SDK 对 finish_reason 的访问方式可能不同，这里做个通用判断
            finish_reason = str(response.candidates[0].finish_reason)
            
            if "MAX_TOKENS" in finish_reason:
                print(f"[!] Token 不足 (MAX_TOKENS)，正在扩容重试...")
                current_max_tokens *= 2  # 翻倍：1024 -> 2048 -> 4096
                time.sleep(1) # 稍微歇一秒，防止接口频繁
                continue # 进入下一次循环
            
            # 2. 检查是否有有效文本
            if response.text:
                return response.text.strip()
            
            # 3. 如果没文本也没报错 MAX_TOKENS (可能是安全过滤等其他原因)
            print(f"[!] 响应为空，未知原因: {finish_reason}")
            return "Error: No text generated."

        except Exception as e:
            print(f"[!] 调用异常: {e}")
            # 如果是 API 网络错误，也可以在这里选择 continue 重试
            return f"API Error: {str(e)}"

    return "Error: Exceeded max retries (Model is too chatty!)"

# ================= API 接口 =================

@app.post("/summarize")
async def api_summarize(request: URLRequest):
    """
    供浏览器插件调用的接口
    """
    # 1. 抓取
    text, error = fetch_clean_content(request.url)
    
    if error:
        # 返回 400 错误给前端
        raise HTTPException(status_code=400, detail=error)
    
    # 2. 总结
    summary = generate_summary_gemini(text)
    
    # 3. 返回 JSON
    return {"summary": summary}

if __name__ == "__main__":
    # 启动服务器，监听 8000 端口
    print("🚀 后端服务启动中... 请保持此窗口打开")
    uvicorn.run(app, host="127.0.0.1", port=8000)