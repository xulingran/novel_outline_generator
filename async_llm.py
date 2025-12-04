import asyncio
import random
import os
from config import API_PROVIDER, API_KEY, API_BASE, MODEL_NAME, MAX_RETRY, USE_PROXY, PROXY_URL, GEMINI_SAFETY_SETTINGS

# 配置代理
if USE_PROXY and PROXY_URL:
    os.environ['HTTP_PROXY'] = PROXY_URL
    os.environ['HTTPS_PROXY'] = PROXY_URL
    print(f"🌐 已配置代理: {PROXY_URL}")

# 全局变量：Gemini 安全设置
gemini_safety_settings = None

# 根据 API_PROVIDER 初始化客户端
if API_PROVIDER == "gemini":
    try:
        import google.generativeai as genai
        # Gemini API 通过环境变量使用代理
        genai.configure(api_key=API_KEY)
        
        # 配置安全设置
        safety_mapping = {
            "BLOCK_NONE": genai.types.HarmBlockThreshold.BLOCK_NONE,
            "BLOCK_ONLY_HIGH": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            "BLOCK_MEDIUM_AND_ABOVE": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "BLOCK_LOW_AND_ABOVE": genai.types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        safety_threshold = safety_mapping.get(GEMINI_SAFETY_SETTINGS, genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH)
        
        # 创建安全设置配置
        gemini_safety_settings = [
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": safety_threshold,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": safety_threshold,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": safety_threshold,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": safety_threshold,
            },
        ]
        
        print(f"✅ Gemini API 初始化成功 (模型: {MODEL_NAME})")
        print(f"   ℹ️  安全设置: {GEMINI_SAFETY_SETTINGS}")
        if USE_PROXY and PROXY_URL:
            print(f"   ℹ️  Gemini API 将通过环境变量使用代理")
    except ImportError:
        print("❌ 错误: 未安装 google-generativeai 库")
        print("💡 请运行: pip install google-generativeai")
        raise
    except Exception as e:
        print(f"❌ Gemini API 配置失败: {e}")
        raise
else:
    try:
        from openai import AsyncOpenAI
        # OpenAI 客户端支持通过 http_client 配置代理
        import httpx
        http_client = None
        if USE_PROXY and PROXY_URL:
            http_client = httpx.AsyncClient(proxies=PROXY_URL)
            print(f"🌐 OpenAI 客户端已配置代理: {PROXY_URL}")
        
        if API_BASE:
            openai_client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, http_client=http_client)
        else:
            openai_client = AsyncOpenAI(api_key=API_KEY, http_client=http_client)
        print(f"✅ OpenAI API 初始化成功 (模型: {MODEL_NAME})")
    except ImportError as e:
        print("❌ 错误: 未安装 openai 库")
        print("💡 请运行: pip install openai")
        raise


async def call_llm_openai(prompt, chunk_id=None):
    """调用 OpenAI 兼容 API"""
    chunk_info = f" [块 {chunk_id}]" if chunk_id else ""
    for attempt in range(MAX_RETRY):
        try:
            if attempt == 0:
                print(f"  🔄 调用 OpenAI API{chunk_info}...")
            res = await openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            result = res.choices[0].message.content
            print(f"  ✅ OpenAI API 调用成功{chunk_info}")
            return result
        except Exception as e:
            wait = (2 ** attempt) + random.random()
            print(f"  ⚠️ 调用失败{chunk_info}，{wait:.1f}s 后重试 (尝试 {attempt + 1}/{MAX_RETRY}): {e}")
            await asyncio.sleep(wait)
    print(f"  ❌ OpenAI API 多次失败{chunk_info}")
    return "ERROR: LLM 多次失败"


async def call_llm_gemini(prompt, chunk_id=None):
    """调用 Gemini API"""
    import google.generativeai as genai
    
    chunk_info = f" [块 {chunk_id}]" if chunk_id else ""
    
    # 使用全局安全设置，如果未设置则使用默认值
    safety_settings = gemini_safety_settings
    
    for attempt in range(MAX_RETRY):
        try:
            if attempt == 0:
                print(f"  🔄 调用 Gemini API{chunk_info}...")
            
            # 获取模型（每次调用时重新获取，避免状态问题）
            model = genai.GenerativeModel(MODEL_NAME)
            
            # Gemini API 是同步的，需要在异步环境中运行
            if safety_settings:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.generate_content(
                        prompt,
                        safety_settings=safety_settings
                    )
                )
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.generate_content(prompt)
                )
            
            # 检查是否有内容被阻止
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                feedback = response.prompt_feedback
                if hasattr(feedback, 'block_reason') and feedback.block_reason:
                    block_reason = feedback.block_reason.name if hasattr(feedback.block_reason, 'name') else str(feedback.block_reason)
                    error_msg = f"内容被阻止 (原因: {block_reason})"
                    if hasattr(feedback, 'safety_ratings') and feedback.safety_ratings:
                        ratings_info = []
                        for rating in feedback.safety_ratings:
                            category = rating.category.name if hasattr(rating.category, 'name') else str(rating.category)
                            probability = rating.probability.name if hasattr(rating.probability, 'name') else str(rating.probability)
                            ratings_info.append(f"{category}: {probability}")
                        error_msg += f" [详情: {', '.join(ratings_info)}]"
                    
                    print(f"  ⚠️ {error_msg}{chunk_info}")
                    # 如果被阻止，不重试，直接返回错误信息
                    return f"ERROR: 内容被安全过滤器阻止 - {error_msg}"
            
            # 提取文本内容
            if hasattr(response, 'text') and response.text:
                print(f"  ✅ Gemini API 调用成功{chunk_info}")
                return response.text
            else:
                # 检查是否有候选响应
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                        if text_parts:
                            result = ''.join(text_parts)
                            print(f"  ✅ Gemini API 调用成功{chunk_info}")
                            return result
                
                # 如果没有候选响应，检查是否被阻止
                if hasattr(response, 'prompt_feedback'):
                    error_msg = "Gemini API 返回空内容（可能被安全过滤器阻止）"
                else:
                    error_msg = "Gemini API 返回空内容"
                raise ValueError(error_msg)
                
        except ValueError as e:
            # 对于内容被阻止的情况，不重试
            if "被安全过滤器阻止" in str(e) or "被阻止" in str(e):
                print(f"  ❌ {str(e)}{chunk_info}")
                return f"ERROR: {str(e)}"
            raise
        except Exception as e:
            wait = (2 ** attempt) + random.random()
            error_detail = str(e)
            # 检查是否是内容被阻止的错误
            if "PROHIBITED_CONTENT" in error_detail or "block_reason" in error_detail:
                print(f"  ⚠️ 内容被安全过滤器阻止{chunk_info}: {error_detail}")
                print(f"  💡 提示: 可以尝试在 config.py 中调整 GEMINI_SAFETY_SETTINGS 设置")
                # 对于内容阻止，不重试
                return f"ERROR: 内容被安全过滤器阻止 - {error_detail}"
            
            print(f"  ⚠️ 调用失败{chunk_info}，{wait:.1f}s 后重试 (尝试 {attempt + 1}/{MAX_RETRY}): {error_detail}")
            await asyncio.sleep(wait)
    
    print(f"  ❌ Gemini API 多次失败{chunk_info}")
    return "ERROR: LLM 多次失败"


async def call_llm(prompt, chunk_id=None):
    """统一的 LLM 调用接口，根据配置自动选择 API"""
    if API_PROVIDER == "gemini":
        return await call_llm_gemini(prompt, chunk_id)
    else:
        return await call_llm_openai(prompt, chunk_id)
