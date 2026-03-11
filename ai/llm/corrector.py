import os
from typing import Generator
from zhipuai import ZhipuAI

class LLMCorrector:
    """
    Uses ZhipuAI GLM-4-Flash-250414 via the official SDK for contextual ASR error correction.
    API key is read from the ZHIPUAI_API_KEY environment variable.
    """
    def __init__(self, model_id: str = "glm-4-flash-250414"):
        self.model_id = model_id
        api_key = os.environ.get("ZHIPUAI_API_KEY", "")
        if not api_key:
            print("Warning: ZHIPUAI_API_KEY not set. LLM correction will be skipped.")
        self.client = ZhipuAI(api_key=api_key) if api_key else None
        print(f"LLM Corrector initialized: {model_id}")

    def _build_messages(self, raw_text: str, historical_context: list[str]) -> list:
        system_prompt = (
            "你是一个智能语音识别后处理纠错助手。"
            "你的任务是：仅修正【当前待处理文本】中的同音词错误和语句不通顺的地方，并加上适当的标点符号。"
            "严格要求："
            "1. 只输出【当前待处理文本】修正后的单句结果，长度应与原文大致相当；"
            "2. **禁止复述、重写或合并历史上下文；**"
            "3. 历史上下文仅供你理解语境，不得出现在输出中；"
            "4. 不要输出任何解释、前缀、序号或额外内容。"
        )
        user_content = ""
        if historical_context:
            context_lines = "\n".join(f"- {s}" for s in historical_context[-3:])
            user_content += f"【历史上下文（仅供参考，不得出现在输出中）】\n{context_lines}\n\n"
        user_content += f"【当前待处理文本】{raw_text}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _log_request(self, messages: list) -> None:
        print("\n" + "=" * 60)
        print("[LLM REQUEST]")
        for msg in messages:
            print(f"  [{msg['role'].upper()}] {msg['content']}")
        print("=" * 60)

    def _log_response(self, text: str) -> None:
        print("\n" + "-" * 60)
        print(f"[LLM RESPONSE] {text}")
        print("-" * 60 + "\n")

    def correct(self, raw_text: str, historical_context: list[str] = []) -> str:
        """非流式纠错，返回完整字符串。回退用。"""
        if not raw_text.strip():
            return ""
        if self.client is None:
            return raw_text.strip()

        messages = self._build_messages(raw_text, historical_context)
        self._log_request(messages)
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.3,
                max_tokens=256,
            )
            corrected = response.choices[0].message.content.strip()
            result = corrected if corrected else raw_text.strip()
            self._log_response(result)
            return result
        except Exception as e:
            print(f"LLM correction failed: {e}")
            return raw_text.strip()

    def correct_stream(self, raw_text: str, historical_context: list[str] = []) -> Generator[str, None, None]:
        """
        流式纠错，逐 token yield 字符串片段。
        如果客户端不可用则直接 yield 原始文本。
        """
        if not raw_text.strip():
            return
        if self.client is None:
            yield raw_text.strip()
            return

        messages = self._build_messages(raw_text, historical_context)
        self._log_request(messages)
        collected_tokens = []
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.3,
                max_tokens=256,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    collected_tokens.append(token)
                    yield token
        except Exception as e:
            print(f"LLM streaming correction failed: {e}")
        finally:
            self._log_response("".join(collected_tokens))
