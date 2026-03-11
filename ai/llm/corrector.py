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

    # 句末标点集合，用于代码层面判断上一句是否已结束
    _SENTENCE_END = frozenset("。！？…")

    def _build_messages(self, raw_text: str, historical_context: list[str]) -> list:
        system_prompt = (
            "你是一个智能语音识别后处理纠错助手，处理实时流式 ASR 逐片段输出的文本。\n"
            "每次调用你只会收到一小段原始识别文本（约几秒钟的语音），你需要完成以下任务：\n\n"
            "【任务一：文本修正】\n"
            "修正当前文本中的同音词错误和不通顺之处。\n\n"
            "【任务二：句内标点】\n"
            "在当前文本内部语义停顿处插入适当的标点符号（逗号、顿号、问号等）。\n\n"
            "【任务三：衔接标点（严格按系统提示执行）】\n"
            "user 消息中会明确告诉你【上一句是否已结束】：\n"
            "- 若标注为【上一句已结束】：当前文本是全新句子的开头，输出绝对不能以任何标点开头；\n"
            "- 若标注为【上一句未结束】：当前文本是上一句的接续，若语义确实接续则在输出最开头加上逗号\uff0c。\n\n"
            "【任务四：句子完整性判断】\n"
            '- 当前文本语义完整 \u2192 输出必须以\u201c\u3002\u201d结尾；\n'
            '- 当前文本明显是半句话或被截断 \u2192 输出绝对不能以\u201c\u3002\u201d结尾。\n\n'
            "【严格约束】\n"
            "1. 只输出修正后的当前文本，长度应与原文大致相当；\n"
            "2. 禁止复述、重写或合并历史上下文；\n"
            "3. 不要输出任何解释、前缀、序号或多余内容。"
        )

        # 代码层面判断上一句是否已以句末标点结束，直接告知 LLM，消除歧义
        prev_sentence_ended = True  # 无历史时视为新句子
        if historical_context:
            last = historical_context[-1].rstrip()
            prev_sentence_ended = bool(last) and last[-1] in self._SENTENCE_END

        user_content = ""
        if historical_context:
            recent = historical_context[-3:]
            context_lines = "\n".join(
                f"- {'[最后一条] ' if i == len(recent) - 1 else ''}{s}"
                for i, s in enumerate(recent)
            )
            user_content += f"【历史上下文（仅供参考，不得出现在输出中）】\n{context_lines}\n\n"

        ended_hint = "【上一句已结束】" if prev_sentence_ended else "【上一句未结束】"
        user_content += f"{ended_hint}\n【当前待处理文本】{raw_text}"

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
