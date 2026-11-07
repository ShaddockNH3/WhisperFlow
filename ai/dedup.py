"""幻觉拦截：Whisper 会把同一句反复吐出来，LLM 纠错也会复读上下文。

原来这两个阈值（0.85 / 0.7）和"最近 10 条"藏在 pipeline 的两个方法里，
这里抽成独立模块，只用标准库 difflib，能单独测。
"""

import difflib

DEFAULT_REPEAT_THRESHOLD = 0.85
DEFAULT_HISTORY_THRESHOLD = 0.7
DEFAULT_HISTORY_SIZE = 10


def similarity(left: str, right: str) -> float:
    """两段文本的相似度，0~1。空串对比一律 0。"""
    left_text = (left or "").strip()
    right_text = (right or "").strip()
    if not left_text or not right_text:
        return 0.0
    return difflib.SequenceMatcher(None, left_text, right_text).ratio()


class RepeatFilter:
    """相邻两句的重复拦截：和上一句太像就认为模型在复读。"""

    def __init__(self, threshold: float = DEFAULT_REPEAT_THRESHOLD) -> None:
        self.threshold = threshold
        self._last_accepted = ""

    @property
    def last_accepted(self) -> str:
        return self._last_accepted

    def accept(self, text: str) -> bool:
        """返回 True 表示这句要，False 表示丢掉（复读）。"""
        candidate = (text or "").strip()
        if not candidate:
            return True  # 空文本不参与去重，交给上层处理

        if similarity(candidate, self._last_accepted) > self.threshold:
            return False

        self._last_accepted = candidate
        return True

    def reset(self) -> None:
        self._last_accepted = ""


class TextHistory:
    """纠错结果的上下文窗口：太像最近几句就不入库。"""

    def __init__(
        self,
        size: int = DEFAULT_HISTORY_SIZE,
        threshold: float = DEFAULT_HISTORY_THRESHOLD,
    ) -> None:
        self.size = size
        self.threshold = threshold
        self._entries: list[str] = []

    @property
    def entries(self) -> list[str]:
        return list(self._entries)

    def should_commit(self, text: str, lookback: int = 3) -> bool:
        """和最近几条比较，够不像才值得记下来。"""
        candidate = (text or "").strip()
        if not candidate:
            return False

        for previous in self._entries[-lookback:]:
            if similarity(candidate, previous) > self.threshold:
                return False
        return True

    def commit(self, text: str, lookback: int = 3) -> bool:
        """通过检查就入队并裁掉最旧的，返回是否记下了。"""
        if not self.should_commit(text, lookback):
            return False

        self._entries.append(text.strip())
        while len(self._entries) > self.size:
            self._entries.pop(0)
        return True
