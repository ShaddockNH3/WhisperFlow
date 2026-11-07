"""幻觉拦截的单测。只用标准库，不需要 torch。"""

import unittest

from ai.dedup import RepeatFilter, TextHistory, similarity


class SimilarityTest(unittest.TestCase):
    def test_identical_texts(self):
        self.assertAlmostEqual(similarity("今天天气不错", "今天天气不错"), 1.0)

    def test_empty_side_is_zero(self):
        self.assertEqual(similarity("", "内容"), 0.0)
        self.assertEqual(similarity("内容", ""), 0.0)
        self.assertEqual(similarity(None, None), 0.0)

    def test_different_texts_low(self):
        self.assertLess(similarity("今天天气不错", "明天要下雨了"), 0.5)


class RepeatFilterTest(unittest.TestCase):
    def test_first_phrase_always_accepted(self):
        self.assertTrue(RepeatFilter().accept("你好世界"))

    def test_identical_second_phrase_rejected(self):
        guard = RepeatFilter()
        guard.accept("你好世界")
        self.assertFalse(guard.accept("你好世界"))

    def test_similar_above_threshold_rejected(self):
        guard = RepeatFilter(threshold=0.6)
        guard.accept("今天天气真不错啊")
        self.assertFalse(guard.accept("今天天气真不错呀"))

    def test_different_phrase_accepted_and_updates_state(self):
        guard = RepeatFilter()
        guard.accept("你好世界")
        self.assertTrue(guard.accept("明天要下雨了"))
        self.assertEqual(guard.last_accepted, "明天要下雨了")

    def test_empty_text_does_not_change_state(self):
        guard = RepeatFilter()
        guard.accept("你好世界")
        self.assertTrue(guard.accept("   "))
        self.assertEqual(guard.last_accepted, "你好世界")

    def test_reset_forgets_state(self):
        guard = RepeatFilter()
        guard.accept("你好世界")
        guard.reset()
        self.assertTrue(guard.accept("你好世界"))


class TextHistoryTest(unittest.TestCase):
    def test_commit_then_reject_repeat(self):
        history = TextHistory()
        self.assertTrue(history.commit("今天讲的是语音识别"))
        self.assertFalse(history.commit("今天讲的是语音识别"))

    def test_empty_not_committed(self):
        self.assertFalse(TextHistory().commit("   "))

    def test_size_limit_drops_oldest(self):
        history = TextHistory(size=2, threshold=0.99)
        history.commit("第一句")
        history.commit("完全不同的第二句")
        history.commit("再一句不同的话")
        self.assertEqual(len(history.entries), 2)
        self.assertEqual(history.entries[-1], "再一句不同的话")

    def test_lookback_limits_comparison_window(self):
        history = TextHistory(size=10, threshold=0.8)
        history.commit("甲甲甲甲")
        history.commit("乙乙乙乙")
        history.commit("丙丙丙丙")
        # 只比最近两条，甲不在窗口里
        self.assertTrue(history.should_commit("甲甲甲甲", lookback=2))

    def test_entries_returns_copy(self):
        history = TextHistory()
        history.commit("一句话")
        snapshot = history.entries
        snapshot.append("脏数据")
        self.assertEqual(len(history.entries), 1)


if __name__ == "__main__":
    unittest.main()
