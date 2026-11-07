"""配置读取的单测：默认值、环境变量覆盖、坏值回退。"""

import os
import unittest
from unittest import mock

from ai.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_WHISPER_MODEL,
    PROJECT_ROOT,
    load_pipeline_settings,
    load_stream_settings,
)


class PipelineSettingsTest(unittest.TestCase):
    def test_defaults_match_old_hardcoded_values(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = load_pipeline_settings()

        self.assertEqual(settings.whisper_model_id, DEFAULT_WHISPER_MODEL)
        self.assertEqual(settings.llm_model_id, DEFAULT_LLM_MODEL)
        self.assertTrue(settings.use_llm)
        self.assertAlmostEqual(settings.vad_threshold, 0.15)

    def test_env_overrides(self):
        env = {
            "WHISPERFLOW_WHISPER_MODEL": "openai/whisper-base",
            "WHISPERFLOW_LLM_MODEL": "glm-4-air",
            "WHISPERFLOW_USE_LLM": "false",
            "WHISPERFLOW_VAD_THRESHOLD": "0.4",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = load_pipeline_settings()

        self.assertEqual(settings.whisper_model_id, "openai/whisper-base")
        self.assertEqual(settings.llm_model_id, "glm-4-air")
        self.assertFalse(settings.use_llm)
        self.assertAlmostEqual(settings.vad_threshold, 0.4)

    def test_bad_float_falls_back(self):
        with mock.patch.dict(os.environ, {"WHISPERFLOW_VAD_THRESHOLD": "不是数字"}, clear=True):
            self.assertAlmostEqual(load_pipeline_settings().vad_threshold, 0.15)

    def test_relative_weights_resolve_against_project_root(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = load_pipeline_settings()

        self.assertEqual(settings.resolved_weights(), PROJECT_ROOT / "ai/denoise/weights/denoise_crnn.pt")
        self.assertTrue(settings.resolved_weights().is_absolute())

    def test_stream_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            stream = load_stream_settings()

        self.assertAlmostEqual(stream.interim_interval_s, 1.5)
        self.assertEqual(stream.min_interim_samples, 20_000)


if __name__ == "__main__":
    unittest.main()
