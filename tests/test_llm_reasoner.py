import unittest
import os
import tempfile
from llm_reasoner import load_config, parse_llm_response, build_prompt


class TestConfig(unittest.TestCase):
    def test_missing_key_returns_none(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "config.ini")
        with open(path, "w", encoding="utf-8") as f:
            f.write("[llm]\nbase_url=https://api.deepseek.com\nmodel=deepseek-chat\napi_key=\n")
        cfg = load_config(path)
        self.assertIsNone(cfg)  # api_key 为空 -> None

    def test_valid_config(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "config.ini")
        with open(path, "w", encoding="utf-8") as f:
            f.write("[llm]\nbase_url=https://api.deepseek.com\nmodel=deepseek-chat\napi_key=sk-test\n")
        cfg = load_config(path)
        self.assertEqual(cfg["api_key"], "sk-test")
        self.assertEqual(cfg["model"], "deepseek-chat")

    def test_malformed_config_returns_none(self):
        # 含非法行(裸中文, 非注释非键值)时应降级返回 None, 不抛异常
        d = tempfile.mkdtemp()
        path = os.path.join(d, "config.ini")
        with open(path, "w", encoding="utf-8") as f:
            f.write("[llm]\nbase_url=https://api.deepseek.com\nmodel=deepseek-chat\napi_key=sk-test\n优\n")
        cfg = load_config(path)
        self.assertIsNone(cfg)


class TestParseAndPrompt(unittest.TestCase):
    def test_parse_response_extracts_models_and_numbers(self):
        text = ("反推理数学模型:\n1. 马尔可夫链\n2. 均匀分布\n\n"
                "【下期预测】\n平码: 03,12,18,25,33,41\n特码: 07\n"
                "【推理】\n基于转移矩阵")
        res = parse_llm_response(text)
        self.assertIn("马尔可夫链", res.inferred_models)
        self.assertIn("7", [str(x) for x in res.predicted_set])
        self.assertIn(41, res.predicted_set)
        self.assertIn("转移矩阵", res.reasoning)

    def test_build_prompt_contains_summary(self):
        p = build_prompt("统计摘要内容XYZ", recent_specials=[7, 8, 9])
        self.assertIn("XYZ", p)
        self.assertIn("7", p)


if __name__ == "__main__":
    unittest.main()
