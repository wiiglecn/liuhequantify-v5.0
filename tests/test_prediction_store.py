#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prediction_store 单元测试: 序列化 / 存储 / 命中核对。"""
import os
import unittest
import tempfile

from data_fetcher import Record
from predictor import PredictionGroup, SpecialPrediction
from dimensions import DimensionPrediction, ZodiacPool
from special_pool import SpecialPool, WidePool
from llm_reasoner import LLMResult
import prediction_store as ps


def mk(expect, special, regular=None, zodiac="鼠", wave="red"):
    return Record(expect, f"t{expect}", regular or [1, 2, 3, 4, 5, 6], special,
                  [wave] * 7, [zodiac] * 7)


def _empty_run(run_id="r", basis="2026002"):
    return {"run_id": run_id, "basis_expect": basis, "groups": [],
            "specials": [], "pools": [], "wide": {}, "zodiac": {}, "dims": []}


class TestSerialize(unittest.TestCase):
    def test_serialize_captures_metadata_and_predictions(self):
        recs = [mk("2026001", 7), mk("2026002", 8)]
        payload = {
            "records": recs,
            "report": {},
            "llm": LLMResult(inferred_models=["均匀分布"], predicted_set=[1, 2, 3, 4, 5, 6, 7],
                             reasoning="推理文本"),
            "groups": [PredictionGroup(name="A", regular=[1, 2, 3, 4, 5, 6], special=7,
                                       strategy="频率加权")],
            "sp": [SpecialPrediction(name="A", special=7, strategy="频率热号")],
            "dims": [DimensionPrediction(name="波色", value="红波", strategy="融合")],
            "pools": [SpecialPool(name="A", numbers=[1, 2, 3], strategy="频率集合")],
            "wide": WidePool(numbers=list(range(1, 21)), strategy="融合"),
            "zp": ZodiacPool(zodiacs=["鼠", "牛", "虎"], strategy="融合"),
            "zq": ZodiacPool(zodiacs=["鼠", "牛", "虎", "兔"], strategy="融合"),
            "bt": 30,
        }
        run = ps.serialize_run(payload)
        # 元数据
        self.assertEqual(run["basis_expect"], "2026002")
        self.assertEqual(run["basis_special"], 8)
        self.assertEqual(run["history_count"], 2)
        self.assertEqual(run["backtest_n"], 30)
        self.assertEqual(run["llm_models"], ["均匀分布"])
        # 预测明细可 JSON 化
        self.assertEqual(run["groups"][0]["special"], 7)
        self.assertEqual(run["groups"][0]["regular"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(run["wide"]["numbers"], list(range(1, 21)))
        self.assertEqual(run["zodiac"]["zodiacs"], ["鼠", "牛", "虎"])
        self.assertEqual(run["zodiac_quad"]["zodiacs"], ["鼠", "牛", "虎", "兔"])
        import json
        json.dumps(run)  # 不抛异常即可

    def test_serialize_without_llm(self):
        payload = {"records": [mk("2026001", 1)], "llm": None, "groups": [],
                   "sp": [], "dims": [], "pools": [], "wide": None, "zp": None,
                   "zq": None, "bt": 5}
        run = ps.serialize_run(payload)
        self.assertEqual(run["llm_models"], [])
        self.assertEqual(run["llm_reasoning"], "")
        self.assertEqual(run["wide"], {})
        self.assertEqual(run["zodiac"], {})
        self.assertEqual(run["zodiac_quad"], {})


class TestVerify(unittest.TestCase):
    def _run(self, basis="2026002", special=7, regular=None, zodiac="鼠",
             wave="red", wide=None, dims=None, zodiacs=None, zodiacs_quad=None,
             specials=None):
        return {
            "basis_expect": basis,
            "groups": [{"special": special, "regular": regular or [1, 2, 3, 4, 5, 6]}],
            "specials": specials or [{"special": special}],
            "pools": [{"numbers": [special]}],
            "wide": {"numbers": wide or list(range(1, 21))},
            "zodiac": {"zodiacs": zodiacs or ["鼠", "牛", "虎"]},
            "zodiac_quad": {"zodiacs": zodiacs_quad or ["鼠", "牛", "虎", "兔"]},
            "dims": dims or [{"name": "波色", "value": wave}],
        }

    def test_pending_when_no_next_draw(self):
        run = self._run(basis="2026099")
        recs = [mk("2026001", 1), mk("2026002", 2)]
        self.assertIsNone(ps.verify_run(run, recs))

    def test_hit_when_next_draw_matches(self):
        run = self._run(basis="2026002", special=7, wave="red",
                        dims=[{"name": "波色", "value": "红波"}, {"name": "尾数", "value": "7"}],
                        wide=list(range(1, 21)))
        recs = [mk("2026001", 1), mk("2026002", 2),
                mk("2026003", 7, regular=[1, 2, 3, 4, 5, 6], zodiac="鼠", wave="red")]
        v = ps.verify_run(run, recs)
        self.assertIsNotNone(v)
        self.assertEqual(v["actual_expect"], "2026003")
        self.assertEqual(v["actual_special"], 7)
        self.assertTrue(v["full_special_hit"])
        self.assertEqual(v["full_regular_hits"], 6)
        self.assertTrue(v["special_hit_any"])
        self.assertTrue(v["wide_hit"])
        self.assertTrue(v["zodiac_hit"])
        self.assertTrue(v["zodiac_quad_hit"])
        self.assertTrue(v["dim_hits"]["波色"])
        self.assertTrue(v["dim_hits"]["尾数"])
        self.assertEqual(v["dim_hit_count"], 2)

    def test_miss_when_next_draw_differs(self):
        run = self._run(basis="2026002", special=7,
                        regular=[10, 11, 12, 13, 14, 15], wave="blue",
                        wide=list(range(30, 50)), zodiacs=["牛", "虎", "兔"],
                        zodiacs_quad=["牛", "虎", "兔", "龙"],
                        dims=[{"name": "波色", "value": "蓝波"}])
        recs = [mk("2026001", 1), mk("2026002", 2),
                mk("2026003", 8, regular=[1, 2, 3, 4, 5, 6], zodiac="鼠", wave="red")]
        v = ps.verify_run(run, recs)
        self.assertFalse(v["full_special_hit"])
        self.assertEqual(v["full_regular_hits"], 0)
        self.assertFalse(v["special_hit_any"])
        self.assertFalse(v["wide_hit"])
        self.assertFalse(v["zodiac_hit"])
        self.assertFalse(v["zodiac_quad_hit"])
        self.assertFalse(v["dim_hits"]["波色"])
        self.assertEqual(v["dim_actuals"]["波色"], "红波")


class TestStorage(unittest.TestCase):
    def setUp(self):
        self._orig = ps.STORE_FILE
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        ps.STORE_FILE = path

    def tearDown(self):
        try:
            os.remove(ps.STORE_FILE)
        except OSError:
            pass
        ps.STORE_FILE = self._orig

    def test_save_load_roundtrip(self):
        self.assertTrue(ps.save_run(_empty_run("r1")))
        self.assertTrue(ps.save_run(_empty_run("r2")))
        loaded = ps.load_runs()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["run_id"], "r1")
        self.assertEqual(loaded[1]["run_id"], "r2")

    def test_delete_and_clear(self):
        ps.save_run(_empty_run("r1"))
        ps.save_run(_empty_run("r2"))
        self.assertTrue(ps.delete_run("r1"))
        self.assertEqual(len(ps.load_runs()), 1)
        self.assertEqual(ps.load_runs()[0]["run_id"], "r2")
        self.assertTrue(ps.clear_runs())
        self.assertEqual(ps.load_runs(), [])

    def test_load_missing_file_returns_empty(self):
        os.remove(ps.STORE_FILE)
        self.assertEqual(ps.load_runs(), [])

    def test_max_runs_cap(self):
        for i in range(ps.MAX_RUNS + 5):
            ps.save_run(_empty_run(f"r{i:04d}"))
        loaded = ps.load_runs()
        self.assertEqual(len(loaded), ps.MAX_RUNS)
        # 最旧的 5 条被丢弃, 首条为 r0005
        self.assertEqual(loaded[0]["run_id"], "r0005")


if __name__ == "__main__":
    unittest.main()
