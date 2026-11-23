"""Kiểm tra bất biến dữ liệu và metric; không chạy đánh giá mô hình trên test."""
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data import (FEATURES, build_pairs, candidates, fit_text, generate_assets,
                      generate_labels, load_data, pair_features, save_data, validate_data)
from src.experiment import evaluate, group_sizes


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tstd, cls.kho = generate_assets()
        cls.vec = fit_text(cls.tstd)
        cls.gui = generate_labels(cls.tstd, cls.kho, cls.vec)

    def test_default_integrity_and_reproducibility(self):
        summary = validate_data(self.tstd, self.kho, self.gui)
        self.assertEqual(summary.loc["train", "so_cap"], 109487)
        tstd2, kho2 = generate_assets()
        pd.testing.assert_frame_equal(self.tstd, tstd2)
        pd.testing.assert_frame_equal(self.kho, kho2)
        q = self.tstd.iloc[0]
        expected = {"SIM_KHO_03860", "SIM_KHO_02617", "SIM_KHO_01629"}
        actual = self.gui.loc[self.gui.SO_BAO_CAO.eq(q.so_bc) & self.gui.STT.gt(0), "MA_KHO"]
        self.assertEqual(set(actual), expected)

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            save_data(self.tstd, self.kho, self.gui, directory)
            tstd, kho, gui = load_data(directory)
            validate_data(tstd, kho, gui)
            pd.testing.assert_frame_equal(tstd, self.tstd, check_like=True)
            pd.testing.assert_frame_equal(kho, self.kho, check_like=True)
            pd.testing.assert_frame_equal(gui, self.gui)

    def test_reject_duplicate_answer(self):
        bad = self.gui.copy()
        bad.loc[2, "MA_KHO"] = bad.loc[1, "MA_KHO"]
        with self.assertRaises(ValueError):
            validate_data(self.tstd, self.kho, bad)

    def test_reject_future_answer(self):
        q = self.tstd.iloc[0]
        future = self.kho.loc[self.kho.xa.eq(q.xa) & self.kho.ngay.gt(q.ngay)].iloc[0]
        bad = self.gui.copy()
        bad.loc[1, "MA_KHO"] = future.ma_kho
        with self.assertRaisesRegex(ValueError, "hợp lệ"):
            validate_data(self.tstd, self.kho, bad)

    def test_reject_same_report_answer(self):
        bad = self.kho.copy()
        answer = self.gui.iloc[1].MA_KHO
        bad.loc[bad.ma_kho.eq(answer), "so_bc"] = self.tstd.iloc[0].so_bc
        with self.assertRaisesRegex(ValueError, "hợp lệ"):
            validate_data(self.tstd, bad, self.gui)

    def test_price_is_not_a_feature(self):
        q = self.tstd.iloc[0].copy()
        uv = candidates(q, self.kho)
        before = pair_features(q, uv, self.vec)
        q["gia_m2"] *= 10
        uv["gia_m2"] *= 5
        pd.testing.assert_frame_equal(before, pair_features(q, uv, self.vec))
        self.assertEqual(before.columns.tolist(), FEATURES)

    def test_dev_text_does_not_fit_vocabulary(self):
        changed = self.tstd.copy()
        changed.loc[changed.tap.ne("train"), "chi_tiet"] = "zzholdoutonlyword"
        self.assertEqual(fit_text(changed).vocabulary_, self.vec.vocabulary_)

    def test_test_pairs_are_guarded(self):
        with self.assertRaisesRegex(ValueError, "đánh giá cuối"):
            build_pairs(self.tstd, self.kho, self.gui, self.vec, "test")


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.queries = pd.DataFrame({"so_bc": ["Q1", "Q2"], "gia_m2": [100., 100.]})
        self.pairs = pd.DataFrame({"so_bc": ["Q1"] * 4 + ["Q2"] * 4,
                                   "ma_kho": list("abcdefgh"),
                                   "y": [1, 1, 1, 0, 1, 1, 1, 0],
                                   "gia_ung_vien": [100.] * 8})

    def test_perfect_ranking(self):
        metrics, _ = evaluate(self.pairs, self.queries)
        self.assertEqual(metrics["Hit@3"], 100)
        self.assertAlmostEqual(metrics["NDCG@3"], 1)
        self.assertEqual(metrics["APE_trung_vi_%"], 0)

    def test_empty_query_counts_as_miss(self):
        metrics, _ = evaluate(self.pairs.iloc[:4], self.queries)
        self.assertEqual(metrics["Hit@3"], 50)
        self.assertAlmostEqual(metrics["NDCG@3"], .5)
        self.assertEqual(metrics["phu_3_uv_%"], 50)
        self.assertEqual(metrics["n_APE"], 1)

    def test_filtered_labels_keep_original_idcg(self):
        # Mỗi query chỉ còn 1 nhãn; không được báo NDCG=1 sau khi mất 2 đáp án.
        metrics, _ = evaluate(self.pairs.iloc[[0, 4]], self.queries)
        self.assertAlmostEqual(metrics["Hit@3"], 100 / 3)
        self.assertAlmostEqual(metrics["NDCG@3"], 1 / np.sum(1 / np.log2([2, 3, 4])))
        self.assertEqual(metrics["phu_3_uv_%"], 0)

    def test_group_rows_must_be_contiguous(self):
        np.testing.assert_array_equal(group_sizes(self.pairs), [4, 4])
        with self.assertRaisesRegex(ValueError, "liền nhau"):
            group_sizes(self.pairs.iloc[[0, 4, 1, 5, 2, 6, 3, 7]])


if __name__ == "__main__":
    unittest.main()
