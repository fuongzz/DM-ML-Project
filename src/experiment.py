"""Thí nghiệm tiến độ: baseline và một cấu hình LambdaRank, chỉ đánh giá dev.

Chạy sau python -m src.data: python -m src.experiment
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from .data import (DEFAULT_DATA, FEATURES, ROOT, build_pairs, fit_text,
                   load_data, validate_data)

MODEL_PARAMS = dict(objective="lambdarank", n_estimators=300, learning_rate=.05,
                    num_leaves=15, min_child_samples=20, random_state=8,
                    n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)


def group_sizes(pairs):
    """LightGBM yêu cầu các dòng của một query phải nằm liền nhau."""
    starts = pairs.so_bc.ne(pairs.so_bc.shift())
    if starts.sum() != pairs.so_bc.nunique():
        raise ValueError("Các dòng cùng nhóm không liền nhau")
    sizes = pairs.groupby("so_bc", sort=False).size().to_numpy()
    if sizes.sum() != len(pairs):
        raise ValueError("Tổng kích thước nhóm không khớp số dòng")
    return sizes


def simple_order(pairs):
    """Luật cùng đường → giống mô tả → mới hơn, mã chỉ để xử lý điểm bằng nhau."""
    return pairs.sort_values(
        ["so_bc", "cung_duong", "ct_sim", "do_cu_ngay", "ma_kho"],
        ascending=[True, False, False, True, True], kind="stable")


def score_order(pairs, scores):
    if len(scores) != len(pairs) or not np.isfinite(scores).all():
        raise ValueError("Điểm xếp hạng không hợp lệ")
    return pairs.assign(_score=np.asarray(scores)).sort_values(
        ["so_bc", "_score", "ma_kho"], ascending=[True, False, True], kind="stable")


def evaluate(ordered_pairs, queries):
    """Mẫu số cố định theo toàn bộ queries, kể cả query bị POC lọc hết.

    NDCG dùng IDCG của ba nhãn dương gốc, không chuẩn hóa lại theo nhãn còn
    sống sau lọc. APE chỉ tính khi thực sự chọn đủ ba ứng viên.
    """
    groups = {key: g for key, g in ordered_pairs.groupby("so_bc", sort=False)}
    discounts = 1 / np.log2(np.arange(2, 5))
    ideal_dcg = discounts.sum()
    rows = []
    for _, q in queries.iterrows():
        g = groups.get(q.so_bc)
        if g is None:
            y = np.array([], dtype=int)
            hits = alive = selected = 0
            ape = np.nan
        else:
            top = g.head(3)
            y = top.y.to_numpy()
            hits, alive, selected = int(y.sum()), int(g.y.sum()), len(top)
            ape = abs(top.gia_ung_vien.mean() - q.gia_m2) / q.gia_m2 * 100 if selected == 3 else np.nan
        rows.append({"so_bc": q.so_bc, "hit3": hits / 3,
                     "ndcg3": float(np.dot(y, discounts[:len(y)]) / ideal_dcg),
                     "at_least_one": hits > 0, "all_three": hits == 3,
                     "ceiling": min(alive, 3) / 3,
                     "coverage": selected == 3, "ape": ape})
    detail = pd.DataFrame(rows)
    metrics = {"n_tstd": len(queries), "Hit@3": detail.hit3.mean() * 100,
               "NDCG@3": detail.ndcg3.mean(), ">=1_trung_%": detail.at_least_one.mean() * 100,
               "du_3_%": detail.all_three.mean() * 100,
               "tran_%": detail.ceiling.mean() * 100,
               "phu_3_uv_%": detail.coverage.mean() * 100,
               "APE_trung_vi_%": detail.ape.median(),
               "n_APE": int(detail.ape.notna().sum())}
    return metrics, detail


def run_dev(data_dir=DEFAULT_DATA, output_dir=ROOT / "reports/progress"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Ghi trước khi chạy, không chọn cấu hình bằng kết quả dev và không mở test.
    protocol = {
        "report_date": "23/11/2026",
        "execution_time": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "question": "LambdaRank có cải thiện Hit@3/NDCG@3 trên dev so với luật đơn giản?",
        "hypothesis": "Kết hợp 8 đặc trưng giúp học luật tương đồng có nhiễu tốt hơn luật đơn giản",
        "success_criterion": "Hit@3 cao hơn luật đơn giản; NDCG@3 không thấp hơn. Kết quả mô tả, không tuyên bố ý nghĩa thống kê.",
        "evaluation_split": "dev", "test_evaluated": False,
        "model_params": MODEL_PARAMS, "features": FEATURES,
        "random_baseline_seed": 8, "hyperparameter_search": False,
        "limitations": "Dữ liệu/giá/nhãn mô phỏng. Nhãn được sinh từ cùng 8 đặc trưng; kết quả là học lại luật có nhiễu.",
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    tstd, kho, gui = load_data(data_dir)
    summary = validate_data(tstd, kho, gui)
    vectorizer = fit_text(tstd)
    train = build_pairs(tstd, kho, gui, vectorizer, "train")
    dev = build_pairs(tstd, kho, gui, vectorizer, "dev")
    queries = tstd.loc[tstd.tap.eq("dev")]
    results, details = [], []

    def record(name, ordered):
        metrics, per_query = evaluate(ordered, queries)
        results.append({"phuong_an": name, **metrics})
        details.append(per_query.assign(phuong_an=name))

    record("Ngẫu nhiên (seed 8)", score_order(dev, np.random.default_rng(8).random(len(dev))))
    record("Luật đơn giản", simple_order(dev))
    poc = dev.loc[dev.cung_nhom_kc.eq(1) & dev.ngo_ratio.between(.7, 1.5)]
    record("Lọc POC + luật đơn giản", simple_order(poc))
    model = lgb.LGBMRanker(**MODEL_PARAMS)
    model.fit(train[FEATURES], train.y, group=group_sizes(train))
    record("LambdaRank (8 đặc trưng)", score_order(dev, model.predict(dev[FEATURES])))
    result = pd.DataFrame(results)
    importance = pd.DataFrame({"dac_trung": FEATURES,
                               "gain": model.booster_.feature_importance(importance_type="gain")})
    importance = importance.sort_values("gain", ascending=False)
    result.to_csv(output_dir / "dev_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(details, ignore_index=True).to_csv(output_dir / "dev_per_query.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    summary.to_csv(output_dir / "data_summary.csv", encoding="utf-8-sig")
    train.to_csv(output_dir / "train_pairs.csv", index=False)
    dev.to_csv(output_dir / "dev_pairs.csv", index=False)
    model.booster_.save_model(str(output_dir / "lambdarank.txt"))
    versions = {"numpy": np.__version__, "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__}
    (output_dir / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    return result, summary, importance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/progress")
    args = parser.parse_args()
    results, _, _ = run_dev(args.data, args.output)
    print(results.round(3).to_string(index=False))
    print("Chưa đánh giá test T10. Kết quả chỉ áp dụng dữ liệu mô phỏng.")


if __name__ == "__main__":
    main()
