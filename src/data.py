"""Bộ dữ liệu mô phỏng v1. Không đọc dữ liệu doanh nghiệp.

Các phân vị ghi cố định dưới đây là thống kê tổng hợp đã khảo sát trong
notebook. Các giới hạn đuôi, địa bàn, văn bản, giá, ngày và luật nhãn là
giả định mô phỏng. Chạy: python -m src.data
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FEATURES = ["cung_duong", "cung_nhom_kc", "dien_tich_lech", "mat_tien_lech",
            "ngo_lech", "kc_lech", "do_cu_ngay", "ct_sim"]
QUANTILES = [0, .05, .25, .5, .75, .95, 1]
AREA_ANCHORS = [20, 41, 68.84, 100, 150, 350, 600]
AREA_BINS = [0, 68.84, 100, 150, 350, float("inf")]
FRONTAGE = [[3, 3.9, 4, 4.5, 6.33], [3.9, 4.1, 5, 5, 7.18],
            [4.12, 5, 5, 6, 8.55], [4.44, 6.27, 9, 10.86, 16],
            [5.13, 12, 16.4, 22.06, 51.52]]
ALLEY = [[5, 7, 8, 14, 25], [1.66, 3, 4.5, 7.5, 13]]
TEXTS = ["Khu dân cư yên tĩnh.", "Khu ở ít ồn, sinh hoạt yên tĩnh.",
         "Gần chợ, thuận tiện mua sắm.", "Khu vực gần chợ dân sinh.",
         "Gần trường học.", "Khu vực gần trường.", "Gần khu sản xuất.",
         "Khu vực có nhiều cơ sở sản xuất."]
WINDOWS = {"train": ("2026-01-01", "2026-07-31"),
           "dev": ("2026-08-01", "2026-09-30"),
           "test": ("2026-10-01", "2026-10-31")}
LABEL_SEEDS = {"train": 27, "dev": 28, "test": 29}
NAMES = {"ma_kho": "Mã kho", "so_bc": "Số báo cáo định giá",
         "tinh": "Tỉnh/Thành phố", "xa": "Xã/Phường/Thị trấn",
         "duong": "Đường phố", "mdsd": "Mục đích sử dụng đất",
         "dien_tich": "Diện tích (m2)", "mat_tien": "Kích thước mặt tiền (m)",
         "kc": "Khoảng cách đến đường chính (m)",
         "ngo": "Độ rộng ngõ/ngách nhỏ nhất (Từ đường chính đến BĐS)",
         "chi_tiet": "Chi tiết", "ngay": "Thời điểm hiệu lực",
         "dvss": "Loại đơn giá", "gia_m2": "Đơn giá quyền sử dụng đất (đ/ĐVT)",
         "tap": "Tập"}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/gia_lap"


def _conditional_values(groups, anchors, seed):
    rng = np.random.default_rng(seed)
    values = np.full(len(groups), np.nan)
    for i, points in enumerate(anchors):
        mask = np.asarray(groups) == i
        values[mask] = np.interp(rng.random(mask.sum()), QUANTILES,
                                 [.5 * points[0], *points, 1.5 * points[-1]])
    return values.round(2)


def generate_assets():
    """Giữ nguyên các stream seed 18–26 và cách lấy mẫu của notebook cũ."""
    area = np.interp(np.random.default_rng(18).random(10_000), QUANTILES,
                     AREA_ANCHORS).round(2)
    area_group = pd.cut(area, AREA_BINS, labels=False)
    pool = pd.DataFrame({"dien_tich": area,
                         "mat_tien": _conditional_values(area_group, FRONTAGE, 19)})
    rng = np.random.default_rng(20)
    zero = rng.random(len(pool)) < 6576 / 12364
    distance = np.zeros(len(pool))
    distance[~zero] = np.interp(rng.random((~zero).sum()), QUANTILES,
                               [1, 30, 80, 150, 290, 650, 2000]).round(0)
    pool["kc"] = distance
    pool["ngo"] = _conditional_values(np.where(zero, 0, 1), ALLEY, 21)
    rng = np.random.default_rng(22)
    wards = rng.integers(1, 11, len(pool))
    streets = rng.integers(1, 6, len(pool))
    pool["xa"] = [f"XA_{i:02d}" for i in wards]
    pool["duong"] = [f"XA_{i:02d}_DUONG_{j:02d}" for i, j in zip(wards, streets)]
    contexts = np.random.default_rng(23).choice(TEXTS, len(pool))
    prefixes = np.where(zero, "Tiếp giáp đường chính.", "Vị trí trong ngõ.")
    pool["chi_tiet"] = [f"{a} {b}" for a, b in zip(prefixes, contexts)]

    # Giá hoàn toàn giả định, không phải thống kê giá VACOM. Không dùng làm feature.
    base = 20_000_000 + 5_000_000 * (wards - 1)
    price_noise = np.random.default_rng(30).normal(0, .08, len(pool))
    pool["gia_m2"] = (base * np.where(zero, 1.25, .85)
                      * (1 + .02 * np.minimum(pool["ngo"], 30))
                      * (100 / area) ** .1 * np.exp(price_noise)).round(0)
    sampled = pool.sample(n=5700, replace=False, random_state=24)
    tstd, kho = sampled.iloc[:700].copy(), sampled.iloc[700:].copy()
    tstd["ma_kho"] = [f"SIM_TD_{i + 1:05d}" for i in range(len(tstd))]
    kho["ma_kho"] = [f"SIM_KHO_{i + 1:05d}" for i in range(len(kho))]
    tstd["so_bc"] = [f"SIM_BC_TD_{i + 1:05d}" for i in range(len(tstd))]
    kho["so_bc"] = [f"SIM_BC_KHO_{i // 3 + 1:05d}" for i in range(len(kho))]
    tstd["tap"] = ["train"] * 400 + ["dev"] * 100 + ["test"] * 200
    tstd["ngay"] = pd.NaT
    rng = np.random.default_rng(25)
    for split, (start, end) in WINDOWS.items():
        mask = tstd["tap"].eq(split)
        tstd.loc[mask, "ngay"] = rng.choice(pd.date_range(start, end).to_numpy(), mask.sum())
    reports = kho["so_bc"].drop_duplicates()
    dates = pd.Series(np.random.default_rng(26).choice(
        pd.date_range("2025-01-01", "2026-10-27").to_numpy(), len(reports)), index=reports)
    kho["ngay"] = kho["so_bc"].map(dates)
    for table in (tstd, kho):
        table["ngay"] = table["ngay"].astype("datetime64[ns]")
        table["tinh"] = "SIM_HCM"
        table["mdsd"] = "Đất ở"
        table["dvss"] = "đ/m2"
    return tstd.reset_index(drop=True), kho.reset_index(drop=True)


def candidates(query, kho):
    mask = (kho["tinh"].eq(query["tinh"]) & kho["xa"].eq(query["xa"])
            & kho["mdsd"].eq(query["mdsd"]) & kho["dvss"].eq(query["dvss"])
            & kho["ngay"].between(query["ngay"] - pd.Timedelta(days=365), query["ngay"])
            & kho["so_bc"].ne(query["so_bc"]))
    return kho.loc[mask].copy()


def fit_text(tstd):
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2)).fit(
        tstd.loc[tstd["tap"].eq("train"), "chi_tiet"])


def pair_features(query, uv, vectorizer):
    f = pd.DataFrame(index=uv.index)
    f["cung_duong"] = uv["duong"].eq(query["duong"]).astype(int)
    f["cung_nhom_kc"] = uv["kc"].eq(0).eq(query["kc"] == 0).astype(int)
    for col in ("dien_tich", "mat_tien", "ngo"):
        f[f"{col}_lech"] = (uv[col] - query[col]).abs() / query[col]
    f["kc_lech"] = (uv["kc"] - query["kc"]).abs()
    f["do_cu_ngay"] = (query["ngay"] - uv["ngay"]).dt.days
    f["ct_sim"] = cosine_similarity(vectorizer.transform(uv["chi_tiet"]),
                                     vectorizer.transform([query["chi_tiet"]])).ravel()
    return f[FEATURES]


def teacher_score(f):
    """Luật nhãn v1 đã chốt, không thay đổi theo kết quả mô hình."""
    return (2 * f["cung_duong"] + f["cung_nhom_kc"] + f["ct_sim"]
            + 1 / (1 + f["dien_tich_lech"]) + .5 / (1 + f["mat_tien_lech"])
            + .5 / (1 + f["ngo_lech"]) + .5 / (1 + f["kc_lech"] / 100)
            + .25 * (1 - f["do_cu_ngay"] / 365))


def generate_labels(tstd, kho, vectorizer):
    """Sinh nhãn tất cả tập bằng luật cố định; không đánh giá mô hình trên test."""
    records = []
    for split, seed in LABEL_SEEDS.items():
        rng = np.random.default_rng(seed)
        for _, q in tstd.loc[tstd["tap"].eq(split)].iterrows():
            uv = candidates(q, kho)
            if len(uv) < 3:
                raise ValueError(f"Thiếu ứng viên: {q['so_bc']}")
            f = pair_features(q, uv, vectorizer)
            score = teacher_score(f) + rng.uniform(-.5, .5, len(f))
            selected = uv.loc[score.nlargest(3).index, "ma_kho"]
            records.append({"SO_BAO_CAO": q["so_bc"], "MA_KHO": q["ma_kho"], "STT": 0})
            records.extend({"SO_BAO_CAO": q["so_bc"], "MA_KHO": key, "STT": stt}
                           for stt, key in enumerate(selected, 1))
    return pd.DataFrame(records)


def validate_data(tstd, kho, gui):
    """Kiểm tra cấu trúc và tính hợp lệ của nhãn, không chấm điểm test."""
    if (len(tstd), len(kho), len(gui)) != (700, 5000, 2800):
        raise ValueError("Sai kích thước ba bảng")
    if not tstd["so_bc"].is_unique:
        raise ValueError("Trùng báo cáo TSTĐ")
    if set(tstd.ma_kho) & set(kho.ma_kho):
        raise ValueError("Trùng mã tài sản giữa hai bảng")
    for table, prefix in ((tstd, "SIM_TD_"), (kho, "SIM_KHO_")):
        if set(table.columns) - set(NAMES):
            raise ValueError("Có cột ngoài schema mô phỏng được phép xuất")
        if table.isna().any().any() or not table.ma_kho.is_unique:
            raise ValueError("Thiếu dữ liệu hoặc trùng mã tài sản")
        if not table.ma_kho.str.startswith(prefix).all():
            raise ValueError("Mã tài sản không thuộc bộ mô phỏng")
        if not table.xa.str.fullmatch(r"XA_\d{2}").all():
            raise ValueError("Địa bàn không phải mã giả")
        if not all(d.startswith(x + "_DUONG_") for x, d in zip(table.xa, table.duong)):
            raise ValueError("Đường không thuộc xã")
        numeric = table[["dien_tich", "mat_tien", "ngo", "kc", "gia_m2"]]
        if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any().any():
            raise ValueError("Giá trị số không hợp lệ")
        if (numeric[["dien_tich", "mat_tien", "ngo", "gia_m2"]] <= 0).any().any():
            raise ValueError("Thuộc tính phải dương")
        if not (table.tinh.eq("SIM_HCM") & table.mdsd.eq("Đất ở") & table.dvss.eq("đ/m2")).all():
            raise ValueError("Sai phạm vi mô phỏng")
    if kho.groupby("so_bc").ngay.nunique().gt(1).any():
        raise ValueError("Báo cáo kho có nhiều ngày")
    if not kho.ngay.between(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-27")).all():
        raise ValueError("Ngày kho ngoài phạm vi mô phỏng")
    if tstd.tap.value_counts().to_dict() != {"train": 400, "test": 200, "dev": 100}:
        raise ValueError("Sai số lượng tập")
    for split, (start, end) in WINDOWS.items():
        if not tstd.loc[tstd.tap.eq(split), "ngay"].between(pd.Timestamp(start), pd.Timestamp(end)).all():
            raise ValueError("Sai cửa sổ chia tập")
    if gui.duplicated(["SO_BAO_CAO", "STT"]).any() or gui.duplicated(["SO_BAO_CAO", "MA_KHO"]).any():
        raise ValueError("Nhãn lặp")
    if set(gui.SO_BAO_CAO) != set(tstd.so_bc):
        raise ValueError("Nhãn không khớp báo cáo")
    labels = {k: v for k, v in gui.groupby("SO_BAO_CAO", sort=False)}
    rows = []
    for _, q in tstd.iterrows():
        group = labels[q.so_bc]
        if sorted(group.STT.tolist()) != [0, 1, 2, 3]:
            raise ValueError("Báo cáo phải có STT 0,1,2,3")
        if group.loc[group.STT.eq(0), "MA_KHO"].iloc[0] != q.ma_kho:
            raise ValueError("STT 0 không khớp TSTĐ")
        uv = candidates(q, kho)
        if not set(group.loc[group.STT.gt(0), "MA_KHO"]).issubset(set(uv.ma_kho)):
            raise ValueError("Nhãn không phải ứng viên hợp lệ (ngày/báo cáo/địa bàn)")
        if len(uv) < 3:
            raise ValueError("Không đủ ứng viên")
        rows.append({"tap": q.tap, "so_bc": q.so_bc, "so_uv": len(uv)})
    counts = pd.DataFrame(rows)
    return counts.groupby("tap", sort=False).agg(
        so_tstd=("so_bc", "size"), so_cap=("so_uv", "sum"),
        uv_it_nhat=("so_uv", "min"), uv_trung_vi=("so_uv", "median"),
        uv_nhieu_nhat=("so_uv", "max"))


def build_pairs(tstd, kho, gui, vectorizer, split):
    if split not in ("train", "dev"):
        raise ValueError("Tiến độ chỉ dựng bảng train/dev; test dành cho đánh giá cuối")
    answers = gui.loc[gui.STT.gt(0)].groupby("SO_BAO_CAO").MA_KHO.agg(set)
    frames = []
    for _, q in tstd.loc[tstd.tap.eq(split)].iterrows():
        uv = candidates(q, kho)
        f = pair_features(q, uv, vectorizer)
        f["y"] = uv.ma_kho.isin(answers[q.so_bc]).astype(int)
        f["so_bc"], f["ma_kho"] = q.so_bc, uv.ma_kho
        # Metadata đánh giá/POC, không đưa vào FEATURES.
        f["gia_tstd"], f["gia_ung_vien"] = q.gia_m2, uv.gia_m2
        f["ngo_ratio"] = uv.ngo / q.ngo
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def save_data(tstd, kho, gui, directory=DEFAULT_DATA):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in (("tstd", tstd), ("kho", kho)):
        table.rename(columns=NAMES).to_csv(directory / f"{name}.csv", index=False,
                                           encoding="utf-8-sig", date_format="%Y-%m-%d")
    gui.to_csv(directory / "gui.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "version": "simulation-v1", "recorded_date": "23/11/2026",
        "synthetic_only": True, "test_evaluated": False,
        "rows": {"tstd": len(tstd), "kho": len(kho), "gui": len(gui)},
        "features": FEATURES, "label_seeds": LABEL_SEEDS, "label_noise": [-.5, .5],
        "reference_aggregates": {"area_interior_quantiles": AREA_ANCHORS[1:-1],
                                 "frontage_by_area": FRONTAGE, "zero_distance_rate": 6576 / 12364,
                                 "positive_distance_quantiles": [30, 80, 150, 290, 650],
                                 "alley_by_zero_distance": ALLEY},
        "assumptions": ["Các giới hạn đuôi là giả định, trừ min/max KC tham khảo",
                        "10 xã giả, mỗi xã 5 đường, phân bố đều", "16 mẫu mô tả tự viết",
                        "Đất ở, cùng đơn vị đ/m2, không có tọa độ hoặc thông tin liên hệ",
                        "Giá giả: nền 20–65 triệu đ/m2 theo xã; hệ số KC/ngõ/diện tích và nhiễu log 0,08",
                        "Nhãn học lại luật có nhiễu trên 8 đặc trưng; không đại diện chuyên viên thật",
                        "Trần sau bộ lọc cơ bản là 100% theo cách tạo nhãn",
                        "Báo cáo kho và TSTĐ có tiền tố riêng nên không phát sinh cùng báo cáo trong bản mặc định"],
        "sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                   for name in ("tstd.csv", "kho.csv", "gui.csv")},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def load_data(directory=DEFAULT_DATA):
    directory = Path(directory)
    reverse = {v: k for k, v in NAMES.items()}
    tables = [pd.read_csv(directory / f"{name}.csv", encoding="utf-8-sig").rename(columns=reverse)
              for name in ("tstd", "kho")]
    for table in tables:
        table["ngay"] = pd.to_datetime(table["ngay"], format="%Y-%m-%d").astype("datetime64[ns]")
    return *tables, pd.read_csv(directory / "gui.csv", encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    tstd, kho = generate_assets()
    gui = generate_labels(tstd, kho, fit_text(tstd))
    summary = validate_data(tstd, kho, gui)
    save_data(tstd, kho, gui, args.output)
    print(summary.to_string())
    print(f"Đã ghi tstd.csv/kho.csv/gui.csv và manifest.json tại {args.output}")


if __name__ == "__main__":
    main()
