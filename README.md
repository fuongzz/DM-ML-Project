# Learning-to-Rank chọn tài sản so sánh trong định giá bất động sản

Project môn Data Mining – Machine Learning · Nhóm 18

| Thành viên | MSSV |
|---|---|
| Trần Thanh Phương | 20235810 |
| Lê Vũ Nguyên Hoàng | 20235723 |
| Đinh Hữu Nhật Minh | 20235778 |
| Phạm Đức Anh | 20235652 |
| Tăng Nhật Minh | 20235786 |

## Đề tài

Khi cần định giá quyền sử dụng một thửa đất, chuyên viên có thể tham khảo những tài sản tương tự đã có thông tin giá rao bán
hoặc giao dịch. Chuyên viên so sánh vị trí, diện tích, mặt tiền, đường/ngõ và các điều kiện liên quan, rồi điều chỉnh những khác
biệt để ước tính giá trị tài sản cần định giá. Đây là cách tiếp cận theo phương pháp so sánh.

Tài sản cần định giá được gọi là tài sản thẩm định (TSTĐ). Những tài sản được chọn làm căn cứ đối chiếu gọi là tài sản so sánh
(TSSS). Thông tin về các tài sản có thể dùng để so sánh được lưu trong một kho dữ liệu, gọi tắt là kho TSSS.

Ví dụ giả định: cần định giá một thửa đất rộng khoảng 100 m², nằm trong ngõ rộng 4 m. Kho có nhiều tài sản ở các đường,
diện tích và thời điểm khác nhau. Chuyên viên cần tìm những tài sản phù hợp để đối chiếu, cân nhắc đồng thời nhiều yếu tố.
Một tài sản có thể cùng đường nhưng chênh lệch lớn về diện tích; tài sản khác có kích thước gần hơn nhưng ở đường khác.
Vì vậy, sau bước lọc điều kiện cơ bản, vẫn cần lựa chọn và sắp thứ tự ưu tiên giữa nhiều ứng viên.

Đề tài tập trung hỗ trợ bước lựa chọn này. Đầu vào là thông tin của một TSTĐ và kho tài sản có sẵn tại thời điểm thẩm định;
đầu ra là danh sách tối đa 3 TSSS được đề xuất để chuyên viên xem xét. Nhóm đặt bài toán dưới dạng học xếp hạng
(Learning-to-Rank): mô hình học cách ưu tiên các ứng viên từ những ví dụ lựa chọn, dựa trên mức tương đồng về vị trí, kích thước
và mô tả. Việc điều chỉnh giá và đưa ra kết luận định giá thuộc bước nghiệp vụ tiếp theo.

Trong phạm vi môn học, nhóm xây dựng dữ liệu mô phỏng và tạo các ví dụ lựa chọn bằng luật có nhiễu. Nhóm sử dụng LightGBM
LambdaRank, so sánh với cách chọn ngẫu nhiên và các quy tắc đơn giản, chia tập theo thời gian và số báo cáo mô phỏng.
Mục tiêu là thực hành và đánh giá quy trình học máy trong thiết lập mô phỏng; kết quả chưa phản ánh hiệu quả định giá thực tế.

## Mục tiêu và kế hoạch

Vận dụng kiến thức đã học trong môn Data Mining – Machine Learning, kết hợp với việc tìm hiểu thêm về xử lý dữ liệu và học xếp
hạng (Learning-to-Rank), để xây dựng mô hình hỗ trợ lựa chọn tài sản so sánh trong định giá bất động sản.

Thông qua project, nhóm hướng đến việc thực hành một quy trình học máy từ xác định bài toán, khám phá và làm sạch dữ liệu,
xây dựng đặc trưng đến huấn luyện, đánh giá và phân tích kết quả. Mô hình dự kiến sử dụng LightGBM LambdaRank để xếp hạng
các ứng viên và đề xuất tối đa 3 TSSS cho mỗi TSTĐ. Nhóm sẽ so sánh với các phương pháp cơ sở để đánh giá hiệu quả và nhận diện
những hạn chế của cách tiếp cận.

Kế hoạch dự kiến kéo dài **6 tuần**, có thể điều chỉnh trong khoảng **4–8 tuần** theo lịch môn học. Các tuần được tính từ khi
nhóm bắt đầu triển khai; phạm vi ban đầu mô phỏng bối cảnh HCM bằng địa danh giả, với vài trăm TSTĐ và một notebook chạy xuyên suốt quy trình.

| Thời gian | Công việc chính | Kết quả dự kiến |
|---|---|---|
| Tuần 1 | Thống nhất phạm vi, phân công; tìm hiểu Learning-to-Rank; thiết kế bộ sinh dữ liệu mô phỏng, luật tạo nhãn có nhiễu và điều kiện lọc ứng viên. | Mô tả bài toán, thiết kế dữ liệu và kế hoạch thí nghiệm ban đầu. |
| Tuần 2 | Chuẩn bị dữ liệu; khảo sát giá trị thiếu, trùng lặp và phân phối thuộc tính; làm sạch các trường cần dùng, ghép nhãn và chia train/dev/test theo thời gian, nhóm theo báo cáo. | Notebook khám phá dữ liệu, thống kê các tập và kiểm tra nhãn hợp lệ. |
| Tuần 3 | Sinh ứng viên, đo trần sau lọc; xây dựng phương pháp chọn ngẫu nhiên và xếp hạng theo quy tắc trên cùng tập ứng viên; đánh giá trên dev. | Kết quả cơ sở và báo cáo tiến độ tối đa 3 trang, kèm liên kết mã nguồn. |
| Tuần 4 | Xây dựng đặc trưng cặp về vị trí, kích thước và độ giống văn bản; huấn luyện LightGBM LambdaRank với cấu hình ban đầu. | Mô hình đầu tiên và bảng so sánh với các phương pháp cơ sở trên dev. |
| Tuần 5 | Phân tích các trường hợp lựa chọn chưa phù hợp; so sánh có/không có đặc trưng văn bản trên dev; chốt cấu hình và tiêu chí đánh giá trước khi chạy test một lần. | Kết quả test, nhận xét về đóng góp của đặc trưng và hạn chế của mô hình. |
| Tuần 6 | Kiểm tra notebook chạy từ đầu trên train/dev, sử dụng kết quả test đã lưu; hoàn thiện README, báo cáo cuối kì, slide và bản nộp. | Mã nguồn có hướng dẫn chạy, báo cáo và phần trình bày hoàn chỉnh. |

Chỉ số chính là **Hit@3**, bổ sung **NDCG@3**, trần sau lọc và độ phủ ứng viên để giải thích kết quả. Nhóm ghi nhận rõ nguồn
dữ liệu và các hạn chế; kết luận chỉ giới hạn trong thiết lập mô phỏng.

Nếu thời gian chỉ có 4 tuần, nhóm gộp các bước chuẩn bị và khám phá dữ liệu, giữ một cấu hình LambdaRank và ưu tiên hoàn thành
quy trình cùng các mốc so sánh. Nếu có 7–8 tuần, thời gian bổ sung dành cho xử lý phát sinh, phân tích kết quả và luyện trình bày.
Mốc báo cáo tiến độ sẽ được điều chỉnh theo hạn nộp thực tế của giảng viên, sử dụng các kết quả đã hoàn thành tại thời điểm nộp.

## Trạng thái hiện tại

**Đã hoàn thiện bộ dữ liệu mô phỏng v1 và thí nghiệm ML ban đầu ngày 23/11/2026.** Bộ sinh chạy độc lập, không đọc dữ liệu
thật; notebook đã được chuyển thành quy trình có thể chạy từ bản clone public. Test T10 chưa được đánh giá mô hình.

[Đo — `python -m src.data` và `python -m src.experiment`] Dữ liệu có 700 TSTĐ, kho dùng chung 5.000 tài sản và 2.800 dòng hồ sơ.
Train có 109.487 cặp, dev có 27.546 cặp; mỗi nhóm đúng ba nhãn dương, không có cặp trùng hoặc giá trị thiếu.
Đây là dữ liệu mô phỏng, không phải lựa chọn thực tế của chuyên viên.

## Cấu trúc và cách chạy

```text
ML_Project/
├── src/data.py                      sinh, kiểm tra, xuất/đọc dữ liệu và tạo đặc trưng
├── src/experiment.py                baseline, metric và LambdaRank trên dev
├── notebooks/01_thiet_ke_du_lieu.ipynb  quy trình chạy xuyên suốt
├── tests/test_pipeline.py           kiểm tra bất biến dữ liệu và metric
├── scripts/build_progress_report.py dựng PDF từ kết quả đã chạy
├── scripts/package_submission.py    đóng gói bản nộp bằng danh sách cho phép
├── data/gia_lap/                    CSV và manifest sinh ra, gitignore
├── reports/progress/                kết quả, protocol, phiên bản thư viện, gitignore
└── output/pdf/Group18_ProgressReport.pdf  báo cáo tiến độ 3 trang
```

Dùng Python 3.11 trở lên (lần chạy hiện tại: Python 3.12). Từ thư mục repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data
python -m unittest discover -s tests -v
python -m src.experiment
```

Hoặc mở notebook, chọn Python của `.venv` và **Run All**. Notebook không cần file doanh nghiệp hoặc kết nối mạng sau khi cài thư viện.

Để dựng lại PDF và ZIP sau khi đã có kết quả:

```bash
python -m pip install -r requirements-report.txt
python scripts/build_progress_report.py
python scripts/package_submission.py
```

## Các tập dữ liệu

Phương án mô phỏng được chốt ngày **22/11/2026**. Các tài sản được sinh mới từ tham số tổng hợp và giả định tự đặt.
Dữ liệu CSV dùng tên cột tiếng Việt tương ứng các trường gốc cần dùng; không tái tạo toàn bộ cột quản trị của dữ liệu doanh nghiệp.

| Bảng | Số dòng | Nội dung |
|---|---:|---|
| `tstd.csv` | 700 | Một tài sản mỗi báo cáo, có ngày hiệu lực và cột `Tập` |
| `kho.csv` | 5.000 | Kho dùng chung; ba tài sản mỗi báo cáo giả, báo cáo cuối có hai |
| `gui.csv` | 2.800 | 700 hồ sơ, mỗi hồ sơ có STT 0 là TSTĐ và STT 1–3 là các TSSS được chọn |

`gui.SO_BAO_CAO` nối với `tstd["Số báo cáo định giá"]`; `gui.MA_KHO` của STT 1–3 nối với `kho["Mã kho"]`.
**STT không phải nhãn y.** Với mỗi cặp TSTĐ–ứng viên, y=1 nếu mã ứng viên nằm trong ba lựa chọn của hồ sơ, còn lại y=0.

| Tập | TSTĐ | Thời gian | Cặp dùng cho ML | Nhãn dương |
|---|---:|---|---:|---:|
| Train | 400 | T1–T7/2026 | 109.487 | 1.200 |
| Dev | 100 | T8–T9/2026 | 27.546 | 300 |
| Test | 200 | T10/2026 | Chưa dựng bảng đánh giá | 600, đã sinh bằng luật cố định |

Test chỉ được sinh và kiểm tra cấu trúc/tính hợp lệ; `build_pairs` hiện chỉ cho phép train/dev. Chưa dự đoán hoặc đo Hit@3/NDCG@3
trên test. Việc đánh giá cuối cần chốt câu hỏi, cấu hình và tiêu chí trước.

**Phân biệt nguồn tham số:**

- Tham khảo thống kê tổng hợp đất ở HCM: phân vị diện tích, mặt tiền theo nhóm diện tích, tỷ lệ KC=0, phân vị KC dương và ngõ theo KC.
  Mốc đầy đủ được ghi trong `src/data.py` và `data/gia_lap/manifest.json`; không đọc hoặc sao chép dòng thật khi sinh dữ liệu.
- Giả định tự đặt: diện tích 20–600 m²; giới hạn đuôi mặt tiền/ngõ bằng 0,5×P5 và 1,5×P95; 10 xã giả, mỗi xã 5 đường, phân bố đều;
  cùng mục đích đất ở và đơn vị đ/m²; ngày chọn đều trong các cửa sổ; 16 mô tả từ mẫu câu tự viết.
- Giá hoàn toàn mô phỏng: mức nền 20–65 triệu đ/m² theo xã giả, hệ số KC/ngõ/diện tích và nhiễu log với độ lệch chuẩn 0,08, seed 30.
  Giá chỉ phục vụ APE; không phải thống kê thị trường hoặc dữ liệu giá doanh nghiệp.
- Seed các thuộc tính 18–26; nhãn train/dev/test dùng các stream riêng 27/28/29. Không sửa tham số hoặc seed để làm đẹp kết quả.

CSV và manifest lưu tại `data/gia_lap/`, vẫn được gitignore. Người dùng có thể sinh lại toàn bộ từ code. Không có dòng dữ liệu thật,
tọa độ, số nhà, tên khách hàng hoặc thông tin liên hệ trong bộ dữ liệu này.

### Bộ lọc và đặc trưng

Ứng viên cùng tỉnh/xã/mục đích/đơn vị, ngày hiệu lực nằm trong `[ngày TSTĐ − 365 ngày, ngày TSTĐ]`, khác báo cáo.
Bộ mặc định dùng mã báo cáo kho và TSTĐ khác tiền tố nên chưa có ca cùng báo cáo; test tự động có ca đưa nhãn cùng báo cáo để
kiểm tra việc loại bỏ. Trần sau bộ lọc cơ bản là 100% vì nhãn được sinh trong chính tập ứng viên hợp lệ.

Tám đặc trưng trong `FEATURES`: cùng đường, cùng nhóm KC=0/dương, lệch tương đối diện tích/mặt tiền/ngõ, lệch KC tuyệt đối,
số ngày cũ và TF-IDF cosine mô tả. TF-IDF chỉ fit trên mô tả TSTĐ train, giữ nguyên khi transform dev/test.

Bảng cặp có 14 cột: 8 đặc trưng, nhãn y, hai khóa và ba metadata (`gia_tstd`, `gia_ung_vien`, `ngo_ratio`) cho đánh giá/POC.
Mô hình chỉ nhận `FEATURES`, không nhận khóa, giá TSTĐ, metadata hoặc điểm sinh nhãn.

### Luật nhãn v1 (ghi nhận trước thí nghiệm ngày 23/11/2026)

| Thành phần điểm cơ sở | Công thức đóng góp |
|---|---|
| Cùng đường | `2 × cung_duong` |
| Cùng nhóm KC | `cung_nhom_kc` |
| Giống văn bản | `ct_sim` |
| Giống diện tích | `1 / (1 + dien_tich_lech)` |
| Giống mặt tiền | `0,5 / (1 + mat_tien_lech)` |
| Giống độ rộng ngõ | `0,5 / (1 + ngo_lech)` |
| Gần nhau về KC | `0,5 / (1 + kc_lech / 100)` |
| Độ mới | `0,25 × (1 − do_cu_ngay / 365)` |

Cộng nhiễu đều trong `[-0,5; 0,5]`, chọn ba điểm cao nhất. Các trọng số là giả định tự đặt, không phải trọng số chuyên gia xác nhận.
Nhãn được sinh từ chính các đặc trưng quan sát được; kết quả ML thể hiện khả năng học lại luật có nhiễu.

## Kết quả thí nghiệm hiện thời

[Đo — `python -m src.experiment`, 23/11/2026, **100 TSTĐ dev**] Câu hỏi đặt trước: LambdaRank có cải thiện so với luật đơn giản?
Tiêu chí mô tả: Hit@3 cao hơn luật đơn giản, NDCG@3 không thấp hơn. Protocol được ghi trước khi chạy vào `reports/progress/protocol.json`.
Chỉ chạy một cấu hình, không tìm kiếm tham số hoặc early stopping.

Cấu hình: LightGBM LambdaRank, 300 cây, learning rate 0,05, 15 lá, min_child_samples=20, seed 8; các dòng cùng báo cáo liền nhau
và `group` là số ứng viên từng TSTĐ. [Tài liệu LGBMRanker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html).

| Phương án | Hit@3 (%) | NDCG@3 | ≥1 trúng (%) | APE trung vị (%) |
|---|---:|---:|---:|---:|
| Ngẫu nhiên (seed 8) | 2,0 | 0,021 | 6,0 | 22,1 |
| Luật đơn giản | 43,3 | 0,469 | 83,0 | 12,2 |
| Lọc POC + luật đơn giản | 30,3 | 0,335 | 68,0 | 7,3 |
| LambdaRank (8 đặc trưng) | 57,7 | 0,618 | 98,0 | 8,8 |

- Luật đơn giản ưu tiên cùng đường → giống văn bản → mới hơn; cùng tập ứng viên cơ bản với ngẫu nhiên và LambdaRank.
- POC ở đây chỉ là lọc cùng nhóm KC và ngõ 0,7–1,5×, sau đó dùng luật đơn giản; không phải toàn bộ hệ thống doanh nghiệp.
  Trần sau POC là 46,3%, độ phủ đủ ba ứng viên 99%. Chưa chạy POC + LambdaRank.
- Hit@3 có mẫu số `3 × số TSTĐ`. NDCG@3 giữ mốc lý tưởng của ba nhãn gốc, không chuẩn hóa lại sau khi POC loại mất nhãn.
  Query bị lọc hết vẫn được tính vào mẫu số.
- APE dùng trung bình đơn giá ba ứng viên, chưa áp dụng tỷ lệ điều chỉnh. POC tính trên 99 hồ sơ đủ ba ứng viên, các phương án khác
  trên 100; không so APE mà bỏ qua khác biệt độ phủ. Giá và kết quả đều là mô phỏng.
- LambdaRank hơn luật đơn giản 14,3 điểm phần trăm Hit@3 trên dev; chưa kiểm định ý nghĩa thống kê hoặc kết luận khả năng thực tế.

Có 12 kiểm tra tự động: tính hợp lệ/tái lập, xuất–đọc CSV, nhãn trùng, nhãn tương lai, cùng báo cáo, giá không đi vào đặc trưng,
TF-IDF không học từ holdout, chặn dựng bảng test, mẫu số metric khi mất nhóm/nhãn và tính liên tiếp của nhóm LightGBM.
Môi trường lần chạy được lưu trong `reports/progress/versions.json`.

## Báo cáo tiến độ và bản nộp

Bản nháp tối đa 3 trang: `output/pdf/Group18_ProgressReport.pdf`, gồm bài toán, dữ liệu, kết quả dev với mã nguồn và bước tiếp theo.
Mã nguồn: [fuongzz/DM-ML-Project](https://github.com/fuongzz/DM-ML-Project).
**Cần commit/push phiên bản code và notebook mới trước khi nộp liên kết.** Chưa nộp Assignment thay nhóm.

`scripts/package_submission.py` tạo `dist/Nhom18.zip` từ danh sách cho phép: code, notebook, thư viện, kiểm tra, PDF, CSV mô phỏng
và các kết quả dev được chọn. Không đóng gói `AGENTS.md`, `CLAUDE.md`, `tham_khao/`, dữ liệu gốc/cache, bản notebook khảo sát cũ,
`.git/`, `.venv/`, `.env`, mô hình hay các file tạm. `.gitignore` chỉ kiểm soát Git, không tự lọc file khi nén ZIP.

## Bước tiếp theo cho phần ML

Giải thích LambdaRank và group; phân tích lỗi theo TSTĐ; thử bỏ văn bản trên dev; đo tác động của lọc POC. Sau đó chốt cấu hình
và tiêu chí trước khi đánh giá test T10 một lần, hoàn thiện báo cáo cuối kì và slide. Không dùng kết quả test để sửa luật sinh dữ liệu.
