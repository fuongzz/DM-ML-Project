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

Đề tài xây dựng mô hình học máy chọn tối đa 3 tài sản so sánh (TSSS) cho một tài sản thẩm định (TSTĐ) trong định giá quyền sử dụng
đất theo phương pháp so sánh. Dữ liệu là kho định giá bất động sản thực tế, trong đó các TSSS chuyên viên đã chọn trong từng hồ sơ được
dùng làm nhãn. Nhóm làm sạch dữ liệu, xây dựng đặc trưng cho từng cặp TSTĐ–TSSS (vị trí, kích thước, văn bản mô tả) và huấn luyện mô
hình xếp hạng LightGBM LambdaRank, so sánh với hệ thống quy tắc cứng hiện có, chia train/test theo thời gian và theo số báo cáo.

<!-- PHUONG-VIET:BAT-DAU ke-hoach -->
## Mục tiêu và kế hoạch

Vận dụng kiến thức đã học trong môn Data Mining – Machine Learning, kết hợp với việc tìm hiểu thêm về xử lý dữ liệu và học xếp
hạng (Learning-to-Rank), để xây dựng mô hình hỗ trợ lựa chọn tài sản so sánh trong định giá bất động sản.

Thông qua project, nhóm hướng đến việc thực hành một quy trình học máy từ xác định bài toán, khám phá và làm sạch dữ liệu,
xây dựng đặc trưng đến huấn luyện, đánh giá và phân tích kết quả. Mô hình dự kiến sử dụng LightGBM LambdaRank để xếp hạng
các ứng viên và đề xuất tối đa 3 TSSS cho mỗi TSTĐ. Nhóm sẽ so sánh với các phương pháp cơ sở để đánh giá hiệu quả và nhận diện
những hạn chế của cách tiếp cận.

Kế hoạch dự kiến kéo dài **6 tuần**, có thể điều chỉnh trong khoảng **4–8 tuần** theo lịch môn học. Các tuần được tính từ khi
nhóm bắt đầu triển khai; phạm vi ban đầu tập trung vào HCM với vài trăm TSTĐ và một notebook chạy xuyên suốt quy trình.

| Thời gian | Công việc chính | Kết quả dự kiến |
|---|---|---|
| Tuần 1 | Thống nhất phạm vi, phân công, phương án dữ liệu được phép sử dụng; tìm hiểu Learning-to-Rank, cách xác định nhãn và điều kiện lọc ứng viên. Nếu dùng dữ liệu mô phỏng, xác định cách sinh và giới hạn sử dụng. | Mô tả bài toán, phương án dữ liệu và kế hoạch thí nghiệm ban đầu. |
| Tuần 2 | Chuẩn bị dữ liệu; khảo sát giá trị thiếu, trùng lặp và phân phối thuộc tính; làm sạch các trường cần dùng, ghép nhãn và chia train/dev/test theo thời gian, nhóm theo báo cáo. | Notebook khám phá dữ liệu, thống kê các tập và kiểm tra nhãn hợp lệ. |
| Tuần 3 | Sinh ứng viên, đo trần sau lọc; xây dựng phương pháp chọn ngẫu nhiên và xếp hạng theo quy tắc trên cùng tập ứng viên; đánh giá trên dev. | Kết quả cơ sở và báo cáo tiến độ tối đa 3 trang, kèm liên kết mã nguồn. |
| Tuần 4 | Xây dựng đặc trưng cặp về vị trí, kích thước và độ giống văn bản; huấn luyện LightGBM LambdaRank với cấu hình ban đầu. | Mô hình đầu tiên và bảng so sánh với các phương pháp cơ sở trên dev. |
| Tuần 5 | Phân tích các trường hợp lựa chọn chưa phù hợp; so sánh có/không có đặc trưng văn bản trên dev; chốt cấu hình và tiêu chí đánh giá trước khi chạy test một lần. | Kết quả test, nhận xét về đóng góp của đặc trưng và hạn chế của mô hình. |
| Tuần 6 | Kiểm tra notebook chạy từ đầu trên train/dev, sử dụng kết quả test đã lưu; hoàn thiện README, báo cáo cuối kì, slide và bản nộp. | Mã nguồn có hướng dẫn chạy, báo cáo và phần trình bày hoàn chỉnh. |

Chỉ số chính là **Hit@3**, bổ sung **NDCG@3**, trần sau lọc và độ phủ ứng viên để giải thích kết quả. Nhóm ghi nhận rõ nguồn
dữ liệu và các hạn chế; nếu dùng dữ liệu mô phỏng, kết luận chỉ giới hạn trong thiết lập mô phỏng.

Nếu thời gian chỉ có 4 tuần, nhóm gộp các bước chuẩn bị và khám phá dữ liệu, giữ một cấu hình LambdaRank và ưu tiên hoàn thành
quy trình cùng các mốc so sánh. Nếu có 7–8 tuần, thời gian bổ sung dành cho xử lý phát sinh, phân tích kết quả và luyện trình bày.
Mốc báo cáo tiến độ sẽ được điều chỉnh theo hạn nộp thực tế của giảng viên, sử dụng các kết quả đã hoàn thành tại thời điểm nộp.
<!-- PHUONG-VIET:KET-THUC ke-hoach -->

## Trạng thái hiện tại

Repo đang ở giai đoạn chuẩn bị. `src/` và `notebooks/` chưa có code triển khai; chưa có kết quả thí nghiệm hay báo cáo của nhóm.
Bản đóng gói hiện tại là khung project, chưa đủ để chạy huấn luyện và đánh giá.

## Cấu trúc phần nộp

```
Nhom18/
├── README.md        giới thiệu, trạng thái và hướng dẫn chạy
├── requirements.txt thư viện cần cài
├── notebooks/       khám phá, học từng bước
└── src/             code của nhóm
```

Khi hoàn thiện, bổ sung bộ dữ liệu demo được phép chia sẻ, notebook chính và báo cáo theo yêu cầu môn học.

## Dữ liệu

Dữ liệu doanh nghiệp gốc không kèm trong bản nộp. Nhóm chưa chốt phương án dữ liệu dùng để nộp; hiện chưa có bộ dữ liệu demo
hoặc mô phỏng đi kèm. Nếu dùng dữ liệu mô phỏng, nhóm sẽ ghi rõ cách sinh dữ liệu, cách tạo nhãn và giới hạn của kết quả.

## Bắt đầu

Môi trường dự kiến: Python 3.11. Từ thư mục project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Hiện chỉ có thể chuẩn bị môi trường. Hướng dẫn chạy notebook, đường dẫn dữ liệu và cách tái lập kết quả sẽ được bổ sung khi có
phần triển khai.

## Đóng gói bản nộp

Gói hiện tại nằm tại `dist/Nhom18.zip`, với thư mục tương ứng là `dist/Nhom18/`. Đây là bản chụp tại thời điểm đóng gói;
cần đóng gói lại sau khi bổ sung code, dữ liệu demo và báo cáo.

Chỉ đưa vào gói các file phục vụ bài nộp. Các thành phần sau được giữ ngoài gói:

- `AGENTS.md`, `CLAUDE.md` và thư mục `tham_khao/`.
- Dữ liệu doanh nghiệp, cache và kết quả trung gian trong `data/`, `reports/`.
- `.git/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store` và `.gitkeep`.
- `.env`, thông tin đăng nhập và khóa API.

Trước bản nộp cuối, kiểm tra output notebook và file đính kèm để không chứa dữ liệu doanh nghiệp hoặc thông tin cá nhân.
`.gitignore` chỉ kiểm soát Git, không tự loại file khi nén ZIP.
