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

_(Phương viết)_
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
