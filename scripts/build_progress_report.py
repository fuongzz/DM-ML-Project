"""Dựng bản nháp PDF 3 trang từ kết quả đã chạy, không chạy lại thí nghiệm.

python -m pip install -r requirements-report.txt
python scripts/build_progress_report.py
"""
from pathlib import Path
import csv
import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/Group18_ProgressReport.pdf"
REPO = "https://github.com/fuongzz/DM-ML-Project"
NAVY = colors.HexColor("#17334D")
TEAL = colors.HexColor("#177C80")
PALE = colors.HexColor("#EDF5F6")


def main():
    regular = ["/System/Library/Fonts/Supplemental/Arial.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"]
    bold = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"]
    font = next((p for p in regular if Path(p).exists()), None)
    font_bold = next((p for p in bold if Path(p).exists()), None)
    if not font or not font_bold:
        raise RuntimeError("Cần font Arial hoặc DejaVu Sans hỗ trợ tiếng Việt")
    pdfmetrics.registerFont(TTFont("Body", font))
    pdfmetrics.registerFont(TTFont("BodyBold", font_bold))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TextVI", fontName="Body", fontSize=10.5, leading=15,
                              textColor=NAVY, spaceAfter=8))
    styles.add(ParagraphStyle(name="TitleVI", fontName="BodyBold", fontSize=18, leading=23,
                              textColor=NAVY, spaceAfter=12))
    styles.add(ParagraphStyle(name="HeadingVI", fontName="BodyBold", fontSize=12.5, leading=17,
                              textColor=TEAL, spaceBefore=10, spaceAfter=8))
    styles.add(ParagraphStyle(name="SmallVI", parent=styles["TextVI"], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="TableVI", parent=styles["TextVI"], fontSize=9.5,
                              leading=13, spaceAfter=0))
    styles.add(ParagraphStyle(name="TableHeadVI", parent=styles["TableVI"], fontName="BodyBold",
                              textColor=colors.white))

    def p(text, style="TextVI"):
        return Paragraph(text, styles[style])

    def table(rows, widths):
        wrapped = [[p(escape(str(v)), "TableHeadVI" if i == 0 else "TableVI")
                    for v in row] for i, row in enumerate(rows)]
        item = Table(wrapped, colWidths=widths, hAlign="LEFT", repeatRows=1)
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, 0), .5, TEAL),
        ]))
        return item

    with (ROOT / "reports/progress/dev_results.csv").open(encoding="utf-8-sig") as f:
        results = list(csv.DictReader(f))
    versions = json.loads((ROOT / "reports/progress/versions.json").read_text())
    protocol = json.loads((ROOT / "reports/progress/protocol.json").read_text())
    if protocol["test_evaluated"] or protocol["evaluation_split"] != "dev":
        raise ValueError("Báo cáo này chỉ dành cho thí nghiệm dev")

    # Dùng cùng phần giới thiệu với README để hai bản không lệch nội dung.
    problem_section = (ROOT / "README.md").read_text().split(
        "## Đề tài\n", 1
    )[1].split("\n## ", 1)[0].strip()
    problem_paragraphs = [
        p(escape(" ".join(block.splitlines())))
        for block in problem_section.split("\n\n") if block.strip()
    ]

    story = [
        p("DATA MINING - MACHINE LEARNING | NHÓM 18 | 23/11/2026", "SmallVI"),
        p("Báo cáo tiến độ<br/>Học xếp hạng lựa chọn tài sản so sánh", "TitleVI"),
        p("Trần Thanh Phương · Lê Vũ Nguyên Hoàng · Đinh Hữu Nhật Minh · Phạm Đức Anh · Tăng Nhật Minh", "SmallVI"),
        p("1. Mô tả bài toán", "HeadingVI"),
        *problem_paragraphs,
        p("2. Các tập dữ liệu", "HeadingVI"),
        p("Ngày 22/11/2026, nhóm chọn sinh dữ liệu mô phỏng thay cho việc chia sẻ dữ liệu doanh nghiệp trong mô tả đăng ký. "
          "Các phân vị diện tích, mặt tiền, KC và ngõ tham khảo từ thống kê tổng hợp; không sao chép dòng dữ liệu thật. "
          "Địa danh, mã hồ sơ, mô tả, giá và nhãn đều được sinh mới. Thay đổi này được nêu rõ trong báo cáo tiến độ."),
        table([["Bảng", "Số dòng", "Nội dung"],
               ["tstd.csv", "700", "400 train T1-T7; 100 dev T8-T9; 200 test T10/2026."],
               ["kho.csv", "5.000", "Kho dùng chung; ngày hiệu lực 01/2025-27/10/2026; 10 xã giả, 50 đường."],
               ["gui.csv", "2.800", "700 hồ sơ; mỗi hồ sơ có một dòng TSTĐ (STT 0) và ba lựa chọn (STT 1-3)."]], [85, 65, 361]),
        Spacer(1, 10),
        p("Diện tích được mô phỏng trong 20-600 m²; mặt tiền phụ thuộc nhóm diện tích, ngõ phụ thuộc KC bằng 0/dương. "
          "Văn bản có 16 tổ hợp tự viết. Các seed được cố định để tái lập. Giá sinh theo công thức giả định, chỉ dùng tính APE.", "SmallVI"),
        PageBreak(),
        p("3. Phương pháp và kết quả thí nghiệm hiện thời", "TitleVI"),
        p("Mỗi TSTĐ cùng các ứng viên hợp lệ tạo thành một nhóm xếp hạng; mỗi cặp TSTĐ - ứng viên là một dòng học. "
          "Nhãn y=1 đánh dấu ba lựa chọn mô phỏng, các ứng viên còn lại có nhãn y=0."),
        p("Ứng viên phải cùng địa bàn, mục đích và đơn vị so sánh; có ngày hiệu lực không sau TSTĐ và trong 365 ngày; khác báo cáo. "
          "Train có 109.487 cặp, dev có 27.546 cặp. Cả 400 nhóm train và 100 nhóm dev đủ ít nhất ba ứng viên; mỗi nhóm có ba nhãn dương."),
        p("Đặc trưng và cách tạo nhãn", "HeadingVI"),
        table([["Nhóm", "Đặc trưng đầu vào"],
               ["Vị trí", "Cùng đường; cùng nhóm KC=0/dương; chênh lệch KC (m)."],
               ["Kích thước", "Chênh lệch tương đối diện tích, mặt tiền và độ rộng ngõ."],
               ["Văn bản, thời gian", "TF-IDF cosine của Chi tiết; số ngày cũ của ứng viên."]], [125, 386]),
        Spacer(1, 8),
        p("Nhãn được tạo từ tổng điểm của 8 đặc trưng với trọng số tự đặt, cộng nhiễu đều [-0,5; 0,5], rồi chọn top 3. "
          "TF-IDF chỉ fit trên TSTĐ train. Mã, số báo cáo, điểm sinh nhãn và giá TSTĐ không là đầu vào mô hình. "
          "Nhóm kiểm tra nhãn trùng, nhãn tương lai, cùng báo cáo và việc xuất/đọc lại CSV."),
        p("Thiết lập và kết quả trên 100 TSTĐ dev", "HeadingVI"),
        p("LambdaRank dùng cố định 300 cây, learning rate 0,05, 15 lá, min_child_samples=20, seed 8; chưa tìm kiếm tham số. "
          "Protocol được lưu trước lần chạy. Chỉ huấn luyện train và đánh giá dev; chưa đánh giá test T10."),
    ]
    def number(value, digits=1):
        return f"{float(value):.{digits}f}".replace(".", ",")
    rows = [["Phương án", "Hit@3 (%)", "NDCG@3", "≥1 trúng (%)", "APE trung vị (%)"]]
    for row in results:
        rows.append([row["phuong_an"], number(row["Hit@3"]), number(row["NDCG@3"], 3),
                     number(row[">=1_trung_%"]), number(row["APE_trung_vi_%"])])
    story.extend([
        table(rows, [185, 70, 75, 80, 101]),
        Spacer(1, 8),
        p("Hit@3 = số nhãn dương trong top 3 / (3 × số TSTĐ). NDCG@3 dùng ba nhãn dương gốc làm mốc lý tưởng. "
          "APE so trung bình đơn giá ba ứng viên với đơn giá TSTĐ, chưa qua điều chỉnh. APE của POC tính trên 99 hồ sơ đủ ba "
          "ứng viên; các phương án khác trên 100 hồ sơ. Mọi giá đều là giá mô phỏng.", "SmallVI"),
        PageBreak(),
        p("4. Nhận xét, mã nguồn và bước tiếp theo", "TitleVI"),
        p("Nhận xét ban đầu", "HeadingVI"),
        p("LambdaRank đạt Hit@3 57,7%, cao hơn luật đơn giản 14,3 điểm phần trăm; NDCG@3 tăng từ 0,469 lên 0,618. "
          "Kết quả đáp ứng tiêu chí mô tả đặt trước cho thí nghiệm dev, chưa có kiểm định thống kê hoặc kết luận về dữ liệu thực tế."),
        p("Bộ lọc POC dùng cùng nhóm KC và tỷ lệ ngõ 0,7-1,5 lần TSTĐ, sau đó xếp hạng bằng luật đơn giản. Trần sau lọc còn "
          "46,3%, độ phủ đủ ba ứng viên là 99%. Trần bộ lọc cơ bản là 100% do nhãn được sinh ngay trong tập ứng viên này. "
          "Đây là phiên bản luật POC đơn giản hóa của project, không phải toàn bộ hệ thống doanh nghiệp."),
        p("Giới hạn", "HeadingVI"),
        p("Mô hình đang học lại quy tắc được nhóm cài trong bộ sinh, có nhiễu; nhãn không phải lựa chọn thật của chuyên viên. "
          "Văn bản chỉ gồm mẫu câu, địa bàn phân bố đều, chưa mô phỏng đầy đủ tương quan hoặc lỗi nhập liệu. Giá tổng hợp chỉ "
          "minh họa APE. Không diễn giải các chỉ số này thành hiệu quả định giá hoặc khả năng triển khai thực tế."),
        p("Liên kết mã nguồn và tái lập", "HeadingVI"),
        p(f'<link href="{REPO}" color="#177C80">{REPO}</link>'),
        p("Mã chính: src/data.py, src/experiment.py; notebook: notebooks/01_thiet_ke_du_lieu.ipynb. "
          "Sau khi cài requirements.txt, chạy lần lượt python -m src.data và python -m src.experiment. "
          "Bộ sinh chạy độc lập, không yêu cầu file dữ liệu doanh nghiệp.", "SmallVI"),
        p("BẢN NHÁP: cần commit/push phiên bản mã nguồn và notebook mới lên repo trước khi nộp liên kết. "
          "Chưa nộp Assignment hoặc công bố thay nhóm.", "SmallVI"),
        p("Các bước tiếp theo cho báo cáo cuối kì", "HeadingVI"),
        p("1. Giải thích LambdaRank và cách nhóm dữ liệu; phân tích lỗi lựa chọn theo từng TSTĐ.<br/>"
          "2. Thử bỏ đặc trưng văn bản trên dev để đo đóng góp; khảo sát tác động của lọc POC.<br/>"
          "3. Chốt câu hỏi, cấu hình và tiêu chí đạt trước khi đánh giá test T10 một lần.<br/>"
          "4. Hoàn thiện báo cáo cuối kì, slide, mã nguồn và hướng dẫn chạy lại. Không đổi bộ sinh để làm đẹp kết quả."),
        p("Môi trường lần chạy: " + "; ".join(f"{escape(k)} {escape(v)}" for k, v in versions.items()) + ".", "SmallVI"),
    ])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=42, leftMargin=42,
                                 topMargin=40, bottomMargin=44,
                                 title="Group18 Progress Report", author="Nhóm 18")
    def footer(canvas, doc):
        canvas.setStrokeColor(TEAL)
        canvas.line(42, 34, A4[0] - 42, 34)
        canvas.setFont("Body", 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(42, 22, "Nhóm 18 | Dữ liệu mô phỏng | Báo cáo tiến độ")
        canvas.drawRightString(A4[0] - 42, 22, f"{doc.page}")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
