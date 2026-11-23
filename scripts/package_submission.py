"""Đóng gói bằng danh sách cho phép; không nén toàn bộ repo làm việc."""
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "dist/Nhom18"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    files = [ROOT / name for name in ("README.md", "requirements.txt", "requirements-report.txt")]
    for folder, pattern in (("src", "*.py"), ("tests", "*.py"), ("scripts", "*.py"),
                            ("notebooks", "*.ipynb")):
        files.extend(p for p in (ROOT / folder).rglob(pattern)
                     if not any(part.startswith('.') or part == "__pycache__" for part in p.relative_to(ROOT).parts))
    files.extend(ROOT / "data/gia_lap" / name
                 for name in ("tstd.csv", "kho.csv", "gui.csv", "manifest.json"))
    files.extend(ROOT / "reports/progress" / name for name in (
        "dev_results.csv", "data_summary.csv", "feature_importance.csv", "protocol.json", "versions.json"))
    files.append(ROOT / "output/pdf/Group18_ProgressReport.pdf")
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(f"Thiếu file cần đóng gói: {source.relative_to(ROOT)}")
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    archive = ROOT / "dist/Nhom18.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for source in sorted(destination.rglob('*')):
            if source.is_file():
                bundle.write(source, "Nhom18/" + source.relative_to(destination).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert len(bundle.namelist()) == len(files)
    print(f"{archive}: {len(files)} file, {archive.stat().st_size:,} byte")


if __name__ == "__main__":
    main()
