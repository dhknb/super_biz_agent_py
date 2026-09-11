"""Generate protocol PDF ingestion fixtures."""

from pathlib import Path

import fitz


OUT_DIR = Path(__file__).resolve().parent
FONT_CANDIDATES = [
    Path("/mnt/c/Windows/Fonts/simhei.ttf"),
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    Path("/mnt/c/Windows/Fonts/simsun.ttc"),
]


PDFS = {
    "01_valid_basic_protocol.pdf": [
        [
            "新设备检测协议",
            "设备: 低压配电柜 A1",
            "型号: LVD-2026",
            "温度测点 <= 80 ℃",
            "电流阈值 >= 10 A",
            "湿度测点 <= 65 %",
            "备注: 该样例应进入 awaiting_confirmation，validation.valid=true。",
        ]
    ],
    "02_valid_multi_page_protocol.pdf": [
        [
            "多页综合检测协议",
            "设备: 智能环控柜 B2",
            "装置: 边缘采集终端 EG-300",
            "第一页说明: 本页包含基础设备与部分阈值。",
            "柜内温度 <= 45 ℃",
            "输入电流 >= 5 A",
        ],
        [
            "第二页检测规则",
            "输出电压 >= 220 V",
            "噪声水平 <= 70 dB",
            "信号强度 >= -85 dBm",
            "备注: 用于验证 page source 能定位到第 2 页。",
        ],
    ],
    "03_warning_no_device.pdf": [
        [
            "缺少设备字段协议",
            "温度测点 <= 75 ℃",
            "压力测点 >= 1.5 MPa",
            "备注: 没有 设备/型号/装置 行，应产生 devices warning，但仍可 dry-run。",
        ]
    ],
    "04_warning_no_detection_items.pdf": [
        [
            "缺少检测项协议",
            "设备: 备用发电机 G1",
            "型号: GEN-800",
            "本协议只描述设备范围，不包含任何 >=、<=、<、>、= 数值阈值。",
            "备注: 应产生 detection_items warning。",
        ]
    ],
    "05_warning_duplicate_points.pdf": [
        [
            "重复测点协议",
            "设备: 水泵控制柜 P1",
            "水压测点 >= 0.3 MPa",
            "水压测点 <= 1.2 MPa",
            "流量测点 >= 20 L/min",
            "备注: 同一个 measurement_point 出现两次，应产生重复测点 warning。",
        ]
    ],
    "06_edge_decimal_negative_protocol.pdf": [
        [
            "边界数值检测协议",
            "设备: 冷站传感器组 C9",
            "压差测点 >= -3.5 kPa",
            "振动速度 <= 4.5 mm/s",
            "氧气浓度 >= 19.5 %",
            "温升速率 < 2.25 ℃",
            "备注: 覆盖负数、小数、百分比、斜杠单位和小于号。",
        ]
    ],
}


def find_font() -> Path:
    for font in FONT_CANDIDATES:
        if font.exists():
            return font
    raise RuntimeError("No Chinese-capable font found under /mnt/c/Windows/Fonts")


def write_pdf(filename: str, pages: list[list[str]], font_path: Path) -> None:
    page_width, page_height = fitz.paper_size("a4")
    doc = fitz.open()
    for page_index, lines in enumerate(pages, start=1):
        page = doc.new_page(width=page_width, height=page_height)
        page.insert_font(fontname="TestCN", fontfile=str(font_path))
        y = 72
        for line_index, line in enumerate(lines):
            font_size = 18 if page_index == 1 and line_index == 0 else 12
            page.insert_text(
                (72, y),
                line,
                fontname="TestCN",
                fontsize=font_size,
                fill=(0, 0, 0),
            )
            y += 30 if font_size == 18 else 22
        footer = f"fixture {filename} page {page_index} of {len(pages)}"
        page.insert_text(
            (72, page_height - 54),
            footer,
            fontname="helv",
            fontsize=9,
            fill=(0.35, 0.35, 0.35),
        )
    doc.save(OUT_DIR / filename)
    doc.close()


def write_readme() -> None:
    (OUT_DIR / "README.md").write_text(
        "# Protocol PDF ingestion fixtures\n\n"
        "These PDFs are generated for manual testing of `/api/protocol-pdfs/upload` "
        "and the follow-up confirm/reject flow.\n\n"
        "- `01_valid_basic_protocol.pdf`: normal happy path with devices and thresholds.\n"
        "- `02_valid_multi_page_protocol.pdf`: multi-page source extraction and multiple thresholds.\n"
        "- `03_warning_no_device.pdf`: thresholds present, no device labels.\n"
        "- `04_warning_no_detection_items.pdf`: device labels present, no threshold rows.\n"
        "- `05_warning_duplicate_points.pdf`: duplicate measurement point warning.\n"
        "- `06_edge_decimal_negative_protocol.pdf`: negative values, decimals, percent and slash units.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font_path = find_font()
    for filename, pages in PDFS.items():
        write_pdf(filename, pages, font_path)
    write_readme()
    for path in sorted(OUT_DIR.glob("*.pdf")):
        print(path)


if __name__ == "__main__":
    main()
