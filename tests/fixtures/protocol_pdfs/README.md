# Protocol PDF ingestion fixtures

These PDFs are generated for manual testing of `/api/protocol-pdfs/upload` and the follow-up confirm/reject flow.

- `01_valid_basic_protocol.pdf`: normal happy path with devices and thresholds.
- `02_valid_multi_page_protocol.pdf`: multi-page source extraction and multiple thresholds.
- `03_warning_no_device.pdf`: thresholds present, no device labels.
- `04_warning_no_detection_items.pdf`: device labels present, no threshold rows.
- `05_warning_duplicate_points.pdf`: duplicate measurement point warning.
- `06_edge_decimal_negative_protocol.pdf`: negative values, decimals, percent and slash units.
