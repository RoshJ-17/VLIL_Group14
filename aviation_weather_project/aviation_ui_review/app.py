
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import queue
from pathlib import Path

import pandas as pd
import streamlit as st

from io import BytesIO
from datetime import datetime

from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MODIS Aviation Weather Monitor",
    page_icon="🌦️",
    layout="wide",
)

st.title("MODIS Aviation Weather Monitoring")
st.caption(
    "MODIS satellite data · Four-model inference · "
    "Aviation weather decision support"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = SCRIPT_DIR / "decision_batch_pipeline.py"
PYTHON = sys.executable

MODEL_FILES = [
    "eda_results/cloudType/training/sampling_10epoch_comparison/"
    "tabm_sqrt_weighted_best_checkpoint.pt",
    "eda_results/cloudType/preprocessing/scaled_splits/"
    "standard_scaler.joblib",
    "tabm_cbh_output/best_tabm_cbh.pth",
    "tabm_cbh_output/cbh_preprocessing.pkl",
    "tabm_cbh_output/model_config.json",
    "tabm_full_w1_run/best_checkpoint.pt",
    "tabm_full_data/metadata.json",
    "tabm_regression_weighted_run/best_checkpoint.pt",
    "tabm_regression_data/metadata.json",
]

OUTPUT_ROOT = PROJECT_ROOT / "data/outputs/aviation_decision_support"
BATCH_ROOT = PROJECT_ROOT / "data/processed/pipeline_batches"


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

for key, default in {
    "pipeline_running": False,
    "pipeline_complete": False,
    "pipeline_error": None,
    "pipeline_batch_id": None,
    "pipeline_log": [],
    "pipeline_tempdir": None,
    "completed_batch_id": None,
    "pipeline_outputs": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------
# VALIDATION HELPERS
# --------------------------------------------------

def get_timestamp(filename):
    match = re.search(r"\.A(\d{7})\.(\d{4})\.", filename)
    return match.group(1) + match.group(2) if match else None


def validate_uploads(files, expected_prefix):
    errors = []
    timestamps = []

    for file in files:
        name = file.name

        if not name.lower().endswith(".hdf"):
            errors.append(f"Not an HDF file: {name}")
            continue

        if not name.startswith(expected_prefix):
            errors.append(f"Unexpected filename: {name}")
            continue

        timestamp = get_timestamp(name)

        if timestamp is None:
            errors.append(f"Cannot identify MODIS timestamp: {name}")
            continue

        timestamps.append(timestamp)

    if len(timestamps) != len(set(timestamps)):
        errors.append(
            f"Duplicate acquisition timestamps in {expected_prefix} uploads."
        )

    return errors, set(timestamps)


def check_model_files():
    return [
        relative
        for relative in MODEL_FILES
        if not (PROJECT_ROOT / relative).is_file()
    ]


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# PIPELINE EXECUTION
# --------------------------------------------------

def run_pipeline(command, log_queue, result_queue):
    """
    Run the actual pipeline and stream its stdout/stderr.
    No simulated progress or fabricated results.
    """

    process = None

    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            },
        )

        for line in process.stdout:
            log_queue.put(line.rstrip("\n"))

        return_code = process.wait()

        result_queue.put({
            "return_code": return_code,
            "error": None if return_code == 0 else
                f"Pipeline exited with return code {return_code}.",
        })

    except Exception as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()

        result_queue.put({
            "return_code": -1,
            "error": str(exc),
        })


def collect_outputs(batch_id):
    batch_dir = BATCH_ROOT / batch_id
    decision_dir = OUTPUT_ROOT / batch_id
    temporal_dir = decision_dir / "temporal"

    outputs = {}

    candidates = {
        "Features CSV": batch_dir / f"{batch_id}_features.csv",
        "Predictions CSV": (
            batch_dir / "predictions" / f"{batch_id}_predictions.csv"
        ),
        "Pixel decisions CSV": decision_dir / "decision_pixel_level.csv",
        "Grid decisions CSV": decision_dir / "decision_grid_025deg.csv",
        "Timestamp summary CSV": (
            decision_dir / "decision_timestamp_summary.csv"
        ),
        "Temporal grid CSV": temporal_dir / "temporal_grid_analysis.csv",
        "Temporal event log CSV": temporal_dir / "temporal_event_log.csv",
        "Temporal timeline CSV": temporal_dir / "temporal_timeline.csv",
        "Operational profile TXT": (
            decision_dir / "operational_profile_report.txt"
        ),
        "Operational profile JSON": (
            decision_dir / "operational_profile_report.json"
        ),
        "Temporal report TXT": (
            temporal_dir / "temporal_operational_report.txt"
        ),
        "Temporal report JSON": (
            temporal_dir / "temporal_operational_report.json"
        ),
    }

    for label, path in candidates.items():
        if path.is_file():
            outputs[label] = path

    return outputs
# --------------------------------------------------
# PDF REPORT GENERATION
# --------------------------------------------------

def generate_pdf_report(batch_id, outputs):
    """Create a minimal PDF report containing only detected temporal events and recommendations."""

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=f"Aviation Weather Temporal Events Report - {batch_id}",
        author="MODIS Aviation Weather Monitoring",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#174A6E"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=14,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#174A6E"),
        spaceBefore=12,
        spaceAfter=6,
    )

    event_title_style = ParagraphStyle(
        "EventTitle",
        parent=styles["Heading3"],
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#174A6E"),
        spaceBefore=6,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=12,
        spaceAfter=3,
        wordWrap="CJK",
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=body_style,
        fontSize=8,
        leading=10,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#174A6E"),
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=body_style,
        fontSize=8,
        leading=11,
    )

    recom_cell_style = ParagraphStyle(
        "RecomCell",
        parent=body_style,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("MODIS AVIATION WEATHER MONITORING", title_style))
    story.append(Paragraph(
        f"Temporal Events &amp; Aviation Recommendations — Batch: {escape(str(batch_id))}",
        subtitle_style,
    ))

    # Read temporal event log CSV and report
    event_log_path = outputs.get("Temporal event log CSV")
    temporal_report_txt_path = outputs.get("Temporal report TXT")
    temporal_report_json_path = outputs.get("Temporal report JSON")

    events_df = None
    if event_log_path and event_log_path.is_file():
        try:
            events_df = pd.read_csv(event_log_path)
        except Exception:
            events_df = None

    # Fallback to temporal report JSON if event counts are present but CSV is missing
    event_counts = {}
    if temporal_report_json_path and temporal_report_json_path.is_file():
        try:
            temporal_json = read_json(temporal_report_json_path)
            event_counts = temporal_json.get("event_counts", {})
        except Exception:
            pass

    has_events = events_df is not None and not events_df.empty

    if not has_events:
        # Check if the text report provides any info or clearly state no events detected
        story.append(Spacer(1, 10))
        no_events_msg = "No temporal weather events were detected during the observed interval."
        if event_counts and sum(event_counts.values()) == 0:
            no_events_msg = "No temporal weather events were detected during the observed interval (0 events logged)."
        
        notice_data = [[
            Paragraph(f"<b>Status:</b> {no_events_msg}", body_style)
        ]]
        notice_table = Table(notice_data, colWidths=[A4[0] - 36 * mm])
        notice_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(notice_table)
    else:
        story.append(Paragraph(f"Detected Temporal Events ({len(events_df)})", heading_style))
        story.append(Spacer(1, 4))

        available_width = A4[0] - 36 * mm

        for idx, row in events_df.iterrows():
            event_type = str(row.get("event_type", "UNKNOWN_EVENT"))
            timestamp = str(row.get("timestamp", "N/A"))
            lat = row.get("grid_lat_center", "N/A")
            lon = row.get("grid_lon_center", "N/A")

            coord_str = f"{float(lat):.2f}° N, {float(lon):.2f}° E" if pd.notna(lat) and pd.notna(lon) else "N/A"

            # CBH changes
            cbh_prev = row.get("previous_cbh_km")
            cbh_curr = row.get("current_cbh_km")
            cbh_change = row.get("cbh_change_km")
            if pd.notna(cbh_prev) and pd.notna(cbh_curr):
                change_sign = "+" if pd.notna(cbh_change) and cbh_change > 0 else ""
                change_val = f"{change_sign}{float(cbh_change):.2f} km" if pd.notna(cbh_change) else "N/A"
                cbh_str = f"Prev: {float(cbh_prev):.2f} km  →  Curr: {float(cbh_curr):.2f} km (Δ: {change_val})"
            elif pd.notna(cbh_curr):
                cbh_str = f"Curr: {float(cbh_curr):.2f} km"
            else:
                cbh_str = "None / Not Applicable"

            # Rainfall changes
            rain_rate_prev = row.get("previous_mean_detected_rain_rate_mm_hr")
            rain_rate_curr = row.get("current_mean_detected_rain_rate_mm_hr")
            rain_pix_prev = row.get("previous_rain_pixels")
            rain_pix_curr = row.get("current_rain_pixels")

            rain_parts = []
            if pd.notna(rain_pix_prev) and pd.notna(rain_pix_curr):
                rain_parts.append(f"Rain Pixels: {int(rain_pix_prev)} → {int(rain_pix_curr)}")
            elif pd.notna(rain_pix_curr) and int(rain_pix_curr) > 0:
                rain_parts.append(f"Rain Pixels: {int(rain_pix_curr)}")

            if pd.notna(rain_rate_prev) and pd.notna(rain_rate_curr):
                rain_parts.append(f"Mean Rate: {float(rain_rate_prev):.2f} → {float(rain_rate_curr):.2f} mm/hr")
            elif pd.notna(rain_rate_curr):
                rain_parts.append(f"Mean Rate: {float(rain_rate_curr):.2f} mm/hr")

            rain_str = " | ".join(rain_parts) if rain_parts else "None / Not Applicable"

            # Recommendation
            recom = row.get("aviation_recommendation")
            if pd.isna(recom) or not str(recom).strip():
                recom = "No recommendation provided."
            else:
                recom = str(recom)

            event_table_data = [
                [
                    Paragraph("Event Type", table_header_style),
                    Paragraph(f"<b>{escape(event_type)}</b>", table_cell_style),
                ],
                [
                    Paragraph("Timestamp", table_header_style),
                    Paragraph(escape(timestamp), table_cell_style),
                ],
                [
                    Paragraph("Grid Center", table_header_style),
                    Paragraph(escape(coord_str), table_cell_style),
                ],
                [
                    Paragraph("CBH Transition", table_header_style),
                    Paragraph(escape(cbh_str), table_cell_style),
                ],
                [
                    Paragraph("Rainfall Transition", table_header_style),
                    Paragraph(escape(rain_str), table_cell_style),
                ],
                [
                    Paragraph("Aviation Recommendation", table_header_style),
                    Paragraph(escape(recom), recom_cell_style),
                ],
            ]

            card_table = Table(
                event_table_data,
                colWidths=[42 * mm, available_width - 42 * mm],
            )
            card_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EDF2F7")),
                ("BACKGROUND", (1, 0), (1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))

            story.append(card_table)
            story.append(Spacer(1, 8))

    # Page number footer
    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(
            18 * mm,
            10 * mm,
            f"Batch: {batch_id}",
        )
        canvas.drawRightString(
            A4[0] - 18 * mm,
            10 * mm,
            f"Page {document.page}",
        )
        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    buffer.seek(0)
    return buffer.getvalue()

# --------------------------------------------------
# BATCH CONFIGURATION
# --------------------------------------------------

st.subheader("1. Batch configuration")

batch_id = st.text_input(
    "Batch ID",
    placeholder="aviation_20261114_001",
)

col1, col2 = st.columns(2)

with col1:
    mod06_files = st.file_uploader(
        "MOD06_L2 HDF files",
        type=["hdf"],
        accept_multiple_files=True,
        key="mod06",
    )

with col2:
    mod021_files = st.file_uploader(
        "MOD021KM HDF files",
        type=["hdf"],
        accept_multiple_files=True,
        key="mod021",
    )


# --------------------------------------------------
# PIPELINE CONFIGURATION
# --------------------------------------------------

st.subheader("2. Pipeline configuration")

col1, col2 = st.columns(2)

with col1:
    batch_size = st.number_input(
        "Inference batch size",
        min_value=1,
        value=8192,
        step=1024,
    )

with col2:
    temporal_window = st.number_input(
        "Temporal analysis window (minutes)",
        min_value=1,
        value=15,
    )


# --------------------------------------------------
# UPLOAD VALIDATION
# --------------------------------------------------

st.subheader("3. Upload validation")

valid_batch_id = bool(
    batch_id.strip()
    and re.fullmatch(r"[A-Za-z0-9_-]+", batch_id.strip())
)

if batch_id and not valid_batch_id:
    st.error(
        "Use only letters, numbers, hyphens and underscores in the batch ID."
    )

mod06_errors, mod06_times = validate_uploads(
    mod06_files, "MOD06_L2"
) if mod06_files else ([], set())

mod021_errors, mod021_times = validate_uploads(
    mod021_files, "MOD021KM"
) if mod021_files else ([], set())

for error in mod06_errors + mod021_errors:
    st.error(error)

matched_times = mod06_times & mod021_times

if mod06_files and mod021_files:
    st.write(f"Matched acquisition timestamps: **{len(matched_times)}**")
    st.write(
        f"Unmatched MOD06 files: **{len(mod06_times - matched_times)}**"
    )
    st.write(
        f"Unmatched MOD021KM files: **{len(mod021_times - matched_times)}**"
    )

    if not matched_times:
        st.error("No matching MODIS granules. Execution is blocked.")

missing_models = check_model_files()

if missing_models:
    st.error("Required model artifacts are missing:")
    for item in missing_models:
        st.write(f"- {item}")




# --------------------------------------------------
# PREFLIGHT
# --------------------------------------------------

batch_root = BATCH_ROOT / batch_id if valid_batch_id else None
decision_root = OUTPUT_ROOT / batch_id if valid_batch_id else None

batch_exists = bool(
    batch_root and (batch_root.exists() or decision_root.exists())
)

is_current_batch = (
    st.session_state.get("completed_batch_id") == batch_id
)

saved_outputs = collect_outputs(batch_id) if batch_exists else {}

required_outputs = [
    "Predictions CSV",
    "Pixel decisions CSV",
    "Operational profile TXT",
    "Operational profile JSON",
]

saved_batch_complete = all(
    name in saved_outputs for name in required_outputs
)

if batch_exists and saved_batch_complete:
    st.info(
        f"Batch ID '{batch_id}' has existing completed outputs. "
        "Results are available below. You can also re-run to overwrite this batch."
    )

    st.session_state.pipeline_complete = True
    st.session_state.pipeline_batch_id = batch_id
    st.session_state.completed_batch_id = batch_id
    st.session_state.pipeline_outputs = saved_outputs

uploads_valid = bool(
    valid_batch_id
    and mod06_files
    and mod021_files
    and not mod06_errors
    and not mod021_errors
    and matched_times
    and not missing_models
)


# --------------------------------------------------
# PIPELINE EXECUTION UI
# --------------------------------------------------

st.subheader("4. Pipeline execution")

run_clicked = st.button(
    "Run aviation weather pipeline",
    disabled=not uploads_valid or st.session_state.pipeline_running,
    use_container_width=True,
)

if run_clicked:

    tempdir = tempfile.mkdtemp(prefix="aviation_modis_")

    mod06_dir = Path(tempdir) / "MOD06"
    mod021_dir = Path(tempdir) / "MOD021"

    mod06_dir.mkdir()
    mod021_dir.mkdir()

    try:
        for file in mod06_files:
            with open(mod06_dir / Path(file.name).name, "wb") as target:
                target.write(file.getbuffer())

        for file in mod021_files:
            with open(mod021_dir / Path(file.name).name, "wb") as target:
                target.write(file.getbuffer())

        command = [
            PYTHON,
            str(PIPELINE_SCRIPT),
            "--batch-id", batch_id.strip(),
            "--mod06-dir", str(mod06_dir),
            "--mod021-dir", str(mod021_dir),
            "--batch-size", str(int(batch_size)),
            "--temporal-window-minutes", str(int(temporal_window)),
        ]

        st.session_state.pipeline_running = True
        st.session_state.pipeline_complete = False
        st.session_state.completed_batch_id = None
        st.session_state.pipeline_outputs = {}
        st.session_state.pipeline_complete = False
        st.session_state.pipeline_error = None
        st.session_state.pipeline_batch_id = batch_id.strip()
        st.session_state.pipeline_log = []
        st.session_state.pipeline_outputs = {}
        st.session_state.pipeline_tempdir = tempdir

        log_queue = queue.Queue()
        result_queue = queue.Queue()

        worker = threading.Thread(
            target=run_pipeline,
            args=(command, log_queue, result_queue),
            daemon=True,
        )

        worker.start()

        st.session_state.pipeline_worker = worker
        st.session_state.pipeline_log_queue = log_queue
        st.session_state.pipeline_result_queue = result_queue

        st.rerun()

    except Exception as exc:
        shutil.rmtree(tempdir, ignore_errors=True)
        st.error(f"Could not prepare uploaded files: {exc}")


# --------------------------------------------------
# LIVE LOGS AND COMPLETION
# --------------------------------------------------

if st.session_state.pipeline_running:

    st.info(
        f"Pipeline running for batch "
        f"{st.session_state.pipeline_batch_id}"
    )

    log_placeholder = st.empty()
    progress_placeholder = st.empty()

    log_queue = st.session_state.pipeline_log_queue
    result_queue = st.session_state.pipeline_result_queue
    worker = st.session_state.pipeline_worker

    while worker.is_alive() or not result_queue.empty():

        while not log_queue.empty():
            line = log_queue.get_nowait()
            st.session_state.pipeline_log.append(line)

        recent_logs = st.session_state.pipeline_log[-100:]

        log_placeholder.code(
            "\n".join(recent_logs) if recent_logs else
            "Waiting for pipeline output...",
            language="text",
        )

        stages = [
            "STAGE: MODIS extraction",
            "STAGE: Feature CSV combination",
            "STAGE: Four-model inference",
            "STAGE: Aviation decision support",
            "STAGE: Temporal trend analysis",
            "STAGE: Unified operational profile report",
        ]

        completed = sum(
            any(f"Completed: {stage.replace('STAGE: ', '')}" in line
                for line in st.session_state.pipeline_log)
            for stage in stages
        )

        progress_placeholder.progress(
            completed / len(stages),
            text=f"Actual completed stages: {completed}/6",
        )

        if not worker.is_alive():
            break

        import time
        time.sleep(0.5)

    while not log_queue.empty():
        st.session_state.pipeline_log.append(log_queue.get_nowait())

    result = result_queue.get() if not result_queue.empty() else {
        "return_code": -1,
        "error": "Pipeline ended without returning a result.",
    }

    st.session_state.pipeline_running = False

    if result["return_code"] == 0:
        outputs = collect_outputs(st.session_state.pipeline_batch_id)

        required_outputs = [
            "Predictions CSV",
            "Pixel decisions CSV",
            "Operational profile TXT",
            "Operational profile JSON",
        ]

        missing_outputs = [
            name for name in required_outputs if name not in outputs
        ]

        if missing_outputs:
            st.session_state.pipeline_error = (
                "Pipeline returned success, but expected output files "
                "are missing: " + ", ".join(missing_outputs)
            )
        else:
            st.session_state.pipeline_complete = True
            st.session_state.pipeline_outputs = outputs
            st.session_state.completed_batch_id = (
        st.session_state.pipeline_batch_id
    )            

    else:
        st.session_state.pipeline_error = result["error"]

    st.rerun()


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

# --------------------------------------------------
# RESTORE PREVIOUSLY COMPLETED BATCH AFTER RESTART
# --------------------------------------------------

if (
    not st.session_state.pipeline_complete
    and not st.session_state.pipeline_running
    and valid_batch_id
    and batch_exists
):
    saved_outputs = collect_outputs(batch_id)

    required_outputs = [
        "Predictions CSV",
        "Pixel decisions CSV",
        "Operational profile TXT",
        "Operational profile JSON",
    ]

    if all(name in saved_outputs for name in required_outputs):
        st.session_state.pipeline_complete = True
        st.session_state.pipeline_batch_id = batch_id
        st.session_state.completed_batch_id = batch_id
        st.session_state.pipeline_outputs = saved_outputs
        st.session_state.pipeline_error = None

st.subheader("5. Results")

if st.session_state.pipeline_error:
    st.error(
        "Pipeline failed or output verification failed: "
        + st.session_state.pipeline_error
    )

    with st.expander("Pipeline logs"):
        st.code(
            "\n".join(st.session_state.pipeline_log),
            language="text",
        )

if (
    st.session_state.pipeline_complete
    and st.session_state.pipeline_batch_id == batch_id.strip()
    and not st.session_state.pipeline_running
):

    result_batch = st.session_state.pipeline_batch_id

    st.success(f"Pipeline completed for batch: {result_batch}")

    # Display execution time if available in pipeline logs
    exec_summary_lines = []
    in_summary = False
    for line in st.session_state.pipeline_log:
        if "PIPELINE EXECUTION TIME SUMMARY" in line:
            in_summary = True
            exec_summary_lines.append(line)
        elif in_summary:
            if line.startswith("=" * 30) and len(exec_summary_lines) > 2:
                in_summary = False
            else:
                exec_summary_lines.append(line)

    if exec_summary_lines:
        with st.expander("Pipeline Execution Time Breakdown", expanded=True):
            st.code("\n".join(exec_summary_lines), language="text")

    outputs = st.session_state.pipeline_outputs

    # PDF Report Download
    st.subheader("PDF Report")

    if st.button(
        "Generate PDF report",
        key=f"generate_pdf_{result_batch}",
        use_container_width=True,
    ):
        try:
            pdf_bytes = generate_pdf_report(result_batch, outputs)

            st.download_button(
                label="Download aviation weather PDF",
                data=pdf_bytes,
                file_name=f"{result_batch}_aviation_weather_report.pdf",
                mime="application/pdf",
                key=f"download_pdf_{result_batch}",
                use_container_width=True,
            )

            st.success("PDF report generated successfully.")

        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")

    # Display Temporal Events & Recommendations in UI (mirroring the PDF report)
    st.subheader("Detected Temporal Events & Recommendations")

    event_log_path = outputs.get("Temporal event log CSV")
    temporal_report_json_path = outputs.get("Temporal report JSON")

    events_df = None
    if event_log_path and event_log_path.is_file():
        try:
            events_df = pd.read_csv(event_log_path)
        except Exception:
            events_df = None

    has_events = events_df is not None and not events_df.empty

    if not has_events:
        st.info("No temporal weather events were detected during the observed interval.")
    else:
        st.write(f"Total detected events: **{len(events_df)}**")

        for idx, row in events_df.iterrows():
            event_type = str(row.get("event_type", "UNKNOWN_EVENT"))
            timestamp = str(row.get("timestamp", "N/A"))
            lat = row.get("grid_lat_center", "N/A")
            lon = row.get("grid_lon_center", "N/A")

            coord_str = "N/A"
            if pd.notna(lat) and pd.notna(lon):
                try:
                    coord_str = f"{float(lat):.2f}° N, {float(lon):.2f}° E"
                except (ValueError, TypeError):
                    coord_str = f"{lat}, {lon}"

            # CBH changes
            cbh_prev = row.get("previous_cbh_km")
            cbh_curr = row.get("current_cbh_km")
            cbh_change = row.get("cbh_change_km")
            if pd.notna(cbh_prev) and pd.notna(cbh_curr):
                try:
                    change_sign = "+" if pd.notna(cbh_change) and float(cbh_change) > 0 else ""
                    change_val = f"{change_sign}{float(cbh_change):.2f} km" if pd.notna(cbh_change) else "N/A"
                    cbh_str = f"Prev: {float(cbh_prev):.2f} km → Curr: {float(cbh_curr):.2f} km (Δ: {change_val})"
                except (ValueError, TypeError):
                    cbh_str = f"{cbh_prev} → {cbh_curr}"
            elif pd.notna(cbh_curr):
                try:
                    cbh_str = f"Curr: {float(cbh_curr):.2f} km"
                except (ValueError, TypeError):
                    cbh_str = str(cbh_curr)
            else:
                cbh_str = "None / Not Applicable"

            # Rainfall changes
            rain_rate_prev = row.get("previous_mean_detected_rain_rate_mm_hr")
            rain_rate_curr = row.get("current_mean_detected_rain_rate_mm_hr")
            rain_pix_prev = row.get("previous_rain_pixels")
            rain_pix_curr = row.get("current_rain_pixels")

            rain_parts = []
            if pd.notna(rain_pix_prev) and pd.notna(rain_pix_curr):
                try:
                    rain_parts.append(f"Rain Pixels: {int(rain_pix_prev)} → {int(rain_pix_curr)}")
                except (ValueError, TypeError):
                    rain_parts.append(f"Rain Pixels: {rain_pix_prev} → {rain_pix_curr}")
            elif pd.notna(rain_pix_curr):
                try:
                    if int(rain_pix_curr) > 0:
                        rain_parts.append(f"Rain Pixels: {int(rain_pix_curr)}")
                except (ValueError, TypeError):
                    pass

            if pd.notna(rain_rate_prev) and pd.notna(rain_rate_curr):
                try:
                    rain_parts.append(f"Mean Rate: {float(rain_rate_prev):.2f} → {float(rain_rate_curr):.2f} mm/hr")
                except (ValueError, TypeError):
                    pass
            elif pd.notna(rain_rate_curr):
                try:
                    rain_parts.append(f"Mean Rate: {float(rain_rate_curr):.2f} mm/hr")
                except (ValueError, TypeError):
                    pass

            rain_str = " | ".join(rain_parts) if rain_parts else "None / Not Applicable"

            # Recommendation
            recom = row.get("aviation_recommendation")
            if pd.isna(recom) or not str(recom).strip():
                recom = "No recommendation provided."
            else:
                recom = str(recom)

            with st.container():
                st.markdown(f"#### Event {idx + 1}: `{event_type}`")
                cols = st.columns(3)
                with cols[0]:
                    st.write(f"**Timestamp:** {timestamp}")
                with cols[1]:
                    st.write(f"**Grid Center:** {coord_str}")
                with cols[2]:
                    st.write(f"**CBH Transition:** {cbh_str}")

                if rain_str != "None / Not Applicable":
                    st.write(f"**Rainfall Transition:** {rain_str}")

                st.info(f"**Aviation Recommendation:** {recom}")
                st.divider()

elif not st.session_state.pipeline_running:
    st.caption(
        "Upload matching MODIS files and run the pipeline to view "
        "actual predictions and downloadable aviation outputs."
    )
