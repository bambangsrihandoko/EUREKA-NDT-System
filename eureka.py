import streamlit as st
import cv2
import numpy as np
import math
import datetime
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
import os
from fpdf import FPDF
import base64
import html

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Base page configuration
st.set_page_config(page_title="EUREKA - NDT AI Inspection", layout="wide", page_icon="☢️")


def inject_global_styles():
    st.markdown(
        """
        <style>
            :root {
                --eureka-ink: #111111;
                --eureka-muted: #5f6673;
                --eureka-line: #d8dde6;
                --eureka-surface: #ffffff;
                --eureka-soft: #f5f6f8;
                --eureka-primary: #e7302a;
                --eureka-primary-dark: #bf231f;
                --eureka-primary-soft: #fff0ef;
                --eureka-danger: #b42318;
                --eureka-success: #087443;
                --eureka-warn: #b54708;
            }

            .stApp {
                background: #f5f6f8;
                color: var(--eureka-ink);
            }

            .stApp, .stApp p, .stApp label {
                color: var(--eureka-ink);
            }

            .block-container {
                padding-top: 1.35rem;
                padding-bottom: 2.5rem;
                max-width: 1500px;
            }

            section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--eureka-line);
            }

            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            section[data-testid="stSidebar"] label {
                color: var(--eureka-ink);
            }

            h1, h2, h3 {
                letter-spacing: 0;
                color: var(--eureka-ink);
            }

            small, [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] {
                color: var(--eureka-muted) !important;
            }

            div[data-testid="stButton"] > button,
            div[data-testid="stDownloadButton"] > button {
                border-radius: 7px;
                border: 1px solid #cfd5df;
                background: #ffffff;
                color: var(--eureka-ink);
                font-weight: 650;
            }

            div[data-testid="stButton"] > button[kind="primary"],
            div[data-testid="stDownloadButton"] > button[kind="primary"] {
                background: var(--eureka-primary);
                border-color: var(--eureka-primary);
                color: #ffffff;
            }

            div[data-testid="stButton"] > button[kind="primary"] *,
            div[data-testid="stDownloadButton"] > button[kind="primary"] * {
                color: #ffffff !important;
            }

            div[data-testid="stButton"] > button[kind="primary"]:hover,
            div[data-testid="stDownloadButton"] > button[kind="primary"]:hover {
                background: var(--eureka-primary-dark);
                border-color: var(--eureka-primary-dark);
                color: #ffffff;
            }

            input,
            textarea,
            div[data-baseweb="input"] > div,
            div[data-baseweb="textarea"] > div,
            div[data-baseweb="select"] > div,
            div[data-baseweb="base-input"] {
                background: #ffffff !important;
                color: var(--eureka-ink) !important;
                border-color: #cfd5df !important;
            }

            input::placeholder,
            textarea::placeholder {
                color: #8b94a3 !important;
            }

            div[data-baseweb="input"] svg,
            div[data-baseweb="select"] svg {
                color: var(--eureka-ink) !important;
                fill: var(--eureka-ink) !important;
            }

            div[data-testid="stNumberInput"] button {
                background: #ffffff !important;
                color: var(--eureka-ink) !important;
                border-color: #cfd5df !important;
            }

            div[data-testid="stRadio"] label,
            div[data-testid="stRadio"] label span,
            div[data-testid="stCheckbox"] label,
            div[data-testid="stCheckbox"] label span,
            div[data-testid="stToggle"] label,
            div[data-testid="stToggle"] label span {
                color: var(--eureka-ink) !important;
            }

            div[data-testid="stAlert"] {
                border-radius: 8px;
                border: 1px solid #e6ebf2;
            }

            div[data-testid="stAlert"] p,
            div[data-testid="stAlert"] div,
            div[data-testid="stAlert"] span {
                color: var(--eureka-ink) !important;
            }

            div[data-testid="stDataFrame"],
            div[data-testid="stDataEditor"] {
                border: 1px solid var(--eureka-line);
                border-radius: 8px;
                overflow: hidden;
                background: var(--eureka-surface);
            }

            .eureka-header {
                border: 1px solid var(--eureka-line);
                border-top: 4px solid var(--eureka-primary);
                background: #ffffff;
                border-radius: 8px;
                padding: 20px 22px 18px;
                margin-bottom: 16px;
            }

            .eureka-eyebrow {
                color: var(--eureka-primary);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 6px;
            }

            .eureka-header h1 {
                font-size: 2rem;
                line-height: 1.15;
                margin: 0 0 6px;
            }

            .eureka-header p {
                color: var(--eureka-muted);
                font-size: 0.98rem;
                margin: 0;
                max-width: 980px;
            }

            .eureka-card {
                border: 1px solid var(--eureka-line);
                background: var(--eureka-surface);
                border-radius: 8px;
                padding: 15px 16px;
                min-height: 98px;
                margin-bottom: 12px;
            }

            .eureka-card-label {
                color: var(--eureka-muted);
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 5px;
            }

            .eureka-card-value {
                color: var(--eureka-ink);
                font-size: 1.18rem;
                font-weight: 800;
                line-height: 1.2;
                word-break: break-word;
            }

            .eureka-card-detail {
                color: var(--eureka-muted);
                font-size: 0.84rem;
                margin-top: 6px;
            }

            .eureka-panel {
                border: 1px solid var(--eureka-line);
                background: var(--eureka-surface);
                border-radius: 8px;
                padding: 14px 16px 16px;
                margin-bottom: 14px;
            }

            .eureka-panel-title {
                color: var(--eureka-ink);
                font-size: 1.02rem;
                font-weight: 800;
                margin-bottom: 2px;
            }

            .eureka-panel-caption {
                color: var(--eureka-muted);
                font-size: 0.86rem;
                margin-bottom: 10px;
            }

            .eureka-pill {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 0.78rem;
                font-weight: 800;
                border: 1px solid transparent;
            }

            .eureka-pill-success {
                color: var(--eureka-success) !important;
                background: #ecfdf3;
                border-color: #abefc6;
            }

            .eureka-pill-warn {
                color: var(--eureka-warn) !important;
                background: #fffaeb;
                border-color: #fedf89;
            }

            .eureka-pill-danger {
                color: var(--eureka-danger) !important;
                background: #fef3f2;
                border-color: #fecdca;
            }

            .eureka-badge-red {
                color: var(--eureka-primary) !important;
                background: var(--eureka-primary-soft);
                border-color: #ffc9c6;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title, subtitle, eyebrow="EUREKA"):
    st.markdown(
        f"""
        <div class="eureka-header">
            <div class="eureka-eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label, value, detail=""):
    st.markdown(
        f"""
        <div class="eureka-card">
            <div class="eureka-card-label">{html.escape(label)}</div>
            <div class="eureka-card-value">{html.escape(str(value))}</div>
            <div class="eureka-card-detail">{html.escape(str(detail))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(text, tone="success"):
    tone_class = {
        "success": "eureka-pill-success",
        "warning": "eureka-pill-warn",
        "danger": "eureka-pill-danger",
    }.get(tone, "eureka-pill-success")
    st.markdown(
        f'<span class="eureka-pill {tone_class}">{html.escape(text)}</span>',
        unsafe_allow_html=True,
    )


inject_global_styles()

@st.cache_resource
def load_model():
    """Load the YOLOv8 model for weld defect segmentation."""
    model_path = os.path.join(BASE_DIR, "models", "eureka_industri.pt")
    return YOLO(model_path)

model = load_model()

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
KOLOM_TABEL = ["Sumber", "Tipe Cacat", "Conf (%)", "Dimensi (px)", "Luas (px²)", "Dimensi (mm)", "Luas (mm²)"]

if 'history_inspeksi' not in st.session_state:
    st.session_state['history_inspeksi'] = []
if 'px_to_mm_ratio' not in st.session_state:
    st.session_state['px_to_mm_ratio'] = None
if 'calibration_done' not in st.session_state:
    st.session_state['calibration_done'] = False
if 'ai_ran' not in st.session_state:
    st.session_state['ai_ran'] = False
if 'df_defects' not in st.session_state:
    st.session_state['df_defects'] = pd.DataFrame(columns=KOLOM_TABEL)
if 'jumlah_gambar_manual' not in st.session_state:
    st.session_state['jumlah_gambar_manual'] = 0


def has_valid_calibration():
    """Return True only after the user applies a digital calibration."""
    return (
        st.session_state.get('calibration_done', False)
        and st.session_state.get('px_to_mm_ratio') is not None
    )


def apply_calibration_to_defects(df, ratio, max_area_mm2=None):
    """Fill metric columns after calibration and keep the AI area filter in mm²."""
    calibrated_df = df.copy()
    if calibrated_df.empty:
        return pd.DataFrame(columns=KOLOM_TABEL)

    calibrated_df["Dimensi (mm)"] = calibrated_df["Dimensi (px)"].astype(float) * ratio
    calibrated_df["Luas (mm²)"] = calibrated_df["Luas (px²)"].astype(float) * (ratio**2)

    if max_area_mm2 is not None and "Sumber" in calibrated_df.columns:
        ai_rows = calibrated_df["Sumber"].astype(str).str.startswith("AI-")
        valid_area = calibrated_df["Luas (mm²)"] <= max_area_mm2
        calibrated_df = calibrated_df[(~ai_rows) | valid_area]

    return calibrated_df.reset_index(drop=True)


def make_defect_row(source, defect_type, confidence, dimension_px, area_px):
    """Build one defect row; metric values stay unknown until calibration is applied."""
    row = {
        "Sumber": source,
        "Tipe Cacat": defect_type,
        "Conf (%)": float(confidence),
        "Dimensi (px)": float(dimension_px),
        "Luas (px²)": float(area_px),
        "Dimensi (mm)": np.nan,
        "Luas (mm²)": np.nan,
    }

    if has_valid_calibration():
        ratio = st.session_state['px_to_mm_ratio']
        row["Dimensi (mm)"] = float(dimension_px * ratio)
        row["Luas (mm²)"] = float(area_px * (ratio**2))

    return row

# ==========================================
# SIDEBAR: LOGOS, NAVIGATION, AND AI PARAMETERS
# ==========================================
path_brin = os.path.join(BASE_DIR, "BRIN_logo.png")
path_poltek = os.path.join(BASE_DIR, "Poltek Nuklir.png")

# Institution logo injection (Base64)
if os.path.exists(path_brin) and os.path.exists(path_poltek):
    with open(path_brin, "rb") as f:
        brin_b64 = base64.b64encode(f.read()).decode()
    with open(path_poltek, "rb") as f:
        poltek_b64 = base64.b64encode(f.read()).decode()
    
    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
            <img src="data:image/png;base64,{brin_b64}" style="height: 60px; margin-right: 20px;">
            <img src="data:image/png;base64,{poltek_b64}" style="height: 60px;">
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown(
    """
    <div style="padding: 4px 2px 8px;">
        <div style="font-size: 1.35rem; font-weight: 850; color: #17202a; line-height: 1;">EUREKA</div>
        <div style="font-size: 0.78rem; color: #667085; margin-top: 4px;">Radiographic DSS Workstation</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

# Main navigation
menu = st.sidebar.selectbox("Navigation", ["Home", "Single Inspection", "Batch Inspection", "Report Generator"])

st.sidebar.markdown("---")
st.sidebar.markdown("### AI Parameters")
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.99, value=0.55, step=0.05)
batas_luas_maks = st.sidebar.number_input(
    "Maximum Area Filter (mm²)",
    min_value=1.0,
    max_value=5000.0,
    value=50.0,
    step=5.0,
    help="Ignores AI predictions when the calibrated area exceeds this limit (Out-of-Distribution/Base Metal filter).",
)
gunakan_clahe = st.sidebar.checkbox("Enable CLAHE Filter", value=False, help="Normalize industrial radiography image contrast.")

# ==========================================
# REPORT GENERATOR FUNCTION (PDF)
# ==========================================
def generate_pdf_report(nama_file, tgl_uji, lokasi, inspektor, perusahaan, client, tebal_las, status_akhir, df_final, catatan, rasio, image_array=None):
    """Generate a formal PDF inspection report."""
    pdf = FPDF()
    pdf.add_page()
    
    # Letterhead injection.
    path_kop = os.path.join(BASE_DIR, "kop.jpg")
    if os.path.exists(path_kop):
        pdf.image(path_kop, x=10, y=10, w=190)
        pdf.ln(40) 
    else:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="[LETTERHEAD NOT FOUND]", ln=True, align='C')
        pdf.ln(10)

    # Report header.
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 8, txt="RADIOGRAPHIC EVALUATION REPORT (NDT)", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 6, txt="Generated by EUREKA System (AWS D1.1 Standard)", ln=True, align='C')
    pdf.ln(8)
    
    # Inspection administrative data.
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(200, 6, txt="INSPECTION DATA:", ln=True)
    pdf.set_font("Arial", '', 10)
    
    nama_file_display = f"{nama_file[:30]}..." if len(nama_file) > 30 else nama_file
    nama_client_display = client if client else "-"
    
    pdf.cell(35, 6, txt="Image File")
    pdf.cell(55, 6, txt=f": {nama_file_display}")
    pdf.cell(25, 6, txt="Inspector")
    pdf.cell(75, 6, txt=f": {inspektor}", ln=True)
    
    pdf.cell(35, 6, txt="Test Date")
    pdf.cell(55, 6, txt=f": {tgl_uji}")
    pdf.cell(25, 6, txt="Company")
    pdf.cell(75, 6, txt=f": {perusahaan}", ln=True)
    
    pdf.cell(35, 6, txt="Test Location")
    pdf.cell(55, 6, txt=f": {lokasi}")
    pdf.cell(25, 6, txt="Client")
    pdf.cell(75, 6, txt=f": {nama_client_display}", ln=True)
    
    pdf.cell(35, 6, txt="Weld Thickness (S)")
    pdf.cell(55, 6, txt=f": {tebal_las} mm")
    pdf.cell(25, 6, txt="Calibration")
    pdf.cell(75, 6, txt=f": 1 px = {rasio:.4f} mm", ln=True)
    pdf.ln(8)
    
    # =======================================================
    # Inspection visualization injection.
    # =======================================================
    if image_array is not None:
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(200, 6, txt="DISCONTINUITY VISUALIZATION & INSPECTOR ANNOTATION:", ln=True)
        pdf.ln(2)
        
        # Save image array temporarily for FPDF.
        temp_img_path = os.path.join(BASE_DIR, f"temp_report_{datetime.datetime.now().strftime('%H%M%S')}.jpg")
        cv2.imwrite(temp_img_path, image_array)
        
        # Keep the image aspect ratio stable inside the PDF page.
        h_px, w_px = image_array.shape[:2]
        aspect = h_px / w_px
        pdf_w = 120
        pdf_h = pdf_w * aspect
        
        # Limit image height so the table still fits below.
        if pdf_h > 100:
            pdf_h = 100
            pdf_w = pdf_h / aspect
            
        x_offset = (210 - pdf_w) / 2
        
        pdf.image(temp_img_path, x=x_offset, w=pdf_w, h=pdf_h)
        pdf.ln(pdf_h + 5)
        
        # Remove the temporary image file.
        try:
            os.remove(temp_img_path)
        except Exception:
            pass
    # =======================================================
    
    # Defect detection table.
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(200, 6, txt="DETECTION RESULTS (HITL Validated):", ln=True)
    
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(30, 8, "Source", 1)
    pdf.cell(25, 8, "Defect", 1)
    pdf.cell(20, 8, "Conf(%)", 1)
    pdf.cell(35, 8, "Dimension (px)", 1)
    pdf.cell(35, 8, "Dimension (mm)", 1)
    pdf.cell(40, 8, "Area (mm2)", 1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 9)
    if len(df_final) == 0:
        pdf.cell(185, 8, "No defect indication found", 1, ln=True, align='C')
    else:
        for idx, row in df_final.iterrows():
            pdf.cell(30, 8, str(row['Sumber']), 1)
            pdf.cell(25, 8, str(row['Tipe Cacat']), 1)
            pdf.cell(20, 8, f"{row['Conf (%)']:.0f}", 1)
            pdf.cell(35, 8, f"{row['Dimensi (px)']:.2f}", 1)
            pdf.cell(35, 8, f"{row['Dimensi (mm)']:.2f}", 1)
            pdf.cell(40, 8, f"{row['Luas (mm²)']:.2f}", 1)
            pdf.ln()
    pdf.ln(8)
    
    # Final verdict and notes.
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(200, 6, txt="FINAL DECISION (AWS D1.1):", ln=True)
    pdf.set_font("Arial", 'B', 14)
    if "ACCEPT" in status_akhir:
        pdf.set_text_color(0, 150, 0)
    else:
        pdf.set_text_color(255, 0, 0)
    pdf.cell(200, 10, txt=f"EVALUATION STATUS : {status_akhir}", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 6, txt=f"Notes/Justification :\n{catatan if catatan else '-'}")
    
    return pdf.output(dest='S').encode('latin1')

# ==========================================
# HALAMAN: HOME
# ==========================================
if menu == "Home":
    render_page_header(
        "EUREKA",
        "Expert Utility for Radiographic Evaluation, Knowledge, and Analysis.",
        "Industrial Radiography DSS",
    )

    current_file = st.session_state.get('current_filename', "No image loaded")
    calibration_status = "Applied" if has_valid_calibration() else "Pending"
    home_col1, home_col2, home_col3 = st.columns(3)
    with home_col1:
        render_kpi_card("Active Image", current_file, "Current radiographic inspection target")
    with home_col2:
        render_kpi_card("Saved Inspections", len(st.session_state['history_inspeksi']), "Stored in this Streamlit session")
    with home_col3:
        render_kpi_card("Digital Calibration", calibration_status, "Required before metric verdict")
    
    st.markdown("### Start New Inspection")
    uploaded_file = st.file_uploader("Upload Radiographic Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.get('current_filename'):
            st.session_state['uploaded_image'] = uploaded_file
            st.session_state['current_filename'] = uploaded_file.name 
            st.session_state['df_defects'] = pd.DataFrame(columns=KOLOM_TABEL)
            st.session_state['ai_ran'] = False
            st.session_state['calibration_done'] = False
            st.session_state['px_to_mm_ratio'] = None
            st.session_state['jumlah_gambar_manual'] = 0
            if "ai_results" in st.session_state: del st.session_state['ai_results']
            if "editor_cacat" in st.session_state: del st.session_state["editor_cacat"]
        st.success("Image uploaded successfully. Go to **Single Inspection** to continue.")

    st.markdown("---")
    st.markdown("### Inspection History")
    if len(st.session_state['history_inspeksi']) == 0:
        st.info("No inspections have been saved in this session.")
    else:
        df_hist = pd.DataFrame(st.session_state['history_inspeksi'])
        st.dataframe(
            df_hist[["Nama File", "Tanggal", "Status", "Detail"]].rename(
                columns={"Nama File": "File Name", "Tanggal": "Date", "Detail": "Details"}
            ),
            use_container_width=True,
        )

# ==========================================
# HALAMAN: SINGLE INSPECTION (HITL MODULE)
# ==========================================
elif menu == "Single Inspection":
    render_page_header(
        "Single Inspection",
        "Run AI segmentation, calibrate pixel-to-millimeter scale, validate findings, and issue an AWS D1.1 decision.",
        "Radiographic Assessment",
    )
    
    if 'uploaded_image' not in st.session_state or st.session_state['uploaded_image'] is None:
        st.warning("Please upload an image from the Home page first.")
    else:
        file_bytes = np.asarray(bytearray(st.session_state['uploaded_image'].getvalue()), dtype=np.uint8)
        img_cv2 = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        image_h, image_w = img_rgb.shape[:2]
        defect_count = len(st.session_state['df_defects'])
        inspection_col1, inspection_col2, inspection_col3, inspection_col4 = st.columns(4)
        with inspection_col1:
            render_kpi_card("Active File", st.session_state.get('current_filename', "Unknown"), "Loaded image")
        with inspection_col2:
            render_kpi_card("Image Size", f"{image_w} x {image_h} px", "Native resolution")
        with inspection_col3:
            calibration_text = f"1 px = {st.session_state['px_to_mm_ratio']:.4f} mm" if has_valid_calibration() else "Pending"
            render_kpi_card("Calibration", calibration_text, "Digital ruler status")
        with inspection_col4:
            render_kpi_card("Tracked Indications", defect_count, "AI and manual entries")
        
        if st.button("Run AI Auto Annotation", type="primary", use_container_width=True):
            if "editor_cacat" in st.session_state: del st.session_state["editor_cacat"]
            st.session_state['jumlah_gambar_manual'] = 0
                
            with st.spinner('Scanning weld discontinuities...'):
                img_pred = img_rgb.copy()
                
                # CLAHE preprocessing (Contrast Limited Adaptive Histogram Equalization)
                if gunakan_clahe:
                    lab = cv2.cvtColor(img_pred, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    merged = cv2.merge((cl, a, b))
                    img_pred = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

                # YOLOv8 inference
                results = model.predict(img_pred, conf=conf_thresh) 
                st.session_state['ai_results'] = results[0] 
                
                df_list = []
                
                if len(results[0].boxes) > 0:
                    for i, box in enumerate(results[0].boxes):
                        cls_id = int(box.cls[0].item())
                        nama = model.names[cls_id]
                        konfidensi = float(box.conf[0].item()) * 100 
                        w = box.xywh[0][2].item()
                        h = box.xywh[0][3].item()
                        
                        dimensi_px = float(max(w, h))
                        luas_px = float(w * h)
                        
                        df_list.append(make_defect_row(f"AI-{i+1}", nama, konfidensi, dimensi_px, luas_px))
                            
                df_detected = pd.DataFrame(df_list) if len(df_list) > 0 else pd.DataFrame(columns=KOLOM_TABEL)
                if has_valid_calibration():
                    df_detected = apply_calibration_to_defects(df_detected, st.session_state['px_to_mm_ratio'], batas_luas_maks)

                st.session_state['df_defects'] = df_detected
                st.session_state['ai_ran'] = True
                st.rerun()
        
        st.markdown("---")
        col1, col2 = st.columns([6, 5])
        
        with col1:
            st.subheader("Radiograph Spatial Viewer")
            tampilkan_overlay = st.toggle("Show AI Overlay", value=True)
            mode_alat = st.radio(
                "Canvas Operation Mode:",
                ["Navigation (View Only)", "Digital Ruler Calibration", "Manual Polygon Annotation"],
                horizontal=True,
            )
            
            d_mode = "transform"
            color = "#000000"
            
            if "Calibration" in mode_alat:
                d_mode = "line"
                color = "#00FF00"
            elif "Annotation" in mode_alat:
                d_mode = "polygon"
                color = "#FF0000" 
                st.info("Draw a defect polygon. Make sure the polygon is closed, then save it with the button below.")

            img_display = img_rgb.copy()
            
            # Render YOLO masks into OpenCV for a responsive overlay.
            if st.session_state['ai_ran'] and tampilkan_overlay and 'ai_results' in st.session_state:
                overlay = img_display.copy()
                sumber_aktif = st.session_state['df_defects']['Sumber'].tolist() 
                res = st.session_state['ai_results']
                
                if len(res.boxes) > 0:
                    for i, box in enumerate(res.boxes):
                        sumber_id = f"AI-{i+1}"
                        # Keep overlay visibility synchronized with rows retained in the data editor.
                        if sumber_id in sumber_aktif: 
                            cls_id = int(box.cls[0].item())
                            nama = model.names[cls_id]
                            
                            if nama == 'CR': warna = (255, 0, 0)
                            elif nama == 'PO': warna = (0, 0, 255)
                            else: warna = (255, 165, 0)
                            
                            if res.masks is not None:
                                pts = np.int32(res.masks.xy[i])
                                cv2.fillPoly(overlay, [pts], warna)
                            
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), warna, 2)
                            
                            try:
                                conf_val = st.session_state['df_defects'].loc[st.session_state['df_defects']['Sumber'] == sumber_id, 'Conf (%)'].values[0]
                                label_text = f"{sumber_id} ({conf_val:.0f}%)"
                            except:
                                label_text = sumber_id
                                
                            cv2.putText(overlay, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 2)
                
                cv2.addWeighted(overlay, 0.4, img_display, 0.6, 0, img_display)

            h, w = img_rgb.shape[:2]
            kanvas_lebar = 550 
            kanvas_tinggi = int(h * (kanvas_lebar / w))
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=3, stroke_color=color, 
                background_image=Image.fromarray(img_display), 
                width=kanvas_lebar, height=kanvas_tinggi,
                drawing_mode=d_mode, key="canvas_utama", 
            )
            
            faktor_skala = w / kanvas_lebar
            
            # Canvas geometry processing (digital ruler and manual polygon).
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                obj_terakhir = canvas_result.json_data["objects"][-1] 
                
                if obj_terakhir["type"] == "line" and "Calibration" in mode_alat:
                    px_asli = math.sqrt(obj_terakhir["width"]**2 + obj_terakhir["height"]**2) * faktor_skala
                    st.write(f"Extracted spatial distance: **{px_asli:.2f} pixels**")
                    mm_asli = st.number_input("Actual Reference Length (mm):", min_value=0.1, value=15.0)
                    if st.button("Apply Digital Calibration"):
                        if px_asli <= 0:
                            st.error("Calibration line length must be greater than 0 pixels.")
                            st.stop()
                        rasio_baru = mm_asli / px_asli
                        st.session_state['px_to_mm_ratio'] = rasio_baru
                        st.session_state['calibration_done'] = True
                        if not st.session_state['df_defects'].empty:
                            st.session_state['df_defects'] = apply_calibration_to_defects(
                                st.session_state['df_defects'],
                                rasio_baru,
                                batas_luas_maks,
                            )
                        if "editor_cacat" in st.session_state: del st.session_state["editor_cacat"]
                        st.success(f"Calibration applied. Spatial multiplier: 1 px = {rasio_baru:.4f} mm")
                        st.rerun()

            if "Annotation" in mode_alat:
                st.markdown("---")
                if st.button("Add Annotation to Table", type="primary", use_container_width=True):
                    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                        obj_terakhir = canvas_result.json_data["objects"][-1]
                        if obj_terakhir["type"] in ["rect", "polygon", "path"]:
                            w_obj = obj_terakhir.get("width", 0) * obj_terakhir.get("scaleX", 1)
                            h_obj = obj_terakhir.get("height", 0) * obj_terakhir.get("scaleY", 1)
                            px_manual = max(w_obj, h_obj) * faktor_skala
                            luas_px = (w_obj * faktor_skala) * (h_obj * faktor_skala)
                            
                            st.session_state['jumlah_gambar_manual'] += 1
                            
                            baris_baru = pd.DataFrame([
                                make_defect_row(
                                    f"Manual-{st.session_state['jumlah_gambar_manual']}",
                                    "CR",
                                    100.0,
                                    px_manual,
                                    luas_px,
                                )
                            ])
                            st.session_state['df_defects'] = pd.concat([st.session_state['df_defects'], baris_baru], ignore_index=True)
                            if "editor_cacat" in st.session_state: del st.session_state["editor_cacat"]
                            st.rerun()
                        else:
                            st.error("Could not detect a closed polygon. Check the final polygon point and try again.")
                    else:
                        st.error("No discontinuity object has been defined on the inspection canvas.")

        with col2:
            st.subheader("Discontinuity Summary")
            
            if not has_valid_calibration():
                st.info("Metric size and AWS D1.1 verdict will appear after digital calibration is applied.")

            # Interactive data editor
            edited_df = st.data_editor(
                st.session_state['df_defects'],
                column_config={
                    "Sumber": st.column_config.TextColumn("Source", disabled=True),
                    "Tipe Cacat": st.column_config.SelectboxColumn("Defect Type", options=["CR", "PO", "IP"]),
                    "Conf (%)": st.column_config.NumberColumn("Conf (%)", format="%.0f", disabled=True),
                    "Dimensi (px)": st.column_config.NumberColumn("Dim (px)", format="%.1f", disabled=True),
                    "Luas (px²)": None, 
                    "Dimensi (mm)": st.column_config.NumberColumn("Dim (mm)", format="%.2f", disabled=True),
                    "Luas (mm²)": st.column_config.NumberColumn("Area (mm²)", format="%.2f", disabled=True)
                },
                num_rows="dynamic",
                use_container_width=True,
                key="editor_cacat"
            )
            
            if not edited_df.equals(st.session_state['df_defects']):
                st.session_state['df_defects'] = edited_df
                st.rerun()
            
            st.markdown("---")
            st.subheader("Autonomous Acceptance Engine AWS D1.1")
            tebal_las = st.number_input("Actual Weld Thickness (S) in mm:", min_value=1.0, value=10.0, step=1.0)
            
            status_akhir = None

            # AWS D1.1 Clause 8.12 threshold formulation.
            batas_po = min(tebal_las / 3.0, 6.0)
            batas_ip = tebal_las / 3.0
            st.write(f"*(Tolerance Threshold: PO max {batas_po:.2f} mm, IP max {batas_ip:.2f} mm)*")
            
            cr_reject, ip_reject, po_reject = False, False, False
            alasan_reject = []

            metrics_ready = has_valid_calibration() and (
                edited_df.empty or edited_df["Dimensi (mm)"].notna().all()
            )

            if not metrics_ready:
                render_status_pill("Calibration Pending", "warning")
                st.warning("Rule Engine: pending digital calibration. No ACCEPT/REJECT verdict is generated yet.")
            else:
                for idx, row in edited_df.iterrows():
                    tipe = row['Tipe Cacat']
                    dim_mm = row['Dimensi (mm)'] if not pd.isna(row['Dimensi (mm)']) else 0.0

                    if tipe == 'CR': cr_reject = True
                    elif tipe == 'IP' and dim_mm > batas_ip: ip_reject = True
                    elif tipe == 'PO' and dim_mm > batas_po: po_reject = True

                if cr_reject: alasan_reject.append("Crack (zero tolerance / linear indication is not permitted)")
                if ip_reject: alasan_reject.append(f"Incomplete Penetration exceeds the threshold ({batas_ip:.2f} mm)")
                if po_reject: alasan_reject.append(f"Porosity exceeds the threshold ({batas_po:.2f} mm)")

                # Final system decision rendering.
                if len(edited_df) == 0:
                    status_akhir = "ACCEPT"
                    render_status_pill("ACCEPT", "success")
                    st.success("System: **ACCEPT** (No defect discontinuity found)")
                elif cr_reject or ip_reject or po_reject:
                    status_akhir = "REJECT"
                    render_status_pill("REJECT", "danger")
                    st.error("System: **REJECT** (Acceptance criteria violation found)")
                    for alasan in alasan_reject: st.write(f"- {alasan}")
                else:
                    status_akhir = "ACCEPT"
                    render_status_pill("ACCEPT", "success")
                    st.success("System: **ACCEPT** (Detected indications are within tolerance)")
            
            st.markdown("---")
            if st.button("Save Inspection Result", type="primary", disabled=status_akhir is None):
                waktu_sekarang = datetime.datetime.now()
                ringkasan = ", ".join([f"{row['Tipe Cacat']}({row['Dimensi (mm)']:.1f}mm)" for i, row in edited_df.iterrows() if not pd.isna(row['Dimensi (mm)'])])
                
                # Capture the current inspection visualization (original image + AI/manual overlay).
                img_to_save_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
                
                # Blend manual canvas marks into the saved report image when present.
                if canvas_result.image_data is not None:
                    h_orig, w_orig = img_to_save_bgr.shape[:2]
                    canvas_rgba = canvas_result.image_data
                    canvas_resized = cv2.resize(canvas_rgba, (w_orig, h_orig))
                    alpha_mask = canvas_resized[:, :, 3] > 0
                    # Convert canvas RGBA colors into OpenCV BGR format.
                    img_to_save_bgr[alpha_mask] = canvas_resized[alpha_mask, 2::-1] 
                
                st.session_state['history_inspeksi'].append({
                    "Nama File": st.session_state['current_filename'],
                    "Tanggal": waktu_sekarang.strftime("%d/%m/%Y - %H:%M"),
                    "Status": status_akhir,
                    "Detail": ringkasan if ringkasan else "No Defect",
                    "df_raw": edited_df.copy(), 
                    "tebal_las": tebal_las,
                    "rasio": st.session_state['px_to_mm_ratio'],
                    "image_array": img_to_save_bgr
                })
                st.success("Inspection data saved. Open Report Generator to create the formal PDF document.")

# ==========================================
# HALAMAN: REPORT GENERATOR
# ==========================================
elif menu == "Report Generator":
    render_page_header(
        "Report Generator",
        "Prepare a formal NDT report from a saved inspection record, including calibration, verdict, notes, and visual evidence.",
        "Documentation",
    )
    
    if len(st.session_state['history_inspeksi']) == 0:
        st.warning("The inspection archive is empty. Complete a Single Inspection first to activate this feature.")
    else:
        opsi_inspeksi = [f"{item['Nama File']} | {item['Tanggal']}" for item in st.session_state['history_inspeksi']]
        pilihan = st.selectbox("Select Inspection Archive:", opsi_inspeksi)
        
        idx_pilihan = opsi_inspeksi.index(pilihan)
        data_pilihan = st.session_state['history_inspeksi'][idx_pilihan]
        
        report_col1, report_col2, report_col3 = st.columns(3)
        with report_col1:
            render_kpi_card("Selected File", data_pilihan['Nama File'], "Inspection archive")
        with report_col2:
            render_kpi_card("Evaluation Verdict", data_pilihan['Status'], "AWS D1.1 decision")
        with report_col3:
            render_kpi_card("Detection Details", data_pilihan['Detail'], "Saved summary")
        st.markdown("---")
        
        st.subheader("Inspection Administrative Data")
        
        colA, colB = st.columns(2)
        with colA:
            inspektor = st.text_input("Responsible Inspector Name:", value="Bambang Sri Handoko K. R.")
            perusahaan = st.text_input("Executing Institution/Company:", value="Politeknik Teknologi Nuklir Indonesia")
            client = st.text_input("Client Entity (Optional):", value="")
        with colB:
            tanggal_uji = st.date_input("Registration Date:", value=datetime.date.today())
            lokasi = st.text_input("Assessment Location:", value="KSE A Baiquni Yogyakarta")
            catatan = st.text_area("Technical Justification / Additional Notes:")
            
        if st.button("Export PDF Report", type="primary"):
            tgl_str = tanggal_uji.strftime("%d %B %Y")
            
            citra_visual = data_pilihan.get('image_array', None) 
            
            pdf_bytes = generate_pdf_report(
                nama_file=data_pilihan['Nama File'],
                tgl_uji=tgl_str,
                lokasi=lokasi,
                inspektor=inspektor,
                perusahaan=perusahaan,
                client=client,
                tebal_las=data_pilihan['tebal_las'],
                status_akhir=data_pilihan['Status'],
                df_final=data_pilihan['df_raw'],
                catatan=catatan,
                rasio=data_pilihan['rasio'],
                image_array=citra_visual
            )
            
            st.success("PDF file generated successfully.")
            st.download_button(
                label="Download Exported Document",
                data=pdf_bytes,
                file_name=f"NDT_Report_{data_pilihan['Nama File']}.pdf",
                mime="application/pdf"
            )

# ==========================================
# HALAMAN: BATCH INSPECTION
# ==========================================
elif menu == "Batch Inspection":
    render_page_header(
        "Batch Inspection",
        "Process multiple radiographic images with one global calibration ratio and weld thickness setting.",
        "Batch Processing",
    )
    
    st.markdown("### 1. Global Execution Parameters")
    colA, colB = st.columns(2)
    with colA:
        default_ratio = st.session_state.get('px_to_mm_ratio') or 0.100
        global_ratio = st.number_input("Spatial Conversion Ratio (mm/pixel):", min_value=0.001, value=float(default_ratio), step=0.01, format="%.4f")
    with colB:
        tebal_las_batch = st.number_input("Uniform Weld Thickness (mm):", min_value=1.0, value=10.0, step=1.0)
    
    batas_po_batch = min(tebal_las_batch / 3.0, 6.0)
    batas_ip_batch = tebal_las_batch / 3.0
    st.info(f"Global tolerance basis: **PO max {batas_po_batch:.2f} mm** | **IP max {batas_ip_batch:.2f} mm**.")
    
    st.markdown("### 2. Batch Data Acquisition")
    uploaded_files = st.file_uploader("Import multiple radiographic images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"The system is ready to process **{len(uploaded_files)}** files.")
        
        if st.button("Start Batch Processing", type="primary"):
            batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            output_dir = os.path.join(BASE_DIR, "Eureka_Batch_Output")
            os.makedirs(output_dir, exist_ok=True)
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Running pipeline for image {idx+1}/{len(uploaded_files)}: {file.name}")
                
                file_bytes = np.asarray(bytearray(file.getvalue()), dtype=np.uint8)
                img_cv2 = cv2.imdecode(file_bytes, 1)
                img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                
                # CLAHE preprocessing from the global AI parameter.
                if gunakan_clahe:
                    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    merged = cv2.merge((cl, a, b))
                    img_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

                # AI inference using the global confidence parameter from the sidebar.
                results = model.predict(img_rgb, conf=conf_thresh, verbose=False)
                
                ada_cr = False
                max_ip_mm = 0.0
                max_po_mm = 0.0
                detail_cacat = []
                
                # Maximum area filtering in Batch Mode.
                boxes_valid = []
                if len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        w = box.xywh[0][2].item()
                        h = box.xywh[0][3].item()
                        luas_mm2 = (w * h) * (global_ratio**2)
                        
                        if luas_mm2 <= batas_luas_maks:
                            boxes_valid.append(box)
                            cls_id = int(box.cls[0].item())
                            nama_kelas = model.names[cls_id]
                            panjang_mm = max(w, h) * global_ratio
                            
                            detail_cacat.append(nama_kelas)
                            
                            if nama_kelas == 'CR':
                                ada_cr = True
                            elif nama_kelas == 'IP' and panjang_mm > max_ip_mm:
                                max_ip_mm = panjang_mm
                            elif nama_kelas == 'PO' and panjang_mm > max_po_mm:
                                max_po_mm = panjang_mm
                
                # Placeholder for rendering only valid detections in a future iteration.
                if len(boxes_valid) > 0:
                     pass
                     
                res_plotted = results[0].plot() 
                
                img_bgr_to_save = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)
                timestamp = datetime.datetime.now().strftime("%H%M%S")
                nama_file_output = f"Analyzed_{timestamp}_{file.name}"
                path_simpan = os.path.join(output_dir, nama_file_output)
                cv2.imwrite(path_simpan, img_bgr_to_save)
                            
                if ada_cr or (max_ip_mm > batas_ip_batch) or (max_po_mm > batas_po_batch):
                    status_akhir = "REJECT"
                else:
                    status_akhir = "ACCEPT"
                
                unik_cacat = list(set(detail_cacat))
                ringkasan_str = ", ".join(unik_cacat) if unik_cacat else "No Defect"
                
                batch_results.append({
                    "Nama File": file.name,
                    "Deteksi": ringkasan_str,
                    "CR": "Yes" if ada_cr else "No",
                    "Max IP (mm)": f"{max_ip_mm:.2f}",
                    "Max PO (mm)": f"{max_po_mm:.2f}",
                    "Vonis": status_akhir
                })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text(f"Batch processing completed. Visual output saved to: {os.path.abspath(output_dir)}")
            
            st.markdown("### Batch Data Recap")
            batch_display = pd.DataFrame(batch_results).rename(
                columns={"Nama File": "File Name", "Deteksi": "Detection", "Vonis": "Verdict"}
            )
            st.dataframe(batch_display, use_container_width=True)
            
            for res in batch_results:
                st.session_state['history_inspeksi'].append({
                    "Nama File": res["Nama File"],
                    "Tanggal": datetime.datetime.now().strftime("%d/%m/%Y - %H:%M"),
                    "Status": res["Vonis"],
                    "Detail": f"Batch: {res['Deteksi']} (PO={res['Max PO (mm)']}, IP={res['Max IP (mm)']})",
                    "df_raw": pd.DataFrame(), 
                    "tebal_las": tebal_las_batch,
                    "rasio": global_ratio
                })
            st.success("Batch recap table synchronized to the Home archive.")
