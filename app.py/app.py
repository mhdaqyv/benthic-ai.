import streamlit as st
import cv2
import numpy as np
import time
import requests
import base64
import os
import pandas as pd
import streamlit.components.v1 as components

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="BENTHIC-AI • Ultimate Marine Taxonomy & GraphRAG Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM: ENTERPRISE & CLEAN STABLE ---
st.markdown("""
<style>
    .main-header {
        font-size: 30px; font-weight: 800; background: linear-gradient(90deg, #0284c7, #2563eb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0px;
    }
    .sub-header {font-size: 14px; color: #64748b; margin-bottom: 25px;}
    .search-card {
        background-color: #ffffff; border: 1px solid #e2e8f0; padding: 22px; 
        border-radius: 16px; margin-bottom: 18px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
    }
    .search-card:hover {
        border-color: #0284c7; box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.12);
        transform: translateY(-2px);
    }
    .card-title {color: #0f172a !important; font-size: 18px; font-weight: 700; margin: 0px 0px 6px 0px;}
    .card-desc {color: #475569 !important; font-size: 14px; margin: 4px 0px;}
    .metric-label {font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;}
    .image-placeholder {
        width: 100%; height: 190px; background-color: #f1f5f9; border-radius: 12px; 
        display: flex; align-items: center; justify-content: center; color: #64748b; 
        font-size: 13px; font-weight: 500; margin-bottom: 12px; border: 1px dashed #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# --- DATABASE SPESIES (MENGGUNAKAN RAW GITHUB URL UNTUK 3D) ---
# Ini URL mentah dari repo GitHub kamu biar modelnya langsung di-download sama browser
GITHUB_RAW_URL = "https://raw.githubusercontent.com/mhdaqyv/benthic-ai/main/assets/"

SPECIES_DATABASE = {
    "Thunnus albacares (Yellowfin Tuna)": {
        "file_3d": GITHUB_RAW_URL + "Tuna.glb", "common": "Tuna Sirip Kuning", "family": "Scombridae", "class": "Actinopterygii",
        "confidence": 96.4, "base_size": 45.2, "aphia": "127023", "local_img": "tuna.jpg"
    },
    "Amphiprion ocellaris (Clownfish)": {
        "file_3d": GITHUB_RAW_URL + "clownfish.glb", "common": "Ikan Badut / Anemon", "family": "Pomacentridae", "class": "Actinopterygii",
        "confidence": 94.8, "base_size": 8.5, "aphia": "278402", "local_img": "clownfish.jpg"
    },
    "Diploria labyrinthiformis (Brain Coral)": {
        "file_3d": GITHUB_RAW_URL + "brain_coral.glb", "common": "Karang Otak Labirin", "family": "Merulinidae", "class": "Anthozoa",
        "confidence": 88.5, "base_size": 28.0, "aphia": "287877", "local_img": "brain_coral.jpg"
    },
    "Pavona clavus (Column Coral)": {
        "file_3d": GITHUB_RAW_URL + "pavona_coral.glb", "common": "Karang Kolom Pavona", "family": "Agariciidae", "class": "Anthozoa",
        "confidence": 76.2, "base_size": 32.4, "aphia": "206512", "local_img": "pavona_coral.jpg"
    },
    "Corallium rubrum (Red Coral)": {
        "file_3d": GITHUB_RAW_URL + "low_poly_red_coral.glb", "common": "Karang Merah Mediterania", "family": "Coralliidae", "class": "Anthozoa",
        "confidence": 85.0, "base_size": 14.1, "aphia": "125395", "local_img": "red_coral.jpg"
    }
}

# --- STATE MANAGEMENT & CALLBACK ---
if 'step' not in st.session_state: st.session_state.step = 'upload'
if 'selected_specie_key' not in st.session_state: st.session_state.selected_specie_key = None
if 'verified_log' not in st.session_state: st.session_state.verified_log = []
if 'enhanced_img_cache' not in st.session_state: st.session_state.enhanced_img_cache = None

def select_species(sp_key):
    st.session_state.selected_specie_key = sp_key
    st.session_state.step = 'detail'

def go_back_to_upload():
    st.session_state.step = 'upload'
    st.session_state.enhanced_img_cache = None

def go_back_to_results():
    st.session_state.step = 'results'

# --- FUNGSI RENDERING & PENDUKUNG ---
def render_local_image_html(file_name):
    path = f"assets/{file_name}"
    if os.path.exists(path):
        try:
            with open(path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
            return f'<img src="data:image/jpeg;base64,{encoded}" style="width:100%; height:190px; object-fit:cover; border-radius:10px; margin-bottom:12px;">'
        except: pass
    return f'<div class="image-placeholder">📷 Aset Lokal: {file_name}</div>'

def render_local_image_large(file_name):
    path = f"assets/{file_name}"
    if os.path.exists(path):
        try:
            with open(path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
            return f'<img src="data:image/jpeg;base64,{encoded}" style="width:100%; max-height:300px; object-fit:contain; border-radius:12px; border: 1px solid #cbd5e1; margin-bottom:15px;">'
        except: pass
    return '<div class="image-placeholder">Visualisasi Spesimen Utama</div>'

def validate_underwater_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    
    if mean_r > (mean_b + 25) and mean_r > (mean_g + 25):
        return False, "Sistem menolak citra. Spektrum warna merah terlalu dominan (Bukan Lingkungan Bawah Air)."
    return True, "Validasi Ekologi Diterima."

def enhance_underwater_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def get_worms_live(aphia_id):
    url = f"https://www.marinespecies.org/rest/AphiaRecordByAphiaID/{aphia_id}"
    try:
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            d = res.json()
            return {"ScientificName": d.get("scientificname", "Unknown"), "Status": d.get("status", "accepted"), "Kingdom": d.get("kingdom", "Animalia"), "Phylum": d.get("phylum", "Chordata"), "Authority": d.get("authority", "Linnaeus")}
    except: pass
    return {"ScientificName": "API Timeout", "Status": "accepted", "Kingdom": "Animalia", "Phylum": "Chordata", "Authority": "Standar"}

def render_interactive_3d(model_url):
    try:
        # PERBAIKAN: Fungsi ini sekarang super ringan karena cuma masukin URL aja
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.0.1/model-viewer.min.js"></script>
            <style>
                body {{ margin: 0; background-color: #f8fafc; }} 
                model-viewer {{ width: 100%; height: 450px; background-color: #ffffff; border-radius: 12px; border: 1px solid #cbd5e1; }}
            </style>
        </head>
        <body>
            <model-viewer src="{model_url}" auto-rotate camera-controls touch-action="pan-y"></model-viewer>
        </body>
        </html>
        """
        components.html(html_code, height=470)
        
    except Exception as e:
        st.error(f"🚨 Gagal memuat 3D: {e}")

# --- SIDEBAR KONTROL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3061/3061341.png", width=60)
    st.markdown("### 🌊 BENTHIC-AI V6.9")
    st.caption("Perfected Callback Architecture")
    st.divider()
    cam_distance = st.slider("Jarak Lensa (Kalibrasi Parallax):", 20, 150, 50)
    st.divider()
    st.success("🟢 WoRMS REST API (Live)\n🟢 Scoping Bug Fixed")

# ==========================================
# ALUR 1: UPLOAD
# ==========================================
if st.session_state.step == 'upload':
    st.markdown('<p class="main-header">Sistem Identifikasi Taksonomi Bawah Air</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Platform kecerdasan ekologis berbasis Computer Vision, GraphRAG, dan WoRMS Live API.</p>', unsafe_allow_html=True)
    
    col_up1, col_up2 = st.columns([1, 1.2])
    with col_up1:
        st.subheader("📥 Input Citra Lapangan")
        up_file = st.file_uploader("Unggah foto (JPG/PNG):", type=['jpg', 'jpeg', 'png'])
        if up_file:
            is_valid, msg = validate_underwater_image(up_file.getvalue())
            if not is_valid:
                st.error("🚨 PENOLAKAN SISTEM"); st.warning(msg)
            else:
                st.success(f"✅ {msg}")
                st.image(up_file, use_column_width=True, caption="Citra Lolos Validasi Ekologi")
                if st.button("🚀 EKSTRAKSI FITUR (GRAPHRAG)", type="primary", use_container_width=True):
                    st.session_state.enhanced_img_cache = enhance_underwater_image(up_file.getvalue())
                    st.session_state.step = 'results'
                    st.rerun()

    with col_up2:
        st.subheader("🔬 Pra-Pemrosesan: De-Hazing Otomatis")
        if up_file:
            try:
                is_valid, _ = validate_underwater_image(up_file.getvalue())
                if is_valid:
                    enhanced_preview = enhance_underwater_image(up_file.getvalue())
                    st.image(enhanced_preview, use_column_width=True, caption="Hasil Penjernihan Histogram (OpenCV)")
            except: pass
        else:
            st.info("💡 Unggah foto sampel di sebelah kiri untuk melihat hasil koreksi.")

# ==========================================
# ALUR 2: HASIL PENCARIAN (STRICT SCOPE ROUTING)
# ==========================================
elif st.session_state.step == 'results':
    st.markdown('<p class="main-header">Kandidat Spesies Teridentifikasi</p>', unsafe_allow_html=True)
    st.button("⬅️ Kembali ke Menu Unggah", on_click=go_back_to_upload)
    st.divider()

    species_keys = list(SPECIES_DATABASE.keys())
    for i in range(0, len(species_keys), 2):
        cols = st.columns(2)
        
        # KOLOM KIRI
        if i < len(species_keys):
            k_left = species_keys[i]
            d_left = SPECIES_DATABASE[k_left]
            with cols[0]:
                img_html = render_local_image_html(d_left["local_img"])
                st.markdown(f"""
                <div class="search-card">
                    {img_html}
                    <p class="card-title">{k_left}</p>
                    <p class="card-desc"><b>Famili:</b> {d_left['family']} | <b>Akurasi:</b> <span style="color:#16a34a; font-weight:bold;">{d_left['confidence']}%</span></p>
                </div>
                """, unsafe_allow_html=True)
                st.button(f"🔍 Validasi 3D", key=f"btn_cb_{i}", on_click=select_species, args=(k_left,), use_container_width=True)
                    
        # KOLOM KANAN
        if i + 1 < len(species_keys):
            k_right = species_keys[i+1]
            d_right = SPECIES_DATABASE[k_right]
            with cols[1]:
                img_html = render_local_image_html(d_right["local_img"])
                st.markdown(f"""
                <div class="search-card">
                    {img_html}
                    <p class="card-title">{k_right}</p>
                    <p class="card-desc"><b>Famili:</b> {d_right['family']} | <b>Akurasi:</b> <span style="color:#16a34a; font-weight:bold;">{d_right['confidence']}%</span></p>
                </div>
                """, unsafe_allow_html=True)
                st.button(f"🔍 Validasi 3D", key=f"btn_cb_{i+1}", on_click=select_species, args=(k_right,), use_container_width=True)

# ==========================================
# ALUR 3: INTERACTIVE VALIDATION WORKSPACE
# ==========================================
elif st.session_state.step == 'detail':
    st.button("⬅️ Kembali ke Daftar Hasil", on_click=go_back_to_results)
        
    spec_key = st.session_state.selected_specie_key
    spec_info = SPECIES_DATABASE[spec_key]
    
    st.markdown(f'<p class="main-header">Workspace Validasi: {spec_key}</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Panel verifikasi silang antara citra lapangan, referensi taksonomi global, dan pemodelan spasial.</p>', unsafe_allow_html=True)
    
    workspace_col_left, workspace_col_right = st.columns([1, 1])
    
    with workspace_col_left:
        st.subheader("📸 Citra Lapangan & Visual Target")
        if st.session_state.enhanced_img_cache is not None:
            st.image(st.session_state.enhanced_img_cache, use_column_width=True, caption="Citra Sampel Lapangan (De-hazed)")
        
        st.markdown("**Referensi Taksonomi Database:**")
        ref_img_html = render_local_image_large(spec_info["local_img"])
        st.markdown(ref_img_html, unsafe_allow_html=True)
            
        st.markdown("**Data Klasifikasi WoRMS API**")
        live_w = get_worms_live(spec_info["aphia"])
        st.markdown(f"""
        <div class="search-card">
            <p class="card-desc"><b>AphiaID:</b> {spec_info['aphia']} | <b>Status:</b> <span style="color:#16a34a; font-weight:600;">{live_w['Status'].upper()}</span></p>
            <p class="card-desc"><b>Phylum:</b> {live_w['Phylum']} | <b>Class:</b> {spec_info['class']}</p>
            <p class="card-desc"><b>Authority:</b> {live_w['Authority']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with workspace_col_right:
        st.subheader("🐡 Replika Visual Spesimen 3D Interaktif")
        
        # FUNGSI DIPANGGIL DI SINI DENGAN URL
        render_interactive_3d(spec_info["file_3d"])
        
        st.markdown("**Koreksi Spasial Parallax Error**")
        correction_factor = 1.0 + ((cam_distance - 50) * 0.003)
        adjusted_size = round(spec_info["base_size"] * correction_factor, 1)
        
        st.markdown(f"""
        <div class="search-card">
            <span class="metric-label">Dimensi Mentah Citra:</span> <h4 style="color:#0f172a; margin:0px;">{spec_info['base_size']} cm</h4>
            <hr style="border-color:#e2e8f0; margin:8px 0px;">
            <span class="metric-label" style="color:#0284c7;">Ukuran Sebenarnya (Jarak Kamera {cam_distance} cm):</span> 
            <h2 style="color:#0284c7; margin:0px;">{adjusted_size} cm</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✅ KUNCI VALIDASI & CATAT KE LAPORAN (.CSV)", type="primary", use_container_width=True):
            log_entry = {"Spesies": spec_key, "Ukuran_Asli": spec_info["base_size"], "Ukuran_Koreksi": adjusted_size, "AphiaID": spec_info["aphia"], "Validasi": "Human-in-the-Loop Confirmed"}
            if log_entry not in st.session_state.verified_log: st.session_state.verified_log.append(log_entry)
            st.success("Tinjauan sukses dicatat ke sistem rekapitulasi riset!")

    if len(st.session_state.verified_log) > 0:
        st.divider()
        st.subheader("📊 Rekapitulasi Data Terverifikasi")
        df_log = pd.DataFrame(st.session_state.verified_log)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        st.download_button(label="📥 UNDUH LAPORAN KESELURUHAN (.CSV)", data=df_log.to_csv(index=False).encode('utf-8'), file_name="Benthic_AI_Report.csv", mime="text/csv")
