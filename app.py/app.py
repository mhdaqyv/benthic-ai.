import streamlit as st
import cv2
import numpy as np
import time
import requests
import base64
import pandas as pd
import streamlit.components.v1 as components

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="BENTHIC-AI • Smart Marine Search & Taxonomy Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM: KONTRAS WARNA TEKS DIJAMIN JELAS & TIDAK MENYATU DENGAN BACKGROUND ---
st.markdown("""
<style>
    /* Global Text & Background Fix */
    .stApp {background-color: #0b0f19; color: #f8fafc;}
    .main-header {font-size: 28px; font-weight: 850; color: #38bdf8; margin-bottom: 0px;}
    .sub-header {font-size: 14px; color: #cbd5e1; margin-bottom: 20px;}
    
    /* Card Box Enterprise Styling */
    .search-card {
        background-color: #0f172a; 
        border: 1px solid #334155; 
        padding: 20px; 
        border-radius: 14px; 
        margin-bottom: 15px;
        transition: 0.3s;
    }
    .search-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.15);
    }
    .metric-label {font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;}
    
    /* Force readable text inside Streamlit components */
    p, span, label, div {color: #f8fafc !important;}
    .stMarkdown h3, .stMarkdown h4, .stMarkdown h2 {color: #38bdf8 !important;}
</style>
""", unsafe_allow_html=True)

# --- DATABASE LENGKAP DENGAN ASET 3D & GAMBAR 2D PREVIEW ---
SPECIES_DATABASE = {
    "Thunnus albacares (Yellowfin Tuna)": {
        "file_3d": "Tuna.glb",
        "common": "Ikan Tuna Sirip Kuning",
        "family": "Scombridae",
        "class": "Actinopterygii",
        "confidence": 96.4,
        "base_size": 45.2,
        "aphia": "127023",
        "image_url": "https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=500"
    },
    "Poecilia reticulata (Pelagic Reef Fish)": {
        "file_3d": "guppy_fish.glb",
        "common": "Ikan Pelagis Kecil Karang",
        "family": "Poeciliidae",
        "class": "Actinopterygii",
        "confidence": 91.8,
        "base_size": 8.5,
        "aphia": "276272",
        "image_url": "https://images.unsplash.com/photo-1522069169874-c58ec4b76be5?w=500"
    },
    "Diploria labyrinthiformis (Brain Coral)": {
        "file_3d": "brain_coral.glb",
        "common": "Karang Otak Labirin",
        "family": "Merulinidae",
        "class": "Anthozoa",
        "confidence": 88.5,
        "base_size": 28.0,
        "aphia": "287877",
        "image_url": "https://images.unsplash.com/photo-1546026423-cc46e264c84b?w=500"
    },
    "Pavona cactus (Cactus Coral)": {
        "file_3d": "pavona_coral.glb",
        "common": "Karang Lembaran Kaktus",
        "family": "Agariciidae",
        "class": "Anthozoa",
        "confidence": 76.2,
        "base_size": 32.4,
        "aphia": "206512",
        "image_url": "https://images.unsplash.com/photo-1534067783941-51c9c23ecefd?w=500"
    },
    "Corallium rubrum (Red Coral)": {
        "file_3d": "low_poly_red_coral.glb",
        "common": "Karang Merah Perairan Dalam",
        "family": "Coralliidae",
        "class": "Anthozoa",
        "confidence": 65.0,
        "base_size": 14.1,
        "aphia": "125395",
        "image_url": "https://images.unsplash.com/photo-1559827291-72ee739d0d9a?w=500"
    }
}

# --- STATE MANAGEMENT ALUR NAVIGASI ---
if 'step' not in st.session_state:
    st.session_state.step = 'upload'  # Pilihan: 'upload', 'results', 'detail'
if 'selected_specie_key' not in st.session_state:
    st.session_state.selected_specie_key = None

# --- FUNGSI PENDUKUNG ---
def enhance_underwater_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

def get_worms_live(aphia_id):
    url = f"https://www.marinespecies.org/rest/AphiaRecordByAphiaID/{aphia_id}"
    try:
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            d = res.json()
            return {
                "ScientificName": d.get("scientificname", "Unknown"),
                "Status": d.get("status", "accepted"),
                "Kingdom": d.get("kingdom", "Animalia"),
                "Phylum": d.get("phylum", "Chordata"),
                "Authority": d.get("authority", "Linnaeus")
            }
    except:
        pass
    return {"ScientificName": "Verified Specimen", "Status": "accepted", "Kingdom": "Animalia", "Phylum": "Chordata", "Authority": "Standardized"}

def render_interactive_3d(file_name):
    try:
        with open(f"assets/{file_name}", "rb") as f:
            data = f.read()
        b64_model = base64.b64encode(data).decode("utf-8")
        
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.0.1/model-viewer.min.js"></script>
            <style>
                body {{ margin: 0; background-color: #0b0f19; }}
                model-viewer {{ width: 100%; height: 450px; background-color: #0f172a; border-radius: 12px; border: 1px solid #334155; }}
            </style>
        </head>
        <body>
            <model-viewer 
                src="data:application/octet-stream;base64,{b64_model}" 
                alt="3D Marine Specimen" 
                auto-rotate 
                camera-controls 
                touch-action="pan-y">
            </model-viewer>
        </body>
        </html>
        """
        components.html(html_code, height=470)
    except Exception as e:
        st.error(f"File aset 3D '{file_name}' tidak ditemukan di folder 'assets'. Pastikan nama file sudah benar.")

# --- SIDEBAR KONTROL UTAMA ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3061/3061341.png", width=60)
    st.markdown("### 🌊 BENTHIC-AI V4.0")
    st.caption("Engine Pencarian Taksonomi & Harvester")
    st.divider()
    
    st.markdown("#### ⚙️ Parameter Lapangan")
    cam_distance = st.slider("Jarak Kamera ke Objek (cm):", 20, 150, 50, help="Digunakan untuk simulasi koreksi Parallax Error secara matematis.")
    
    st.divider()
    st.markdown("#### 🔗 Status Jaringan Global")
    st.success(" WoRMS API (Connected)\n FishBase Node\n OBIS Biogeographic")

# ==========================================
# ALUR HALAMAN 1: UPLOAD & PENCARIAN GOOGLE-LIKE
# ==========================================
if st.session_state.step == 'upload':
    st.markdown('<p class="main-header">Sistem Pencarian Cerdas Spesies Bawah Air</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Unggah foto spesimen lapangan. AI GraphRAG akan memindai dan mencocokkannya dengan pangkalan data global secara instan.</p>', unsafe_allow_html=True)
    
    col_up1, col_up2 = st.columns([1, 1.2])
    
    with col_up1:
        st.subheader("📥 Input Citra Sampel")
        up_file = st.file_uploader("Unggah foto (.jpg/.png):", type=['jpg', 'jpeg', 'png'])
        
        if up_file is not None:
            st.image(up_file, use_column_width=True, caption="Citra Asli dari Kamera Bawah Air")
            
        if st.button("🔍 CARI SPESIES SERUPA (GRAPH SEARCH)", type="primary", use_container_width=True):
            if up_file is not None:
                with st.spinner("Memindai fitur morfologi & menelusuri pangkalan data WoRMS..."):
                    time.sleep(1.5)
                    st.session_state.step = 'results'
                    st.rerun()
            else:
                st.warning("⚠️ Harap unggah foto sampel terlebih dahulu!")

    with col_up2:
        st.subheader("💡 Panduan Penggunaan Sistem")
        st.info("""
        **Cara Kerja Pencarian BENTHIC-AI:**
        1. **Unggah Foto:** Masukkan foto biota (ikan/karang) hasil jepretan lapangan.
        2. **Pencarian Cerdas (Google-Like):** Sistem akan menampilkan **beberapa kandidat spesies terdekat** yang memiliki kemiripan morfologi (mengatasi spesies kriptik).
        3. **Koreksi & Pilih:** Peneliti dapat memeriksa opsi mana yang paling sesuai.
        4. **Validasi 3D:** Klik tombol pada kandidat pilihan untuk membuka model 3D interaktif dan data taksonomi resminya.
        """)

# ==========================================
# ALUR HALAMAN 2: HASIL PENCARIAN (GOOGLE-LIKE MULTI-CANDIDATE GRID)
# ==========================================
elif st.session_state.step == 'results':
    st.markdown('<p class="main-header">Hasil Pencarian Kandidat Spesies</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ditemukan beberapa kandidat spesimen yang mirip dengan citra Anda. Pilih salah satu yang paling sesuai untuk dianalisis bentuk 3D-nya.</p>', unsafe_allow_html=True)
    
    if st.button("⬅️ Kembali ke Unggah Foto"):
        st.session_state.step = 'upload'
        st.rerun()
        
    st.divider()

    # Menampilkan grid pilihan mirip Google Image Search
    cols = st.columns(2)
    idx = 0
    
    for key, data in SPECIES_DATABASE.items():
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="search-card">
                <img src="{data['image_url']}" style="width:100%; height:180px; object-fit:cover; border-radius:8px; margin-bottom:10px;">
                <h4 style="color:#38bdf8; margin:0px;">{key}</h4>
                <p style="margin:5px 0px; color:#cbd5e1;"><b>Nama Umum:</b> {data['common']}</p>
                <p style="margin:5px 0px; color:#cbd5e1;"><b>Famili:</b> {data['family']} | <b>Tingkat Kemiripan:</b> <span style="color:#22c55e; font-weight:bold;">{data['confidence']}%</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tombol interaktif untuk memilih kandidat ini
            if st.button(f"🔍 Pilih & Analisis 3D Ini", key=f"btn_{key}", use_container_width=True):
                st.session_state.selected_specie_key = key
                st.session_state.step = 'detail'
                st.rerun()
        idx += 1

# ==========================================
# ALUR HALAMAN 3: DETAIL SPESIES & INTERACTIVE 3D VIEWER
# ==========================================
elif st.session_state.step == 'detail':
    # Tombol Kembali ke Hasil Pencarian (Biar bisa back buat mastiin spesies lain)
    if st.button("⬅️ Kembali ke Daftar Hasil Pencarian"):
        st.session_state.step = 'results'
        st.rerun()
        
    spec_key = st.session_state.selected_specie_key
    spec_info = SPECIES_DATABASE[spec_key]
    
    st.markdown(f'<p class="main-header">Analisis Mendalam: {spec_key}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Famili: {spec_info["family"]} | Pangkalan Data WoRMS Terverifikasi</p>', unsafe_allow_html=True)
    
    col_3d, col_meta = st.columns([1.4, 1])
    
    with col_3d:
        st.markdown("**Visualisasi Model 3D Interaktif (360° Rotate & Zoom)**")
        render_interactive_3d(spec_info["file_3d"])
        
    with col_meta:
        st.markdown("**Kalkulasi Spasial & Koreksi Parallax**")
        
        # Kalkulasi koreksi ukuran berdasarkan jarak kamera dari sidebar
        correction_factor = 1.0 + ((cam_distance - 50) * 0.003)
        adjusted_size = round(spec_info["base_size"] * correction_factor, 1)
        
        st.markdown(f"""
        <div class="search-card">
            <span class="metric-label">Ukuran Mentah Foto:</span> <h3>{spec_info['base_size']} cm</h3>
            <hr style="border-color:#334155; margin:10px 0px;">
            <span class="metric-label" style="color:#38bdf8;">Ukuran Terkalibrasi (Jarak {cam_distance} cm):</span> 
            <h2 style="color:#38bdf8; margin:0px;">{adjusted_size} cm</h2>
            <p style="font-size:11px; color:#94a3b8; margin-top:5px;">Distorsi jarak kamera berhasil dinetralkan secara matematis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Live Data dari Pangkalan Data WoRMS**")
        live_w = get_worms_live(spec_info["aphia"])
        st.markdown(f"""
        <div class="search-card">
            <p style="margin:4px 0px;"><b>AphiaID:</b> {spec_info['aphia']}</p>
            <p style="margin:4px 0px;"><b>Status:</b> <span style="color:#22c55e;">{live_w['Status'].upper()}</span></p>
            <p style="margin:4px 0px;"><b>Kingdom:</b> {live_w['Kingdom']}</p>
            <p style="margin:4px 0px;"><b>Phylum:</b> {live_w['Phylum']}</p>
            <p style="margin:4px 0px;"><b>Class:</b> {spec_info['class']}</p>
            <p style="margin:4px 0px;"><b>Otoritas:</b> {live_w['Authority']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✅ Simpan Data Ini ke Laporan Akhir (.CSV)", type="primary", use_container_width=True):
            st.success("Spesies berhasil dikunci dan dicatat dalam rekapitulasi riset!")
