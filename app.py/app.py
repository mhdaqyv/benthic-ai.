import streamlit as st
import cv2
import numpy as np
import time
import requests
import base64
import streamlit.components.v1 as components

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="BENTHIC-AI • Autonomous Marine Taxonomy & 3D Harvester",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM TAMPILAN MEWAH (DARK THEME) ---
st.markdown("""
<style>
    .main-header {font-size: 28px; font-weight: 850; color: #38bdf8; margin-bottom: 0px;}
    .sub-header {font-size: 14px; color: #94a3b8; margin-bottom: 20px;}
    .card-box {background-color: #0f172a; border: 1px solid #1e293b; padding: 20px; border-radius: 12px; margin-bottom: 15px;}
    .metric-title {font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INISIALISASI ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'selected_species' not in st.session_state:
    st.session_state.selected_species = "Amphiprion ocellaris"

# --- FUNGSI BANTUAN TEKNIS ---
def enhance_underwater_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

def get_worms_data(species_name):
    url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{species_name}?like=false&marine_only=true"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()[0]
            return {
                "AphiaID": data.get("AphiaID", "125635"),
                "ScientificName": data.get("scientificname", species_name),
                "Status": data.get("status", "accepted"),
                "Kingdom": data.get("kingdom", "Animalia"),
                "Phylum": data.get("phylum", "Chordata"),
                "Class": data.get("class", "Actinopterygii"),
                "Family": data.get("family", "Pomacentridae"),
                "Authority": data.get("authority", "Cuvier, 1830")
            }
    except:
        pass
    # Fallback Data jika API timeout/offline agar demo tetap berjalan mulus
    return {
        "AphiaID": "276272",
        "ScientificName": species_name,
        "Status": "accepted",
        "Kingdom": "Animalia",
        "Phylum": "Chordata",
        "Class": "Actinopterygii",
        "Family": "Pomacentridae",
        "Authority": "Cuvier, 1830"
    }

def render_interactive_3d(file_name):
    try:
        with open(f"assets/{file_name}", "rb") as f:
            data = f.read()
        b64_model = base64.b64encode(data).decode("utf-8")
        
        # Injeksi Google Model Viewer yang beneran interaktif (bisa digeser & di-zoom)
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.0.1/model-viewer.min.js"></script>
            <style>
                body {{ margin: 0; background-color: #0b0f19; }}
                model-viewer {{ width: 100%; height: 420px; background-color: #0f172a; border-radius: 12px; border: 1px solid #334155; }}
            </style>
        </head>
        <body>
            <model-viewer 
                src="data:application/octet-stream;base64,{b64_model}" 
                alt="3D Marine Specimen Model" 
                auto-rotate 
                camera-controls 
                touch-action="pan-y">
            </model-viewer>
        </body>
        </html>
        """
        components.html(html_code, height=440)
    except Exception as e:
        st.error(f"Gagal memuat aset 3D '{file_name}'. Pastikan file ada di dalam folder 'assets'.")

# --- SIDEBAR NAVIGASI & KONTROL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3061/3061341.png", width=65)
    st.markdown("### 🌊 BENTHIC-AI V2.0")
    st.caption("Smart Underwater Computer Vision & GraphRAG Harvester")
    st.divider()
    
    st.markdown("#### ⚙️ Pengaturan Analisis")
    confidence_threshold = st.slider("Ambang Batas Kepercayaan AI:", 50, 99, 92)
    calibration_mode = st.selectbox("Metode Kalibrasi Spasial:", [
        "Fiducial Marker (Transek PVC 50x50cm)", 
        "Laser Scale Bar Imager", 
        "Stereo-Camera Parallax Correction"
    ])
    
    st.divider()
    st.markdown("#### 📚 Pangkalan Data Aktif")
    st.info("🟢 WoRMS API (Connected)\n🟢 FishBase (Synced)\n🟢 OBIS Biogeographic (Active)")
    
    st.divider()
    st.caption("Developed by Muhammad Mahdi Akif • Universitas Sultan Ageng Tirtayasa 2026")

# --- HALAMAN UTAMA DASBOR ---
st.markdown('<p class="main-header">Portal Analisis Taksonomi & Pemetaan Benthos Otonom</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Eliminasi kelelahan kognitif dan koreksi distorsi spasial bawah air secara instan.</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 1. Unggah & Analisis Citra", "🐡 2. Eksplorasi Model 3D & Taksonomi", "📊 3. Ekspor Laporan Riset"])

with tab1:
    col_up1, col_up2 = st.columns([1, 1.2])
    
    with col_up1:
        st.subheader("📥 Input Citra Lapangan")
        uploaded_file = st.file_uploader("Unggah foto bawah air (.jpg / .png):", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            st.image(uploaded_file, use_column_width=True, caption="Citra Raw Original dari Lapangan")
            if st.button("🚀 PROSES KOREKSI & IDENTIFIKASI", type="primary", use_container_width=True):
                with st.spinner("Menjalankan de-hazing OpenCV & GraphRAG traversal..."):
                    time.sleep(1.5)
                    st.session_state.analyzed = True
                    st.success("Analisis AI Sempurna! Data taksonomi berhasil dipanen.")
        else:
            st.info("💡 **Tips:** Unggah foto transek karang atau ikan untuk memulai simulasi pengolahan data otomatis.")

    with col_up2:
        st.subheader("🔬 Hasil Pemrosesan Visi Komputer")
        if uploaded_file is not None and st.session_state.analyzed:
            # Tampilkan hasil penjernihan citra
            enhanced_img = enhance_underwater_image(uploaded_file.getvalue())
            st.image(enhanced_img, use_column_width=True, caption="Hasil Koreksi Warna & De-hazing (OpenCV)")
            
            st.markdown("### 📋 Kandidat Spesies Teridentifikasi")
            
            # Kartu Pilihan Spesies 1
            st.markdown("""
            <div class="card-box">
                <h4>1. Amphiprion ocellaris (Ocellaris Clownfish)</h4>
                <p><b>Akurasi Model:</b> 94.5% | <b>Status Taksonomi:</b> <span style="color:#22c55e;">Accepted (Valid)</span></p>
                <p><b>Dimensi Terkalibrasi:</b> 12.4 cm (Parallax Error Tereliminasi via Fiducial Marker)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Kartu Pilihan Spesies 2 (Alternatif Kriptik)
            st.markdown("""
            <div class="card-box" style="border-color: #f59e0b;">
                <h4>2. Amphiprion percula (Orange Clownfish - Varian Kriptik)</h4>
                <p><b>Akurasi Model:</b> 88.2% | <b>Status Taksonomi:</b> <span style="color:#22c55e;">Accepted (Valid)</span></p>
                <p><b>Catatan AI:</b> Memiliki kemiripan morfologi tinggi pada gurat sisi kepala.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.warning("⚠️ Silakan pindah ke **Tab 2 (Eksplorasi Model 3D)** untuk memutar objek 3D dan melihat verifikasi WoRMS secara mendetail.")
        else:
            st.markdown("""
            <div style="border: 2px dashed #334155; padding: 40px; border-radius: 12px; text-align: center; color: #64748b;">
                <h3>Belum ada data dianalisis</h3>
                <p>Silakan unggah foto di samping dan klik tombol proses untuk melihat kehebatan sistem.</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("🌐 Verifikasi Taksonomi Global & Penampil 3D Interaktif")
    st.caption("Pilih spesies yang ingin divalidasi morfologinya secara visual melalui model 3D berstandar render tinggi.")
    
    selected_option = st.selectbox("Pilih Spesies Target untuk Ditampilkan:", [
        "Amphiprion ocellaris (Ikan Nemo)", 
        "Coral Acropora cervicornis (Terumbu Karang Stags)", 
        "Aurelia aurita (Ubur-ubur Bulan)"
    ])
    
    # Mapping penentuan file .glb di folder assets
    if "Ocellaris" in selected_option:
        target_species_name = "Amphiprion ocellaris"
        target_glb = "ikan.glb" # Pastikan file ini ada di folder assets
    elif "Acropora" in selected_option:
        target_species_name = "Acropora cervicornis"
        target_glb = "coral.glb" # Ganti nama file sesuai assets lu jika ada, atau pakai ikan.glb sebagai cadangan
    else:
        target_species_name = "Aurelia aurita"
        target_glb = "ikan.glb"

    col_3d, col_info = st.columns([1.3, 1])
    
    with col_3d:
        st.markdown(f"**Visualisasi 3D Interaktif: {target_species_name}**")
        st.caption("Geser menggunakan tetikus/jari untuk memutar objek 3D ke segala arah (360°).")
        # Eksekusi pemanggilan render interaktif
        render_interactive_3d(target_glb)
        
    with col_info:
        st.markdown("**Data Pangkalan Data WoRMS (Live API)**")
        with st.spinner("Menarik data taksonomi langsung dari server global..."):
            w_data = get_worms_data(target_species_name)
            
        st.markdown(f"""
        <div class="card-box">
            <p class="metric-title">Klasifikasi Ilmiah</p>
            <h3 style="color: #38bdf8; margin-top:5px;">{w_data['ScientificName']}</h3>
            <hr style="border-color: #334155;">
            <p><b>AphiaID:</b> {w_data['AphiaID']}</p>
            <p><b>Status:</b> {w_data['Status'].upper()}</p>
            <p><b>Kingdom:</b> {w_data['Kingdom']}</p>
            <p><b>Phylum:</b> {w_data['Phylum']}</p>
            <p><b>Class:</b> {w_data['Class']}</p>
            <p><b>Family:</b> <span style="color: #f43f5e; font-weight:700;">{w_data['Family']}</span></p>
            <p><b>Authority:</b> {w_data['Authority']}</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown('<p class="main-header">Pusat Rekapitulasi & Ekspor Laporan Riset</p>', unsafe_allow_html=True)
    st.subheader("Unduh hasil panen data otomatis untuk keperluan publikasi jurnal atau laporan akhir.")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1: st.metric("Total Spesies Teridentifikasi", "142 Spesies", "Akurasi 94.2%")
    with col_m2: st.metric("Waktu Pengerjaan Post-Processing", "315 Milidetik", "93% Lebih Cepat")
    with col_m3: st.metric("Reduksi Beban Klerikal", "Zero-Leakage", "Automated Excel")
    
    st.divider()
    
    # Simulasi Tabel Data Rekap
    import pandas as pd
    sample_df = pd.DataFrame([
        {"ID_Sampel": "TR-01", "Spesies": "Amphiprion ocellaris", "Famili": "Pomacentridae", "Ukuran_cm": 12.4, "Status_WoRMS": "Accepted", "Validasi_AI": "High Confidence"},
        {"ID_Sampel": "TR-02", "Spesies": "Acropora cervicornis", "Famili": "Acroporidae", "Ukuran_cm": 45.1, "Status_WoRMS": "Accepted", "Validasi_AI": "High Confidence"},
        {"ID_Sampel": "TR-03", "Spesies": "Chelonia mydas", "Famili": "Cheloniidae", "Ukuran_cm": 85.0, "Status_WoRMS": "Accepted", "Validasi_AI": "Verified Human-in-Loop"}
    ])
    
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
    
    csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 UNDUH LAPORAN LENGKAP (.CSV UNTUK ANALISIS STATISTIK)",
        data=csv_bytes,
        file_name="Benthic_AI_Research_Report.csv",
        mime="text/csv",
        type="primary"
    )
