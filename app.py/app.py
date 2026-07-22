import streamlit as st
import cv2
import numpy as np
import time
import requests
import base64
import streamlit.components.v1 as components

# Konfigurasi Halaman Dasar
st.set_page_config(page_title="BENTHIC-AI | Data Harvester", page_icon="🌊", layout="wide")

# Inisialisasi State Management (Untuk pindah-pindah halaman tanpa refresh)
if 'page' not in st.session_state:
    st.session_state.page = 'beranda'
if 'selected_species' not in st.session_state:
    st.session_state.selected_species = ""

# --- FUNGSI BANTUAN ---

# 1. Fungsi Penjernihan Bawah Air (De-hazing Simulasi)
def enhance_underwater_image(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Simple Auto-Contrast / Histogram Equalization untuk RGB
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

# 2. Fungsi Tarik Data Asli dari API WoRMS
def get_worms_data(species_name):
    url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{species_name}?like=false&marine_only=true"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()[0]
            return {
                "AphiaID": data.get("AphiaID", "N/A"),
                "Status": data.get("status", "N/A"),
                "Kingdom": data.get("kingdom", "N/A"),
                "Family": data.get("family", "N/A"),
                "Authority": data.get("authority", "N/A")
            }
        else:
            return None
    except:
        return None

# 3. Fungsi Render 3D Model Viewer (HTML Injection)
def render_3d_model(file_name):
    try:
        # Ubah path sesuaikan dengan nama file .glb lu di folder assets!
        with open(f"assets/{file_name}", "rb") as f:
            data = f.read()
        b64_model = base64.b64encode(data).decode("utf-8")
        
        # HTML + Google Model Viewer
        html_code = f"""
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.0.1/model-viewer.min.js"></script>
        <model-viewer 
            src="data:application/octet-stream;base64,{b64_model}" 
            auto-rotate 
            camera-controls 
            style="width: 100%; height: 400px; background-color: #f0f4f8; border-radius: 10px;">
        </model-viewer>
        """
        components.html(html_code, height=450)
    except Exception as e:
        st.error(f"Gagal memuat 3D Model: Pastikan ada file '{file_name}' di folder 'assets'.")


# --- ANTARMUKA HALAMAN BERANDA (UPLOAD) ---
if st.session_state.page == 'beranda':
    st.title("🌊 BENTHIC-AI: Smart Data Harvester")
    st.markdown("Unggah foto transek bawah air Anda. Sistem akan melakukan penjernihan gambar, kalibrasi ukuran, dan identifikasi spesies secara otonom.")
    
    uploaded_file = st.file_uploader("Upload Foto Sampel (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Foto Asli (Raw)")
            st.image(uploaded_file, use_column_width=True)
            
        with col2:
            st.subheader("Pre-Processing (De-hazing)")
            # Memproses gambar dengan OpenCV
            enhanced_img = enhance_underwater_image(uploaded_file.getvalue())
            st.image(enhanced_img, use_column_width=True)
            
        st.write("---")
        if st.button("Mulai Identifikasi GraphRAG 🔍", use_container_width=True):
            with st.spinner("Mendeteksi Fiducial Marker dan melakukan kueri semantik..."):
                time.sleep(2) # Simulasi loading AI
                st.success("Identifikasi Selesai! Ditemukan Spesies Kriptik.")
                
                # Simulasi Hasil Deteksi
                st.markdown("### 📊 Kandidat Spesies Terdeteksi")
                st.info("⚠️ Peringatan Human-in-the-Loop: Spesies memiliki kemiripan morfologi tinggi. Mohon verifikasi.")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Kandidat Utama", "Amphiprion ocellaris", "Confidence: 94%")
                c2.metric("Dimensi Terkalibrasi", "12.4 cm", "Parallax Error: Eliminated")
                c3.metric("Kandidat Alternatif", "Amphiprion percula", "Confidence: 89%")
                
                # Tombol untuk ke halaman 3D (Ganti 'ikan.glb' dengan nama file asli lu di folder assets)
                if st.button("Lihat Model 3D & Validasi WoRMS (A. ocellaris) 🐡", type="primary"):
                    st.session_state.selected_species = "Amphiprion ocellaris"
                    st.session_state.file_3d = "ikan.glb" # <-- GANTI NAMA INI NANTI
                    st.session_state.page = 'detail'
                    st.rerun()

# --- ANTARMUKA HALAMAN DETAIL (3D & API WORMS) ---
elif st.session_state.page == 'detail':
    st.button("⬅️ Kembali ke Beranda", on_click=lambda: st.session_state.update(page='beranda'))
    
    st.title(f"Verifikasi Spesies: {st.session_state.selected_species}")
    
    colA, colB = st.columns([1.5, 1])
    
    with colA:
        st.subheader("Visualisasi 3D Interaktif")
        st.markdown("*Putar (drag) dan zoom untuk membandingkan fitur morfologi dengan foto sampel Anda.*")
        render_3d_model(st.session_state.file_3d)
        
    with colB:
        st.subheader("Validasi Pangkalan Data Global")
        with st.spinner("Mengambil data live dari API WoRMS..."):
            worms_info = get_worms_data(st.session_state.selected_species)
            
            if worms_info:
                st.success("Tervalidasi secara taksonomi! ✅")
                st.write(f"**AphiaID:** {worms_info['AphiaID']}")
                st.write(f"**Status:** {worms_info['Status']}")
                st.write(f"**Kingdom:** {worms_info['Kingdom']}")
                st.write(f"**Famili:** {worms_info['Family']}")
                st.write(f"**Otoritas:** {worms_info['Authority']}")
            else:
                st.error("Gagal terhubung ke API WoRMS atau spesies tidak ditemukan.")
                
        st.write("---")
        st.subheader("Export Data")
        csv_data = "Species,Dimensi_cm,Status,Lat,Long\nAmphiprion ocellaris,12.4,accepted,-6.12,106.8"
        st.download_button(label="Unduh Rekap Laporan (CSV) 📥", data=csv_data, file_name="benthic_report.csv", mime="text/csv", use_container_width=True)