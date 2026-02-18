import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
from skimage import feature, measure, morphology, filters, color
from scipy.stats import weibull_min
from scipy import ndimage
import matplotlib.pyplot as plt
import io

# --- CONFIGURAZIONE INIZIALE ---
register_heif_opener()
st.set_page_config(page_title="Muffin Lab Pro", layout="wide", page_icon="🧁")

# --- INIZIALIZZAZIONE STORICO (SESSION STATE) ---
if 'history_data' not in st.session_state:
    st.session_state['history_data'] = []

# --- CSS CUSTOM ---
st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; border: 1px solid #e0e0e0; }
    [data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    h1 { font-size: 2rem !important; }
    .success-msg { color: green; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- FUNZIONI DI ELABORAZIONE ---

def load_image(image_file):
    try:
        image = Image.open(image_file)
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
        except: pass
        
        img_np = np.asarray(image)
        if len(img_np.shape) == 2: return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4: return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else: return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Errore file: {e}")
        return None

def fractal_dimension(Z):
    Z = (Z > 0)
    p = min(Z.shape)
    if p <= 1: return 0
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        h_trim = Z.shape[0] - (Z.shape[0] % size)
        w_trim = Z.shape[1] - (Z.shape[1] % size)
        Z_trim = Z[:h_trim, :w_trim]
        sh = Z_trim.shape
        blocks = Z_trim.reshape((sh[0]//size, size, sh[1]//size, size))
        counts.append(np.sum(blocks.sum(axis=(1, 3)) > 0))
    if len(sizes) < 2: return 0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def smart_crop(img, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None, None
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    mask_main = np.zeros_like(mask)
    cv2.drawContours(mask_main, [c], -1, 255, -1)
    
    coords_final = cv2.findNonZero(mask_main)
    if coords_final is not None:
        x_new, y_new, w_new, h_new = cv2.boundingRect(coords_final)
    else:
        x_new, y_new, w_new, h_new = x, y, w, h
        
    pad = 10
    h_img, w_img = img.shape[:2]
    y1, y2 = max(0, y_new-pad), min(h_img, y_new+h_new+pad)
    x1, x2 = max(0, x_new-pad), min(w_img, x_new+w_new+pad)
    
    base_slice = mask_main[y_new+h_new-25 : y_new+h_new-5, x_new:x_new+w_new]
    if base_slice.shape[0] > 0:
        base_widths = np.sum(base_slice > 0, axis=1)
        base_w_px = np.max(base_widths) if len(base_widths) > 0 else w_new
    else:
        base_w_px = w_new

    return img[y1:y2, x1:x2], mask_main[y1:y2, x1:x2], base_w_px

# --- SIDEBAR ---
st.sidebar.header("🎛️ Configurazione")
DIAMETRO_BASE_MM = st.sidebar.number_input("Larghezza Base Reale (mm)", value=50.0, step=0.5)

with st.sidebar.expander("🛠️ Parametri Avanzati", expanded=False):
    st.write("**HSV**")
    c1, c2 = st.columns(2)
    h_min = c1.slider("H Min", 0, 180, 0)
    h_max = c2.slider("H Max", 0, 180, 40)
    s_min = c1.slider("S Min", 0, 255, 30)
    v_min = c1.slider("V Min", 0, 255, 60)
    st.write("**Analisi**")
    MIN_PORE_AREA = st.number_input("Area Min Poro (px)", 20)
    MASK_EROSION = st.number_input("Erosione (px)", 15)

LOWER_HSV = np.array([h_min, s_min, v_min])
UPPER_HSV = np.array([h_max, 255, 255])

# --- GESTIONE STORICO ---
def add_to_history(batch_results):
    """Aggiunge i risultati correnti allo storico globale"""
    if batch_results:
        # Evita duplicati controllando il nome file
        existing_files = [item['File'] for item in st.session_state['history_data']]
        new_items = [item for item in batch_results if item['File'] not in existing_files]
        
        if new_items:
            st.session_state['history_data'].extend(new_items)
            st.toast(f"✅ {len(new_items)} muffin aggiunti allo storico!", icon="💾")
        else:
            st.toast("⚠️ Questi file sono già nello storico.", icon="ℹ️")

def clear_history():
    st.session_state['history_data'] = []
    st.toast("Storico cancellato.", icon="🗑️")

# --- MAIN PAGE ---
st.title("🔬 Muffin Lab: Analisi immagine")

# Selezione Input
input_method = st.radio("Sorgente:", ("📸 Scatta Foto", "📂 Carica da Galleria"), horizontal=True, label_visibility="collapsed")
image_files = []

if input_method == "📸 Scatta Foto":
    cam_file = st.camera_input("Foto Muffin")
    if cam_file: image_files.append(cam_file)
else:
    uploaded = st.file_uploader("Carica Immagini", type=['png', 'jpg', 'jpeg', 'heic'], accept_multiple_files=True)
    if uploaded: image_files = uploaded

# --- TABS PRINCIPALI ---
tab_analysis, tab_history = st.tabs(["👁️ Analisi Corrente", "🗂️ Storico Sessione (Report)"])

current_batch_results = []

with tab_analysis:
    if image_files:
        st.info("Analisi in corso... Controlla i risultati qui sotto.")
        for img_file in image_files:
            fname = img_file.name if hasattr(img_file, 'name') else "Foto_Camera"
            
            # Load & Resize
            img = load_image(img_file)
            if img is None: continue
            
            scale = 1000 / max(img.shape[:2])
            if scale < 1: img = cv2.resize(img, None, fx=scale, fy=scale)
            
            # Process
            crop_img, crop_mask, base_w_px = smart_crop(img, LOWER_HSV, UPPER_HSV)
            
            if crop_img is not None:
                ppm = base_w_px / DIAMETRO_BASE_MM
                h_mm = crop_img.shape[0] / ppm
                w_mm = crop_img.shape[1] / ppm
                
                # Logic
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                kernel_erode = np.ones((int(MASK_EROSION), int(MASK_EROSION)), np.uint8)
                mask_safe = cv2.erode(crop_mask, kernel_erode, iterations=1) > 0
                
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gray_enhanced = clahe.apply(gray)
                thresh = cv2.adaptiveThreshold(gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
                pores = (thresh > 0) & mask_safe
                pores = morphology.remove_small_objects(pores, min_size=5)
                props = measure.regionprops(measure.label(pores))
                
                porosity = (np.sum(pores) / np.sum(mask_safe)) * 100 if np.sum(mask_safe) > 0 else 0
                fd = fractal_dimension(pores)
                
                dough = (~pores) & mask_safe
                dt = ndimage.distance_transform_edt(dough)
                wall_th = (np.mean(dt[dough]) * 2) / ppm if np.any(dough) else 0
                
                areas_mm = [p.area / (ppm**2) for p in props if p.area >= MIN_PORE_AREA]
                w_alpha, w_beta = 0, 0
                if len(areas_mm) > 20:
                    try: w_alpha, _, w_beta = weibull_min.fit(areas_mm, floc=0)
                    except: pass
                
                # --- LAYOUT CARD ---
                with st.expander(f"📄 Risultati: {fname}", expanded=True):
                    # Metriche
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Porosità", f"{porosity:.1f}%")
                    c2.metric("Spessore", f"{wall_th:.2f} mm")
                    c3.metric("FD", f"{fd:.3f}")
                    c4.metric("Weibull α", f"{w_alpha:.2f}")
                    
                    # Immagini
                    col_i1, col_i2 = st.columns(2)
                    col_i1.image(crop_img, channels="BGR", caption="Originale")
                    overlay = color.label2rgb(measure.label(pores), image=gray, bg_label=0, colors=['red'], alpha=0.3)
                    col_i2.image(overlay, caption="Analisi Pori")
                
                # Append Data
                current_batch_results.append({
                    "File": fname,
                    "Porosità (%)": round(porosity, 2),
                    "Spessore Pareti (mm)": round(wall_th, 2),
                    "Frattale (FD)": round(fd, 3),
                    "Weibull Alpha": round(w_alpha, 2),
                    "Weibull Beta": round(w_beta, 2),
                    "Altezza (mm)": round(h_mm, 1),
                    "Larghezza (mm)": round(w_mm, 1)
                })
            else:
                st.warning(f"Oggetto non trovato in {fname}")

        # --- AZIONE DI SALVATAGGIO ---
        if current_batch_results:
            st.divider()
            col_save, col_info = st.columns([1, 3])
            if col_save.button("➕ Aggiungi Batch allo Storico", type="primary"):
                add_to_history(current_batch_results)
    else:
        st.write("👈 Scatta o carica una foto per iniziare.")

# --- TAB STORICO ---
with tab_history:
    st.subheader("🗂️ Storico Dati della Sessione")
    
    if st.session_state['history_data']:
        df_history = pd.DataFrame(st.session_state['history_data'])
        
        # Mostra tabella
        st.dataframe(df_history, use_container_width=True)
        
        # Statistiche Rapide
        st.markdown("### 📊 Medie Sessione")
        cols_stats = ["Porosità (%)", "Spessore Pareti (mm)", "Frattale (FD)", "Weibull Alpha"]
        st.dataframe(df_history[cols_stats].mean().to_frame().T.style.format("{:.2f}"), hide_index=True)

        # Download
        c_down, c_clear = st.columns([1, 1])
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_history.to_excel(writer, index=False, sheet_name='Storico_Completo')
            
        c_down.download_button(
            label="📥 Scarica Excel Storico Completo",
            data=buffer.getvalue(),
            file_name="Muffin_Lab_History.xlsx",
            mime="application/vnd.ms-excel"
        )
        
        if c_clear.button("🗑️ Cancella Storico"):
            clear_history()
            st.rerun()
            
    else:
        st.info("Lo storico è vuoto. Analizza delle foto e premi 'Aggiungi Batch allo Storico'.")
