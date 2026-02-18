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

# --- CSS CUSTOM PER STILE ---
st.markdown("""
<style>
    .css-18e3th9 { padding-top: 0rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- FUNZIONI DI ELABORAZIONE ---

def load_image(uploaded_file):
    """Carica immagine e gestisce orientamento/formato"""
    try:
        image = Image.open(uploaded_file)
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
        except:
            pass
        img_np = np.asarray(image)
        if len(img_np.shape) == 2:
            return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:
            return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Errore file {uploaded_file.name}: {e}")
        return None

def fractal_dimension(Z):
    """Calcolo Frattale Box-Counting"""
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
    """Ritaglio intelligente muffin + calcolo larghezza base per calibrazione"""
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
    
    # Taglio Coda (Opzionale, semplificato)
    # Qui usiamo il bounding rect diretto per robustezza visiva
    coords_final = cv2.findNonZero(mask_main)
    if coords_final is not None:
        x_new, y_new, w_new, h_new = cv2.boundingRect(coords_final)
    else:
        x_new, y_new, w_new, h_new = x, y, w, h
        
    pad = 10
    h_img, w_img = img.shape[:2]
    y1, y2 = max(0, y_new-pad), min(h_img, y_new+h_new+pad)
    x1, x2 = max(0, x_new-pad), min(w_img, x_new+w_new+pad)
    
    # Calcolo Base (ultime 20 righe del mask) per calibrazione
    base_slice = mask_main[y_new+h_new-20 : y_new+h_new-5, x_new:x_new+w_new]
    if base_slice.shape[0] > 0:
        base_widths = np.sum(base_slice > 0, axis=1)
        base_w_px = np.max(base_widths) if len(base_widths) > 0 else w_new
    else:
        base_w_px = w_new

    return img[y1:y2, x1:x2], mask_main[y1:y2, x1:x2], base_w_px

# --- SIDEBAR CONFIGURAZIONE ---
st.sidebar.header("🎛️ Pannello Controllo")

st.sidebar.subheader("1. Calibrazione")
DIAMETRO_BASE_MM = st.sidebar.number_input("Larghezza Base Reale (mm)", value=50.0, step=0.5, help="Misura del pirottino/base")

st.sidebar.subheader("2. Parametri Rilevamento")
with st.sidebar.expander("🛠️ Regola HSV e Filtri", expanded=False):
    col_h1, col_h2 = st.columns(2)
    h_min = col_h1.slider("H Min", 0, 180, 0)
    h_max = col_h2.slider("H Max", 0, 180, 40)
    s_min = col_h1.slider("S Min", 0, 255, 30)
    v_min = col_h1.slider("V Min", 0, 255, 60)
    
    st.markdown("---")
    LBP_RADIUS = st.number_input("LBP Radius", 3)
    MIN_PORE_AREA = st.number_input("Area Min Poro (px)", 20)
    MASK_EROSION = st.number_input("Erosione Bordi (px)", 15)

LOWER_HSV = np.array([h_min, s_min, v_min])
UPPER_HSV = np.array([h_max, 255, 255])
LBP_POINTS = 8 * LBP_RADIUS

# --- MAIN PAGE ---
st.title("🧁 Muffin Lab: Hybrid Edition")
st.markdown("Analisi scientifica avanzata con visualizzazione grafica semplificata.")

uploaded_files = st.file_uploader("📂 Carica immagini (JPG, PNG, HEIC)", accept_multiple_files=True)

if uploaded_files:
    # Contenitore per i risultati Excel
    results_list = []
    
    # Creiamo due tab principali: Uno per vedere le immagini, uno per i dati grezzi
    tab_visual, tab_data = st.tabs(["👁️ Visualizzazione & Report", "📊 Tabella Dati"])

    with tab_visual:
        for uploaded_file in uploaded_files:
            # Layout a "Card" per ogni immagine
            with st.container():
                st.markdown(f"### 📄 {uploaded_file.name}")
                
                # Load
                img = load_image(uploaded_file)
                if img is None: continue
                
                # Resize (Performance)
                scale_factor = 1000 / max(img.shape[:2])
                if scale_factor < 1:
                    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
                
                # Process
                crop_img, crop_mask, base_w_px = smart_crop(img, LOWER_HSV, UPPER_HSV)
                
                if crop_img is not None:
                    # Calcoli Scientifici
                    ppm = base_w_px / DIAMETRO_BASE_MM
                    ppm_area = ppm**2
                    
                    h_mm = crop_img.shape[0] / ppm
                    w_mm = crop_img.shape[1] / ppm
                    
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    
                    # Maschera sicura ed elaborazione
                    kernel_erode = np.ones((int(MASK_EROSION), int(MASK_EROSION)), np.uint8)
                    mask_safe = cv2.erode(crop_mask, kernel_erode, iterations=1) > 0
                    
                    # Pori
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    gray_enhanced = clahe.apply(gray)
                    thresh = cv2.adaptiveThreshold(gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
                    pores = (thresh > 0) & mask_safe
                    pores = morphology.remove_small_objects(pores, min_size=5)
                    pores_labeled = measure.label(pores)
                    props = measure.regionprops(pores_labeled)
                    
                    # Metriche
                    porosity = (np.sum(pores) / np.sum(mask_safe)) * 100 if np.sum(mask_safe) > 0 else 0
                    fd = fractal_dimension(pores)
                    
                    # Spessore Pareti
                    dough = (~pores) & mask_safe
                    dt = ndimage.distance_transform_edt(dough)
                    wall_th = (np.mean(dt[dough]) * 2) / ppm if np.any(dough) else 0
                    
                    # Weibull
                    areas_mm = [p.area / ppm_area for p in props if p.area >= MIN_PORE_AREA]
                    w_alpha, w_beta = 0, 0
                    if len(areas_mm) > 20:
                        try: w_alpha, _, w_beta = weibull_min.fit(areas_mm, floc=0)
                        except: pass

                    # --- VISUALIZZAZIONE "HYBRID" (STILE SCRIPT 1 + DATI SCRIPT 2) ---
                    
                    # 1. Riga Metriche (Grandi Numeri)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Porosità", f"{porosity:.1f} %", help="Percentuale di vuoti vs impasto")
                    m2.metric("Spessore Pareti", f"{wall_th:.2f} mm", help="Spessore medio delle pareti dell'impasto")
                    m3.metric("Frattale (FD)", f"{fd:.3f}", help="Complessità della struttura (Box Counting)")
                    m4.metric("Weibull α", f"{w_alpha:.2f}", delta="Regolare" if w_alpha > 1.5 else "Irregolare", help="Omogeneità alveolatura (>1.5 è buono)")
                    
                    # 2. Riga Immagini (Originale - Overlay Rosso - Mappa Calore)
                    col_img1, col_img2, col_img3 = st.columns(3)
                    
                    with col_img1:
                        st.image(crop_img, channels="BGR", caption=f"Crop (H: {h_mm:.1f}mm)")
                    
                    with col_img2:
                        # Creazione Overlay Rosso (Stile Script 1)
                        overlay = color.label2rgb(pores_labeled, image=gray, bg_label=0, colors=['red'], alpha=0.3)
                        st.image(overlay, caption="Mappa Pori Rilevati", clamp=True)
                        
                    with col_img3:
                        # Mappa Calore Spessori (Stile Script 2 ma pulito)
                        fig_dt, ax_dt = plt.subplots()
                        im = ax_dt.imshow(dt, cmap='jet')
                        ax_dt.axis('off')
                        st.pyplot(fig_dt)
                        st.caption("Mappa Spessori Pareti")
                        plt.close(fig_dt)

                    # 3. Grafico Weibull (Sotto le immagini)
                    if w_alpha > 0:
                        fig_w, ax_w = plt.subplots(figsize=(10, 3))
                        ax_w.hist(areas_mm, bins=40, density=True, alpha=0.5, color='gray', label='Dati')
                        x_w = np.linspace(min(areas_mm), max(areas_mm), 200)
                        ax_w.plot(x_w, weibull_min.pdf(x_w, w_alpha, scale=w_beta), 'r-', lw=2, label=f'Weibull (α={w_alpha:.2f})')
                        ax_w.set_title("Distribuzione Dimensione Pori")
                        ax_w.legend()
                        st.pyplot(fig_w)
                        plt.close(fig_w)
                    
                    # Raccolta dati per Excel
                    results_list.append({
                        "File": uploaded_file.name,
                        "Porosità (%)": porosity,
                        "Frattale (FD)": fd,
                        "Spessore Pareti (mm)": wall_th,
                        "Weibull Alpha": w_alpha,
                        "Altezza (mm)": h_mm,
                        "Larghezza (mm)": w_mm
                    })
                    
                    st.divider()
                else:
                    st.warning("Muffin non rilevato. Regola i parametri HSV nella sidebar.")

    # TAB DATI
    with tab_data:
        if results_list:
            df = pd.DataFrame(results_list)
            st.dataframe(df, use_container_width=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            
            st.download_button("📥 Scarica Excel Completo", output.getvalue(), "Muffin_Analysis.xlsx", "application/vnd.ms-excel")