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

# --- CSS CUSTOM PER STILE SU MOBILE ---
st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; border: 1px solid #e0e0e0; }
    [data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    h1 { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# --- FUNZIONI DI ELABORAZIONE ---

def load_image(image_file):
    """Carica immagine gestendo vari formati e orientamento EXIF"""
    try:
        image = Image.open(image_file)
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image) # Fondamentale per foto da cellulare
        except:
            pass
        
        img_np = np.asarray(image)
        # Conversione in BGR per OpenCV
        if len(img_np.shape) == 2:
            return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:
            return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Errore file: {e}")
        return None

def fractal_dimension(Z):
    """Calcolo dimensione frattale (Box Counting)"""
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
    
    # Pulizia morfologica
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None, None
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    mask_main = np.zeros_like(mask)
    cv2.drawContours(mask_main, [c], -1, 255, -1)
    
    # Taglio Coda (Opzionale) - Qui usiamo il bounding rect diretto per robustezza
    coords_final = cv2.findNonZero(mask_main)
    if coords_final is not None:
        x_new, y_new, w_new, h_new = cv2.boundingRect(coords_final)
    else:
        x_new, y_new, w_new, h_new = x, y, w, h
        
    pad = 10
    h_img, w_img = img.shape[:2]
    y1, y2 = max(0, y_new-pad), min(h_img, y_new+h_new+pad)
    x1, x2 = max(0, x_new-pad), min(w_img, x_new+w_new+pad)
    
    # Calcolo Base (media ultime righe) per calibrazione
    # Prendiamo una striscia vicino al fondo per stimare la larghezza del pirottino
    base_slice = mask_main[y_new+h_new-25 : y_new+h_new-5, x_new:x_new+w_new]
    if base_slice.shape[0] > 0:
        base_widths = np.sum(base_slice > 0, axis=1)
        base_w_px = np.max(base_widths) if len(base_widths) > 0 else w_new
    else:
        base_w_px = w_new

    return img[y1:y2, x1:x2], mask_main[y1:y2, x1:x2], base_w_px

# --- SIDEBAR: PARAMETRI ---
st.sidebar.header("🎛️ Pannello Controllo")

st.sidebar.subheader("1. Calibrazione")
DIAMETRO_BASE_MM = st.sidebar.number_input(
    "Larghezza Base Reale (mm)", 
    value=50.0, step=0.5, 
    help="Inserisci la larghezza reale del pirottino/base del muffin per calibrare i pixel."
)

st.sidebar.subheader("2. Parametri Avanzati")
with st.sidebar.expander("🛠️ Regola Filtri e Colore", expanded=False):
    st.write("**Range Colore (HSV)**")
    col_h1, col_h2 = st.columns(2)
    h_min = col_h1.slider("H Min", 0, 180, 0)
    h_max = col_h2.slider("H Max", 0, 180, 40)
    s_min = col_h1.slider("S Min", 0, 255, 30)
    v_min = col_h1.slider("V Min", 0, 255, 60)
    
    st.write("**Analisi Texture**")
    MIN_PORE_AREA = st.number_input("Area Min Poro (px)", 20)
    MASK_EROSION = st.number_input("Erosione Bordi (px)", 15)

LOWER_HSV = np.array([h_min, s_min, v_min])
UPPER_HSV = np.array([h_max, 255, 255])

# --- MAIN PAGE ---
st.title("🔬 Muffin Lab Mobile")
st.markdown("Analisi scientifica avanzata. Scegli come caricare l'immagine:")

# --- SELEZIONE INPUT (FOTOCAMERA O FILE) ---
input_method = st.radio(
    "Sorgente Immagine:", 
    ("📸 Scatta Foto", "📂 Carica da Galleria"), 
    horizontal=True,
    label_visibility="collapsed"
)

image_files = []

if input_method == "📸 Scatta Foto":
    cam_file = st.camera_input("Inquadra il muffin (su sfondo contrastante)")
    if cam_file is not None:
        image_files.append(cam_file)
else:
    uploaded_files = st.file_uploader("Scegli immagini", type=['png', 'jpg', 'jpeg', 'heic'], accept_multiple_files=True)
    if uploaded_files:
        image_files = uploaded_files

# --- ELABORAZIONE ---
if image_files:
    results_list = []
    
    # Tabs: Visualizzazione vs Dati
    tab_visual, tab_data = st.tabs(["👁️ Analisi Visiva", "📊 Tabella Dati"])

    with tab_visual:
        for img_file in image_files:
            st.markdown("---")
            # Nome file (gestisce sia upload che camera)
            fname = img_file.name if hasattr(img_file, 'name') else "Foto_Camera.jpg"
            st.subheader(f"📄 {fname}")
            
            # 1. Caricamento
            img = load_image(img_file)
            if img is None: continue
            
            # Resize Intelligente (per non bloccare il telefono)
            # Scala l'immagine in modo che il lato lungo sia max 1000px
            h_orig, w_orig = img.shape[:2]
            max_dim = 1000
            if max(h_orig, w_orig) > max_dim:
                scale = max_dim / max(h_orig, w_orig)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            # 2. Smart Crop & Masking
            crop_img, crop_mask, base_w_px = smart_crop(img, LOWER_HSV, UPPER_HSV)
            
            if crop_img is not None:
                # 3. Calcoli Scientifici (Tutta la logica matematica)
                
                # Calibrazione (Pixel per Millimetro)
                ppm = base_w_px / DIAMETRO_BASE_MM
                ppm_area = ppm**2
                
                # Dimensioni Fisiche
                h_mm = crop_img.shape[0] / ppm
                w_mm = crop_img.shape[1] / ppm
                
                # Pre-processing
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                
                # Maschera Sicura (Erosione per togliere la crosta esterna dall'analisi)
                kernel_erode = np.ones((int(MASK_EROSION), int(MASK_EROSION)), np.uint8)
                mask_safe = cv2.erode(crop_mask, kernel_erode, iterations=1) > 0
                
                # Rilevamento Pori (CLAHE + Adaptive Threshold)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gray_enhanced = clahe.apply(gray)
                thresh = cv2.adaptiveThreshold(gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
                
                pores = (thresh > 0) & mask_safe
                pores = morphology.remove_small_objects(pores, min_size=5)
                pores_labeled = measure.label(pores)
                props = measure.regionprops(pores_labeled)
                
                # Metriche Chiave
                porosity = (np.sum(pores) / np.sum(mask_safe)) * 100 if np.sum(mask_safe) > 0 else 0
                fd = fractal_dimension(pores)
                
                # Spessore Pareti (Distance Transform)
                dough = (~pores) & mask_safe
                dt = ndimage.distance_transform_edt(dough)
                wall_th = (np.mean(dt[dough]) * 2) / ppm if np.any(dough) else 0
                
                # Weibull Analysis
                areas_mm = [p.area / ppm_area for p in props if p.area >= MIN_PORE_AREA]
                w_alpha, w_beta = 0, 0
                if len(areas_mm) > 20:
                    try: 
                        w_alpha, _, w_beta = weibull_min.fit(areas_mm, floc=0)
                    except: pass

                # --- VISUALIZZAZIONE "HYBRID" ---
                
                # A. Le Metriche (Grandi e Chiare)
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Porosità", f"{porosity:.1f}%", help="% Vuoto su Pieno")
                col_m2.metric("Spessore Pareti", f"{wall_th:.2f} mm", help="Spessore medio mollica")
                col_m3.metric("Frattale (FD)", f"{fd:.3f}", help="Complessità geometrica")
                col_m4.metric("Weibull α", f"{w_alpha:.2f}", help=">1.5 = alveolatura omogenea")
                
                # B. Le Immagini (Originale vs Analisi)
                col_i1, col_i2 = st.columns(2)
                
                with col_i1:
                    st.image(crop_img, channels="BGR", caption=f"Crop (H:{h_mm:.1f}mm - W:{w_mm:.1f}mm)")
                
                with col_i2:
                    # Overlay Rosso (Visualizzazione Scientifica)
                    overlay = color.label2rgb(pores_labeled, image=gray, bg_label=0, colors=['red'], alpha=0.3)
                    st.image(overlay, caption="Mappa Alveolatura (Rosso)", clamp=True)
                
                # C. Grafico Distribuzione Pori (Weibull)
                if w_alpha > 0:
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.hist(areas_mm, bins=30, density=True, alpha=0.5, color='gray', label='Dati')
                    x_plot = np.linspace(min(areas_mm), max(areas_mm), 200)
                    ax.plot(x_plot, weibull_min.pdf(x_plot, w_alpha, scale=w_beta), 'r-', lw=2, label=f'Fit α={w_alpha:.2f}')
                    ax.legend(fontsize='small')
                    ax.set_title("Distribuzione Dimensione Pori", fontsize='small')
                    ax.axis('off') # Rimuove assi per pulizia su mobile
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Raccolta dati per Excel
                results_list.append({
                    "File": fname,
                    "Porosità (%)": porosity,
                    "Frattale (FD)": fd,
                    "Spessore Pareti (mm)": wall_th,
                    "Weibull Alpha": w_alpha,
                    "Weibull Beta": w_beta,
                    "Altezza (mm)": h_mm,
                    "Larghezza (mm)": w_mm,
                    "Area Poro Avg (mm2)": np.mean(areas_mm) if areas_mm else 0
                })
            else:
                st.warning(f"⚠️ Impossibile rilevare il muffin in {fname}. Prova a migliorare la luce o regolare i parametri HSV nella sidebar.")

    # TAB: DOWNLOAD DATI
    with tab_data:
        if results_list:
            df = pd.DataFrame(results_list)
            st.dataframe(df, use_container_width=True)
            
            # Export Excel in memoria
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Analisi')
            
            st.download_button(
                label="📥 Scarica Excel Completo",
                data=buffer.getvalue(),
                file_name="Analisi_Muffin_Mobile.xlsx",
                mime="application/vnd.ms-excel"
            )
        else:
            st.info("Carica o scatta una foto per vedere i dati.")
