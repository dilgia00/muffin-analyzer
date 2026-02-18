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

# 1. Configurazione Iniziale
register_heif_opener()
st.set_page_config(page_title="Muffin Analyzer", layout="wide", page_icon="🧁")

# --- FUNZIONI DI ELABORAZIONE ---
def load_image(image_file):
    try:
        image = Image.open(image_file)
        # Corregge l'orientamento EXIF (importante per foto da cellulare)
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
        except:
            pass
            
        img_np = np.asarray(image)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_np
    except Exception as e:
        st.error(f"Errore caricamento immagine: {e}")
        return None

def smart_crop(img):
    # Converte in HSV per trovare il muffin (ignora lo sfondo bianco/chiaro)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Range colore per prodotti da forno (molto ampio)
    lower = np.array([0, 20, 50])
    upper = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Pulizia rumore
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img, mask, (0,0,img.shape[1],img.shape[0])

    # Prendi il contorno più grande (il muffin)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Padding opzionale
    pad = 10
    h_img, w_img = img.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(w_img - x, w + 2*pad)
    h = min(h_img - y, h + 2*pad)
    
    return img[y:y+h, x:x+w], mask[y:y+h, x:x+w], (x,y,w,h)

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

# --- INTERFACCIA UTENTE ---
st.title("🧁 Muffin Lab - Analisi Scientifica")
st.markdown("Scatta una foto o caricane una per analizzare **alveolatura, dimensioni e cottura**.")

# 1. Input: Fotocamera o Upload
option = st.radio("Scegli input:", ["📸 Scatta Foto", "📂 Carica File"], horizontal=True)

img_file = None
if option == "📸 Scatta Foto":
    img_file = st.camera_input("Inquadra il muffin intero o la fetta")
else:
    img_file = st.file_uploader("Carica immagine (JPG, PNG, HEIC)", type=['png', 'jpg', 'jpeg', 'heic'])

# 2. Parametri di Calibrazione
with st.expander("⚙️ Calibrazione Dimensioni (Opzionale)", expanded=False):
    st.info("Per ottenere misure in mm, indica la larghezza reale della base (pirottino).")
    DIAMETRO_REALE = st.number_input("Larghezza reale base (mm)", value=50.0, step=0.5)

# --- ANALISI ---
if img_file is not None:
    # Caricamento
    original_img = load_image(img_file)
    
    if original_img is not None:
        # Resize per non bloccare il telefono (max 800px width)
        h_orig, w_orig = original_img.shape[:2]
        scale = 800 / w_orig
        dim = (800, int(h_orig * scale))
        img_resized = cv2.resize(original_img, dim)
        
        st.write("---")
        st.subheader("1. Rilevamento Oggetto")
        
        # Crop
        crop_img, crop_mask, (x,y,w_crop,h_crop) = smart_crop(img_resized)
        
        # Calcolo Calibrazione (PPM)
        # Assumiamo che la larghezza del crop corrisponda grossomodo alla larghezza del muffin
        # Se l'utente inquadra bene, w_crop è la larghezza del muffin in pixel
        pixel_per_mm = w_crop / 65.0 # Fallback: assumiamo un muffin medio di 65mm cupola
        
        # Raffinamento PPM se basato su base nota
        # (Qui semplifichiamo: usiamo la larghezza totale del crop mappata su un diametro "medio" se non specificato meglio, 
        #  ma idealmente l'utente dovrebbe ritagliare. Per ora usiamo un cursore manuale di aggiustamento)
        
        col_img1, col_img2 = st.columns(2)
        col_img1.image(crop_img, channels="BGR", caption="Muffin Ritagliato")
        col_img2.image(crop_mask, clamp=True, caption="Maschera Binaria")
        
        st.write("---")
        st.subheader("2. Analisi Struttura")
        
        with st.spinner("Calcolo parametri complessi (Frattali, Weibull)..."):
            
            # --- ELABORAZIONE IMMAGINE ---
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray_enhanced = clahe.apply(gray)
            
            # Binarizzazione Pori
            thresh = cv2.adaptiveThreshold(gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 25, 5)
            
            # Maschera sicura (leggera erosione per togliere bordi esterni)
            kernel_erode = np.ones((5,5), np.uint8)
            mask_safe = cv2.erode(crop_mask, kernel_erode, iterations=1)
            
            # Pori finali
            pores = (thresh > 0) & (mask_safe > 0)
            pores = morphology.remove_small_objects(pores, min_size=5)
            
            # --- METRICHE ---
            # 1. Porosità
            area_totale = np.sum(mask_safe > 0)
            area_pori = np.sum(pores)
            porosita_pct = (area_pori / area_totale) * 100 if area_totale > 0 else 0
            
            # 2. Frattale
            fd = fractal_dimension(pores)
            
            # 3. Spessore Pareti (Approssimato)
            dough_mask = (~pores) & (mask_safe > 0)
            dist_transform = ndimage.distance_transform_edt(dough_mask)
            spessore_px = np.mean(dist_transform[dough_mask]) * 2 if np.any(dough_mask) else 0
            
            # Calibrazione Dinamica (Slider per aggiustare se la misura non torna)
            ppm_final = w_crop / (DIAMETRO_REALE * 1.2) # Stima iniziale: larghezza crop è circa il 120% della base
            # Lasciamo l'utente aggiustare la scala vedendo i risultati
            
            # --- VISUALIZZAZIONE RISULTATI ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Porosità", f"{porosita_pct:.1f} %", delta=None)
            m1.caption("Percentuale di vuoti")
            
            m2.metric("Frattale (FD)", f"{fd:.3f}")
            m2.caption("Complessità alveolatura")
            
            m3.metric("Spessore Pareti", f"{spessore_px:.1f} px")
            m3.caption("(Valore relativo in pixel)")
            
            # --- GRAFICI ---
            st.subheader("3. Distribuzione Pori e Weibull")
            
            labels = measure.label(pores)
            props = measure.regionprops(labels)
            areas_px = [p.area for p in props if p.area > 10]
            
            if len(areas_px) > 20:
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # Fit Weibull
                shape, loc, scale = weibull_min.fit(areas_px, floc=0)
                
                # Plot Istogramma
                ax.hist(areas_px, bins=40, density=True, alpha=0.6, color='skyblue', label='Dati Reali')
                
                # Plot Curva Weibull
                x = np.linspace(min(areas_px), max(areas_px), 200)
                pdf = weibull_min.pdf(x, shape, scale=scale)
                ax.plot(x, pdf, 'r-', lw=2, label=f'Weibull (α={shape:.2f})')
                
                ax.set_title("Distribuzione Dimensione Pori")
                ax.set_xlabel("Area Poro (pixel)")
                ax.legend()
                
                st.pyplot(fig)
                
                if shape > 1.5:
                    st.success(f"✅ Alveolatura regolare e fine (Alpha {shape:.2f} alto)")
                else:
                    st.warning(f"⚠️ Alveolatura irregolare/grossolana (Alpha {shape:.2f} basso)")
            else:
                st.warning("Non abbastanza pori rilevati per il calcolo statistico.")
            
            # Overlay Visivo
            st.write("Mappa Pori Rilevati (Rosso):")
            overlay = color.label2rgb(labels, image=gray, bg_label=0, colors=['red'], alpha=0.3)
            st.image(overlay, clamp=True, channels="RGB", use_container_width=True)