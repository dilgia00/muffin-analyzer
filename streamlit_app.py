import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
from skimage import measure, morphology, color
from scipy.stats import weibull_min, lognorm, gamma
from scipy import ndimage
import io

# --- CONFIGURAZIONE INIZIALE ---
register_heif_opener()
st.set_page_config(page_title="Muffin Lab Pro", layout="wide", page_icon="🧁")

if 'history_data' not in st.session_state:
    st.session_state['history_data'] = []

st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; border: 1px solid #e0e0e0; }
    [data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    h1 { font-size: 2rem !important; }
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

def apply_filter(gray_img, filter_type, params):
    if filter_type == "CLAHE (Contrasto)":
        clahe = cv2.createCLAHE(clipLimit=params['clip_limit'], tileGridSize=params['tile_grid_size'])
        return clahe.apply(gray_img)
    elif filter_type == "Gaussiano (Smussamento)":
        return cv2.GaussianBlur(gray_img, params['kernel_size'], params['sigma_x'])
    elif filter_type == "Mediano (Rumore Sale/Pepe)":
        return cv2.medianBlur(gray_img, params['kernel_size'])
    elif filter_type == "Gabor (Filtro Direzionale)":
        g_kernel = cv2.getGaborKernel(params['ksize'], params['sigma'], params['theta'], 
                                      params['lambd'], params['gamma'], params['psi'], ktype=cv2.CV_32F)
        return cv2.filter2D(gray_img, cv2.CV_8UC3, g_kernel)
    return gray_img

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
    
    pad = 10
    h_img, w_img = img.shape[:2]
    y1, y2 = max(0, y-pad), min(h_img, y+h+pad)
    x1, x2 = max(0, x-pad), min(w_img, x+w+pad)
    
    base_slice = mask_main[y+h-25 : y+h-5, x:x+w]
    base_w_px = np.max(np.sum(base_slice > 0, axis=1)) if base_slice.shape[0] > 0 and len(np.sum(base_slice > 0, axis=1)) > 0 else w

    return img[y1:y2, x1:x2], mask_main[y1:y2, x1:x2], base_w_px

def add_to_history(batch_results):
    if batch_results:
        existing_files = [item['File'] for item in st.session_state['history_data']]
        new_items = [item for item in batch_results if item['File'] not in existing_files]
        if new_items:
            st.session_state['history_data'].extend(new_items)
            st.toast(f"✅ {len(new_items)} muffin aggiunti allo storico!", icon="💾")

# --- SIDEBAR: CONTROLLI DINAMICI ---
st.sidebar.header("🎛️ Configurazione")
DIAMETRO_BASE_MM = st.sidebar.number_input("Larghezza Base Reale (mm)", value=50.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖼️ Filtro Pre-Elaborazione")
SELECTED_FILTER = st.sidebar.selectbox("Scegli il Filtro", 
    ["CLAHE (Contrasto)", "Gaussiano (Smussamento)", "Mediano (Rumore Sale/Pepe)", "Gabor (Filtro Direzionale)"])

filter_params = {}
with st.sidebar.expander(f"⚙️ Parametri: {SELECTED_FILTER}", expanded=True):
    if SELECTED_FILTER == "CLAHE (Contrasto)":
        filter_params['clip_limit'] = st.slider("Clip Limit (Forza del contrasto)", 1.0, 10.0, 3.0, 0.5)
        t_size = st.slider("Dimensione Griglia (Tile Grid)", 2, 16, 8, 2)
        filter_params['tile_grid_size'] = (t_size, t_size)
    elif SELECTED_FILTER == "Gaussiano (Smussamento)":
        k_size = st.slider("Kernel Size (Dimensione sfocatura - dispari)", 3, 31, 5, 2)
        filter_params['kernel_size'] = (k_size, k_size)
        filter_params['sigma_x'] = st.slider("Sigma X", 0.0, 10.0, 0.0, 0.5)
    elif SELECTED_FILTER == "Mediano (Rumore Sale/Pepe)":
        k_size = st.slider("Kernel Size (dispari)", 3, 31, 5, 2)
        filter_params['kernel_size'] = k_size
    elif SELECTED_FILTER == "Gabor (Filtro Direzionale)":
        k_size = st.slider("Kernel Size", 5, 51, 15, 2)
        filter_params['ksize'] = (k_size, k_size)
        filter_params['sigma'] = st.slider("Sigma (Deviazione Standard)", 0.1, 10.0, 3.0, 0.1)
        theta_deg = st.slider("Theta (Gradi direzionali)", 0, 180, 45, 5)
        filter_params['theta'] = theta_deg * np.pi / 180.0
        filter_params['lambd'] = st.slider("Lambda (Lunghezza d'onda)", 1.0, 30.0, 10.0, 0.5)
        filter_params['gamma'] = st.slider("Gamma (Rapporto spaziale)", 0.1, 2.0, 0.5, 0.1)
        filter_params['psi'] = st.slider("Psi (Fase)", 0.0, 3.14, 0.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Modello Statistico")
STAT_MODEL = st.sidebar.selectbox("Distribuzione dei Pori", ["Weibull", "Lognormale", "Gamma"])

model_params = {}
with st.sidebar.expander(f"📈 Impostazioni: {STAT_MODEL}", expanded=True):
    st.write("Le curve di fitting calcolano i parametri in base alle aree rilevate. Puoi però vincolare alcuni parametri.")
    fix_loc = st.checkbox("Fissa Posizione (Loc = 0)", value=True, help="Forza la partenza della curva da 0. Fisicamente corretto poiché non esistono pori con area negativa.")
    if fix_loc:
        model_params['floc'] = 0

st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Parametri Avanzati (Maschere & HSV)", expanded=False):
    c1, c2 = st.columns(2)
    h_min, h_max = c1.slider("H Min", 0, 180, 0), c2.slider("H Max", 0, 180, 40)
    s_min, v_min = c1.slider("S Min", 0, 255, 30), c1.slider("V Min", 0, 255, 60)
    MIN_PORE_AREA = st.number_input("Area Min Poro (px)", 20)
    MASK_EROSION = st.number_input("Erosione (px)", 15)

LOWER_HSV, UPPER_HSV = np.array([h_min, s_min, v_min]), np.array([h_max, 255, 255])

# --- MAIN PAGE ---
st.title("🔬 Muffin Lab: Analisi Avanzata")

with st.expander("📘 Significato dei Modelli Statistici & Equazioni"):
    st.markdown("""
    I modelli estraggono sempre 3 metriche principali dalla distribuzione delle aree dei pori:
    * **Forma (Shape):** Determina il profilo della curva (simmetria, picchi, asimmetria).
    * **Scala (Scale):** La "diffusione" dei dati o dimensione di riferimento del poro tipico.
    * **Posizione (Loc):** Il punto d'inizio della curva sull'asse X (minimo teorico dell'area).
    
    ---
    **1. Modello di Weibull (Minimo)**
    Ottimo per descrivere particelle o pori. Se Shape > 1, i pori tendono verso una dimensione specifica (alveolatura regolare).
    """)
    st.latex(r"f(x) = \frac{\alpha}{\beta} \left( \frac{x - \gamma}{\beta} \right)^{\alpha - 1} e^{-\left( \frac{x - \gamma}{\beta} \right)^\alpha}")
    
    st.markdown("""
    **2. Modello Lognormale**
    Tipico per processi fisici di accrescimento (es. bolle di gas che si espandono nell'impasto lievitato). Assesta una lunga "coda" verso le dimensioni più ampie.
    """)
    st.latex(r"f(x) = \frac{1}{(x-\gamma) s \sqrt{2\pi}} e^{-\frac{(\ln(x-\gamma) - \mu)^2}{2s^2}} \quad \text{(con Scale} = e^\mu \text{)}")

    st.markdown("""
    **3. Modello Gamma**
    Alternativa flessibile alla Lognormale. Spesso usata per tempi di attesa o accorpamento di difetti in materiali compositi.
    """)
    st.latex(r"f(x) = \frac{(x-\gamma)^{a-1} e^{-(x-\gamma)/\beta}}{\beta^a \Gamma(a)}")

st.info("💡 **Tip per Fotografia:** Per usare il Flash o la massima risoluzione del sensore, scatta prima la foto con l'app nativa e usa 'Carica da Galleria'.")
input_method = st.radio("Sorgente:", ("📂 Carica da Galleria", "📸 Scatta Foto Base"), horizontal=True, label_visibility="collapsed")
image_files = []

if input_method == "📸 Scatta Foto Base":
    cam_file = st.camera_input("Foto Muffin")
    if cam_file: image_files.append(cam_file)
else:
    uploaded = st.file_uploader("Carica Immagini", type=['png', 'jpg', 'jpeg', 'heic'], accept_multiple_files=True)
    if uploaded: image_files = uploaded

tab_analysis, tab_history = st.tabs(["👁️ Analisi Corrente", "🗂️ Storico Sessione (Report)"])
current_batch_results = []

with tab_analysis:
    if image_files:
        for img_file in image_files:
            fname = img_file.name if hasattr(img_file, 'name') else "Foto_Camera"
            img = load_image(img_file)
            if img is None: continue
            
            max_dim = max(img.shape[:2])
            if max_dim > 3000:
                scale_img = 3000 / max_dim
                img = cv2.resize(img, None, fx=scale_img, fy=scale_img)
            
            crop_img, crop_mask, base_w_px = smart_crop(img, LOWER_HSV, UPPER_HSV)
            
            if crop_img is not None:
                ppm = base_w_px / DIAMETRO_BASE_MM
                
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                # Applica il filtro con i parametri scelti dall'utente
                gray_filtered = apply_filter(gray, SELECTED_FILTER, filter_params)
                
                kernel_erode = np.ones((int(MASK_EROSION), int(MASK_EROSION)), np.uint8)
                mask_safe = cv2.erode(crop_mask, kernel_erode, iterations=1) > 0
                
                thresh = cv2.adaptiveThreshold(gray_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
                pores = (thresh > 0) & mask_safe
                pores = morphology.remove_small_objects(pores, min_size=5)
                props = measure.regionprops(measure.label(pores))
                
                porosity = (np.sum(pores) / np.sum(mask_safe)) * 100 if np.sum(mask_safe) > 0 else 0
                fd = fractal_dimension(pores)
                
                dough = (~pores) & mask_safe
                dt = ndimage.distance_transform_edt(dough)
                wall_th = (np.mean(dt[dough]) * 2) / ppm if np.any(dough) else 0
                
                areas_mm = [p.area / (ppm**2) for p in props if p.area >= MIN_PORE_AREA]
                
                shape, loc, scale_param = 0, 0, 0
                if len(areas_mm) > 20:
                    try:
                        # Fitta il modello applicando i vincoli dinamici (es. floc=0)
                        if STAT_MODEL == "Weibull":
                            shape, loc, scale_param = weibull_min.fit(areas_mm, **model_params)
                        elif STAT_MODEL == "Lognormale":
                            shape, loc, scale_param = lognorm.fit(areas_mm, **model_params)
                        elif STAT_MODEL == "Gamma":
                            shape, loc, scale_param = gamma.fit(areas_mm, **model_params)
                    except Exception as e:
                        st.warning(f"Errore fitting {STAT_MODEL}: {e}")
                
                with st.expander(f"📄 Risultati: {fname}", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Porosità", f"{porosity:.1f}%")
                    c2.metric("Spessore Pareti", f"{wall_th:.2f} mm")
                    c3.metric("FD", f"{fd:.3f}")
                    
                    st.markdown(f"**Parametri Modello {STAT_MODEL}:**")
                    cs1, cs2, cs3 = st.columns(3)
                    cs1.metric("Forma (Shape / α o s)", f"{shape:.3f}")
                    cs2.metric("Scala (Scale / β o e^μ)", f"{scale_param:.3f}")
                    cs3.metric("Posizione (Loc / γ)", f"{loc:.3f}")
                    
                    col_i1, col_i2 = st.columns(2)
                    col_i1.image(crop_img, channels="BGR", caption="Originale HD (Croppata)")
                    overlay = color.label2rgb(measure.label(pores), image=gray, bg_label=0, colors=['red'], alpha=0.3)
                    col_i2.image(overlay, caption=f"Pori ({SELECTED_FILTER})")
                
                current_batch_results.append({
                    "File": fname,
                    "Filtro Usato": SELECTED_FILTER,
                    "Param. Filtro": str(filter_params),
                    "Modello Stat.": STAT_MODEL,
                    "Porosità (%)": round(porosity, 2),
                    "Spessore (mm)": round(wall_th, 2),
                    "Frattale (FD)": round(fd, 3),
                    "Forma (Shape)": round(shape, 4),
                    "Scala (Scale)": round(scale_param, 4),
                    "Posizione (Loc)": round(loc, 4)
                })

        if current_batch_results:
            st.divider()
            if st.button("➕ Aggiungi Batch allo Storico", type="primary"):
                add_to_history(current_batch_results)

with tab_history:
    st.subheader("🗂️ Storico Dati")
    if st.session_state['history_data']:
        df_history = pd.DataFrame(st.session_state['history_data'])
        st.dataframe(df_history, use_container_width=True)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_history.to_excel(writer, index=False)
            
        c_down, c_clear = st.columns([1, 1])
        c_down.download_button("📥 Scarica Excel", data=buffer.getvalue(), file_name="Dati_Muffin_Pro.xlsx", mime="application/vnd.ms-excel")
        if c_clear.button("🗑️ Cancella Storico"):
            st.session_state['history_data'] = []
            st.rerun()
        
