import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Soil Fertility Pro",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DEFINISI KONSTANTA & RANGE ---
FIXED_TARGET_NAMES = ['High', 'Moderate']

FIXED_FEATURE_NAMES = [
    'Temperature', 'Rainfall', 'pH', 'Light_Hours', 'Light_Intensity', 'Rh',
    'Nitrogen', 'Phosphorus', 'Potassium', 'Yield',
    'N_Ratio', 'P_Ratio', 'K_Ratio',
    'Soil_Type_Loam', 'Soil_Type_Sandy',
    'Season_Spring', 'Season_Summer', 'Season_Winter',
    'Photoperiod_Long Day', 'Photoperiod_Short Day Period'
]

LIMITS = {
    'temp': {'min': 10.3, 'max': 30.5, 'msg': '10 - 30.5 °C'},
    'rain': {'min': 400.0, 'max': 1712.0, 'msg': '400 - 1712 mm'},
    'ph': {'min': 5.5, 'max': 7.4, 'msg': '5.5 - 7.4'},
    'light_h': {'min': 5.0, 'max': 16.0, 'msg': '5 - 16 Jam'},
    'light_i': {'min': 69.0, 'max': 873.0, 'msg': '69 - 873 Lux'},
    'rh': {'min': 29.0, 'max': 100.0, 'msg': '29 - 100 %'},
    'n': {'min': 53.0, 'max': 224.0, 'msg': '53 - 224 mg/kg'},
    'p': {'min': 13.0, 'max': 277.0, 'msg': '13 - 277 mg/kg'},
    'k': {'min': 35.0, 'max': 398.0, 'msg': '35 - 398 mg/kg'},
    'yield': {'min': 0.7, 'max': 54.0, 'msg': '0.7 - 54 Ton/Ha'}
}

# --- 3. LOAD MODEL ---


@st.cache_resource
def load_model():
    try:
        model_data = torch.load('soil_fertility_model.pth',
                                map_location=torch.device('cpu'),
                                weights_only=False)
        model = model_data['model']
        model.eval()
        return {
            'model': model,
            'scaler_mean': model_data['scaler_mean'],
            'scaler_scale': model_data['scaler_scale']
        }
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


data_model = load_model()
if not data_model:
    st.stop()

model = data_model['model']
scaler_mean = data_model['scaler_mean']
scaler_scale = data_model['scaler_scale']

# --- 4. MANAJEMEN SESSION STATE (INIT 0) ---
input_keys = ['temp', 'rain', 'rh', 'light_h', 'light_i',
              'ph', 'n', 'p', 'k',
              'yield', 'n_r', 'p_r', 'k_r']

for key in input_keys:
    if key not in st.session_state:
        st.session_state[key] = 0.0

if 'soil' not in st.session_state:
    st.session_state['soil'] = 'Loam'
if 'season' not in st.session_state:
    st.session_state['season'] = 'Summer'
if 'photo' not in st.session_state:
    st.session_state['photo'] = 'Short Day Period'


def set_random_data():
    st.session_state['temp'] = round(random.uniform(
        LIMITS['temp']['min'], LIMITS['temp']['max']), 1)
    st.session_state['rain'] = round(random.uniform(
        LIMITS['rain']['min'], LIMITS['rain']['max']), 1)
    st.session_state['ph'] = round(random.uniform(
        LIMITS['ph']['min'], LIMITS['ph']['max']), 2)
    st.session_state['rh'] = round(random.uniform(
        LIMITS['rh']['min'], LIMITS['rh']['max']), 1)
    st.session_state['light_h'] = round(random.uniform(
        LIMITS['light_h']['min'], LIMITS['light_h']['max']), 1)
    st.session_state['light_i'] = round(random.uniform(
        LIMITS['light_i']['min'], LIMITS['light_i']['max']), 1)
    st.session_state['n'] = round(random.uniform(
        LIMITS['n']['min'], LIMITS['n']['max']), 1)
    st.session_state['p'] = round(random.uniform(
        LIMITS['p']['min'], LIMITS['p']['max']), 1)
    st.session_state['k'] = round(random.uniform(
        LIMITS['k']['min'], LIMITS['k']['max']), 1)

    # Soil Type Updated
    st.session_state['soil'] = random.choice(["Sandy Loam", "Loam", "Sandy"])
    st.session_state['season'] = random.choice(
        ["Fall", "Spring", "Summer", "Winter"])
    st.session_state['photo'] = random.choice(
        ["Day Neutral", "Long Day", "Short Day Period"])

    st.session_state['n_r'] = 10.0
    st.session_state['p_r'] = 10.0
    st.session_state['k_r'] = 10.0

    # Hitung Yield berdasarkan rumus terbaru
    avg_ratio = (
        st.session_state['n_r'] + st.session_state['p_r'] + st.session_state['k_r']) / 3
    yield_calc = 20.0 + (avg_ratio - 10.0) * 0.85

    # Penalty Banjir (Rain > 1000mm)
    if st.session_state['rain'] > 1000:
        penalty_rain = (st.session_state['rain'] - 1000) * 0.025
        yield_calc -= penalty_rain

    # Penalty Kering (Rh < 40%)
    if st.session_state['rh'] < 40:
        penalty_dry = (40 - st.session_state['rh']) * 4.0
        yield_calc -= penalty_dry

    # Clipping
    yield_calc = max(0.1, min(yield_calc, 54.0))

    # Reset 0 jika input kosong
    if st.session_state['n'] == 0:
        yield_calc = 0.0

    st.session_state['yield'] = round(yield_calc, 2)


def clear_data():
    for key in input_keys:
        st.session_state[key] = 0.0


# --- 5. SIDEBAR INPUT ---
st.sidebar.title("🎛️ Input Data")

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🎲 Generate Random", use_container_width=True):
    set_random_data()
    st.rerun()

if col_btn2.button("🗑️ Clear Data", use_container_width=True):
    clear_data()
    st.rerun()

st.sidebar.markdown("---")

# GROUP 1: LINGKUNGAN
with st.sidebar.expander("🌤️ Kondisi Lingkungan", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input(
            "Temp (°C)", 0.0, 50.0, key='temp', help=f"Normal: {LIMITS['temp']['msg']}")
        rh = st.number_input("Humidity (%)", 0.0, 100.0,
                             key='rh', help=f"Normal: {LIMITS['rh']['msg']}")
        light_h = st.number_input(
            "Light (H)", 0.0, 24.0, key='light_h', help=f"Normal: {LIMITS['light_h']['msg']}")
    with col2:
        rain = st.number_input(
            "Rain (mm)", 0.0, 3000.0, key='rain', help=f"Normal: {LIMITS['rain']['msg']}")
        light_i = st.number_input(
            "Intensity", 0.0, 5000.0, key='light_i', help=f"Normal: {LIMITS['light_i']['msg']}")

# GROUP 2: KIMIA
with st.sidebar.expander("🧪 Nutrisi & Kimia", expanded=True):
    col3, col4 = st.columns(2)
    with col3:
        ph = st.number_input("pH Tanah", 0.0, 14.0, key='ph',
                             help=f"Normal: {LIMITS['ph']['msg']}")
        n = st.number_input("Nitrogen", 0.0, 1000.0, key='n',
                            help=f"Normal: {LIMITS['n']['msg']}")
    with col4:
        p = st.number_input("Phosphor", 0.0, 1000.0, key='p',
                            help=f"Normal: {LIMITS['p']['msg']}")
        k = st.number_input("Potassium", 0.0, 1000.0, key='k',
                            help=f"Normal: {LIMITS['k']['msg']}")

# GROUP 3: BIOLOGI
with st.sidebar.expander("🌿 Faktor Biologis", expanded=True):
    # Updated Soil List
    soil = st.selectbox(
        "Soil Type", ["Sandy Loam", "Loam", "Sandy"], key='soil')
    season = st.selectbox(
        "Season", ["Fall", "Spring", "Summer", "Winter"], key='season')
    photo = st.selectbox(
        "Photoperiod", ["Day Neutral", "Long Day", "Short Day Period"], key='photo')

# GROUP 4: TARGET & RATIO
# --- LOGIKA YIELD CERDAS (UPDATED) ---
# 1. Base Yield dari Rata-rata Rasio (Korelasi kuat di data Moderate vs High normal)
avg_ratio = (st.session_state['n_r'] +
             st.session_state['p_r'] + st.session_state['k_r']) / 3
yield_calc = 20.0 + (avg_ratio - 10.0) * 0.85

# 2. Penalty Banjir (Rain > 1000mm)
# Data High 0.86 punya Rain 1667.
if st.session_state['rain'] > 1000:
    penalty_rain = (st.session_state['rain'] - 1000) * 0.025
    yield_calc -= penalty_rain

# 3. Penalty Kering (Rh < 40%)
# Data Moderate 6.8 punya Rh 37.
if st.session_state['rh'] < 40:
    penalty_dry = (40 - st.session_state['rh']) * 4.0
    yield_calc -= penalty_dry

# 4. Clipping
yield_calc = max(0.1, min(yield_calc, 54.0))

# Reset 0 jika input kosong
if st.session_state['n'] == 0:
    yield_calc = 0.0

with st.sidebar.expander("🎯 Target & Rasio", expanded=True):
    final_yield = st.number_input("Yield (Ton/Ha)", 0.0, 100.0, key='yield')

    # Rasio MANUAL
    c_r1, c_r2, c_r3 = st.columns(3)
    with c_r1:
        nr = st.number_input("N_Ratio", 0.0, 100.0, key='n_r')
    with c_r2:
        pr = st.number_input("P_Ratio", 0.0, 100.0, key='p_r')
    with c_r3:
        kr = st.number_input("K_Ratio", 0.0, 100.0, key='k_r')

btn_predict = st.sidebar.button(
    "🔍 Prediksi Kesuburan Tanah", type="primary", use_container_width=True)
# --- 6. MAIN CONTENT ---
st.title("🌾 Soil Fertility Analysis")
st.subheader("Soil Fertility Classification")
st.markdown('''
Aplikasi untuk mengklasifikasikan tingkat kesuburan tanah
berdasarkan kondisi lingkungan dan kandungan nutrisi tanah berbasis Neural Network.
''')

if btn_predict:
    # A. CEK DATA NOL
    if st.session_state['n'] == 0 and st.session_state['ph'] == 0:
        st.warning(
            "⚠️ Data input masih 0. Gunakan tombol **'🎲 Generate Random'**.")
        st.stop()

    # B. PREPARE DATA
    input_data = {
        'Temperature': np.clip(st.session_state['temp'], LIMITS['temp']['min'], LIMITS['temp']['max']),
        'Rainfall': np.clip(st.session_state['rain'], LIMITS['rain']['min'], LIMITS['rain']['max']),
        'pH': np.clip(st.session_state['ph'], LIMITS['ph']['min'], LIMITS['ph']['max']),
        'Light_Hours': np.clip(st.session_state['light_h'], LIMITS['light_h']['min'], LIMITS['light_h']['max']),
        'Light_Intensity': np.clip(st.session_state['light_i'], LIMITS['light_i']['min'], LIMITS['light_i']['max']),
        'Rh': np.clip(st.session_state['rh'], LIMITS['rh']['min'], LIMITS['rh']['max']),
        'Nitrogen': np.clip(st.session_state['n'], LIMITS['n']['min'], LIMITS['n']['max']),
        'Phosphorus': np.clip(st.session_state['p'], LIMITS['p']['min'], LIMITS['p']['max']),
        'Potassium': np.clip(st.session_state['k'], LIMITS['k']['min'], LIMITS['k']['max']),
        'Yield': np.clip(final_yield, LIMITS['yield']['min'], LIMITS['yield']['max']),
        'N_Ratio': nr, 'P_Ratio': pr, 'K_Ratio': kr,
        'Soil_Type': [soil], 'Season': [season], 'Photoperiod': [photo]
    }

    # C. ENCODING
    df_raw = pd.DataFrame({k: [v] if not isinstance(
        v, list) else v for k, v in input_data.items()})
    df_enc = pd.get_dummies(
        df_raw, columns=['Soil_Type', 'Season', 'Photoperiod'], drop_first=True)
    df_ready = df_enc.reindex(columns=FIXED_FEATURE_NAMES, fill_value=0)

    # D. PREDICT
    X_scaled = (df_ready.values - scaler_mean) / scaler_scale
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_scaled))
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)

    label = FIXED_TARGET_NAMES[pred_idx]
    conf = probs[pred_idx] * 100

    # --- OUTPUT SECTION ---
    st.divider()
    st.subheader("📊 Hasil Prediksi")

    if label == "High":
        st.success(f"✅ Tingkat Kesuburan: **{label}**")
        st.balloons()
    else:
        st.warning(f"⚠️ Tingkat Kesuburan: **{label}**")

    st.progress(int(conf))
    st.caption(f"Confidence: {conf:.1f}%")

    st.write("**📈 Probabilitas Detail:**")
    prob_df = pd.DataFrame({'Class': FIXED_TARGET_NAMES, 'Probability': probs})
    st.dataframe(prob_df, hide_index=True, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#4CAF50" if cls ==
              "High" else "#FF9800" for cls in FIXED_TARGET_NAMES]
    bars = ax.bar(FIXED_TARGET_NAMES, probs, color=colors,
                  alpha=0.7, edgecolor='black')
    ax.set_ylabel("Probabilitas", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Distribusi Probabilitas Kesuburan", fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10)

    # Menampilkan grafik dengan container width agar responsif dan terlihat "center"/penuh
    st.pyplot(fig, use_container_width=True)

    st.subheader("📋 Ringkasan Parameter Input")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌡️ Temp", f"{st.session_state['temp']:.2f}°C")
    c2.metric("💧 Rain", f"{st.session_state['rain']:.2f}mm")
    c3.metric("💨 Humidity", f"{st.session_state['rh']:.2f}%")
    c4.metric("☀️ Light", f"{st.session_state['light_h']:.2f}h")
    c5.metric("💡 Intensity", f"{st.session_state['light_i']:.2f}lx")

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("⚗️ pH", f"{st.session_state['ph']:.2f}")
    c7.metric("🔵 Nitrogen", f"{st.session_state['n']:.2f} mg")
    c8.metric("🔴 Phosphor", f"{st.session_state['p']:.2f} mg")
    c9.metric("🟡 Potassium", f"{st.session_state['k']:.2f} mg")

    st.markdown("---")

    st.subheader("💡 Rekomendasi Perbaikan")
    recommendations = []
    if label == "High":
        st.success("✅ **Tanah Anda memiliki kesuburan tinggi!**")
        st.info(
            "Kondisi tanah optimal. Pertahankan dengan: Pemupukan rutin, Monitor pH, Rotasi tanaman.")
    else:
        st.warning("⚠️ **Tanah memerlukan perbaikan**")
        if st.session_state['n'] < 100:
            recommendations.append(
                "🔹 **Nitrogen rendah**: Tambahkan pupuk urea.")
        if st.session_state['p'] < 50:
            recommendations.append(
                "🔹 **Phosphorus rendah**: Gunakan pupuk SP-36.")
        if st.session_state['k'] < 100:
            recommendations.append(
                "🔹 **Potassium rendah**: Aplikasikan pupuk KCl.")
        if st.session_state['ph'] < 5.5:
            recommendations.append("🔹 **pH Asam**: Lakukan pengapuran.")
        elif st.session_state['ph'] > 7.5:
            recommendations.append("🔹 **pH Basa**: Berikan Sulfur.")
        if st.session_state['rain'] < 500:
            recommendations.append("🔹 **Hujan Kurang**: Perbaiki irigasi.")
        if st.session_state['rh'] < 50:
            recommendations.append("🔹 **RH Rendah**: Gunakan mulsa.")

    for rec in recommendations:
        st.markdown(rec)
    st.divider()

    with st.expander("📚 Tips Meningkatkan Kesuburan Tanah"):
        st.markdown("""
        **Praktik Terbaik Pengelolaan Tanah:**
        1. **Pemupukan Berimbang**: Gunakan rasio NPK yang sesuai.
        2. **Pengaturan pH**: Jaga di angka 6.0 - 7.0.
        3. **Manajemen Air**: Pastikan drainase baik.
        4. **Rotasi Tanaman**: Gunakan tanaman legum.
        5. **Bahan Organik**: Tambahkan kompos.
        """)

else:
    # --- PANDUAN PENGGUNAAN (KANAN SAAT START) ---
    col_intro, col_model, col_guide = st.columns([1, 1, 1])

    with col_intro:
        st.info("👈 Masukkan data parameter di Sidebar (Kiri).")
        st.write(
            "Jika belum memiliki data, gunakan tombol **'🎲 Generate Random'** di sidebar untuk simulasi.")

    with col_model:
        st.success("✅ **Model Info**")
        st.markdown("""
        - **Algorithm**: Neural Network
        - **Features**: 18 input features
        - **Classes**: High, Moderate
        """)

    with col_guide:
        st.success("ℹ️ **Panduan Penggunaan**")
        st.markdown("""
        1. **Atur Parameter**: Isi data tanah di sidebar.
        2. **Isi Yield**: Masukkan nilai yield secara manual.
        3. **Prediksi**: Klik tombol **'🔍 Prediksi Kesuburan Tanah'**.
        4. **Hasil**: Lihat analisis dan rekomendasi di sini.
        """)
