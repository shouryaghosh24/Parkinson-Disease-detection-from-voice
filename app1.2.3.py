import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from fpdf import FPDF
import tempfile
import os
import joblib
from sklearn.preprocessing import StandardScaler

# ---------------- Config ----------------
MODEL_PATH = "parkinson_model_87.h5"
SCALER_PATH = "scaler.pkl"
TRAIN_DATA_DIR = r"C:\Users\KIIT\Documents\project_ind4.0\data\kaggle_voice"
MFCC_N = 40

# ---------------- Streamlit Page Config ----------------
st.set_page_config(page_title="Parkinson’s Voice Detection", page_icon="🧠", layout="wide")

# ---------------- 🎨 UI & CSS Styling ----------------
st.markdown("""
<style>
/* Background and page container */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #0f2027, #203a43, #2c5364);
    color: #F5F5F5;
    font-family: 'Segoe UI', sans-serif;
    padding-top: 1rem;
}

/* Header */
h1 {
    color: #FFD700 !important;
    text-align: center;
    font-size: 2.7em !important;
    margin-bottom: 0.4em;
    text-shadow: 2px 2px 8px rgba(255, 215, 0, 0.4);
}

/* General paragraph style */
p {
    font-size: 1.07em !important;
    color: #EAEAEA !important;
    line-height: 1.75em !important;
    text-align: justify !important;
}

/* Section headers */
.section-header {
    font-size: 1.6em !important;
    color: #FFD700;
    text-align: center;
    margin-top: 1.7em;
    font-weight: 700;
    text-shadow: 1px 1px 6px rgba(255, 255, 0, 0.4);
}

/* File uploader box */
[data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 15px;
    padding: 25px;
    border: 2px dashed #FFD700;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.15);
}

/* Alerts (success/info/error) */
.stAlert {
    border-radius: 12px !important;
    padding: 1rem !important;
    font-weight: 600;
    font-size: 1.05em;
}
.stAlert[data-baseweb="alert-success"] {
    background-color: rgba(0, 128, 0, 0.15);
    border-left: 6px solid #00FF7F;
}
.stAlert[data-baseweb="alert-error"] {
    background-color: rgba(255, 0, 0, 0.15);
    border-left: 6px solid #FF4C4C;
}
.stAlert[data-baseweb="alert-info"] {
    background-color: rgba(30, 144, 255, 0.15);
    border-left: 6px solid #1E90FF;
}
.stAlert[data-baseweb="alert-warning"] {
    background-color: rgba(255, 165, 0, 0.15);
    border-left: 6px solid #FFD700;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #FFD700, #FF8C00);
    color: #000;
    font-weight: 700;
    border-radius: 10px;
    padding: 12px 24px;
    transition: 0.3s;
    border: none;
    box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
}
.stButton>button:hover {
    background: linear-gradient(90deg, #FFB400, #FF6F00);
    transform: scale(1.05);
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.5);
}

/* Dataframe table */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0px 0px 12px rgba(255, 255, 255, 0.1);
    margin-top: 20px;
}

/* Footer */
.footer {
    color: #FFD700;
    text-align: center;
    font-size: 1.08em;
    margin-top: 50px;
    padding-top: 15px;
    border-top: 1px solid rgba(255, 255, 255, 0.15);
    text-shadow: 1px 1px 5px rgba(255, 215, 0, 0.3);
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header & Instructions ----------------
st.markdown("<h1>🧠 Parkinson’s Disease Detection from Voice</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#E6E6E6;'>
Parkinson’s Disease is a progressive disorder that affects movement, causing tremors, stiffness, and slowness.<br>
Early detection helps in better management and improved quality of life.
</p>
""", unsafe_allow_html=True)

st.markdown("<div class='section-header'>🎙️ Upload a voice recording (.wav) to check your Parkinson’s result</div>", unsafe_allow_html=True)
st.markdown("""
<p style='color:#FFD700; font-weight:600;'>
🎧 If you have a file in <b>.mp3</b> format or any other format, please convert it to <b>.wav</b>.
</p>
<p style='color:#FFD700; font-weight:600;'>
🗣️ Please upload a voice like “aaa” for better results.<br>
⚠️ Even small disturbances in the voice can change the result. Upload the clearest voice sample possible.<br>
🎙️ It is advisable to use a microphone for best accuracy.
</p>
<p style='color:#FF4C4C; font-weight:700;'>
🚨 WARNING: This is an AI model trained on open-source data and may make mistakes. Always consult a medical professional for confirmation.
</p>
""", unsafe_allow_html=True)

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("", type=["wav"])

# ---------------- Load Model ----------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please place your .h5 model there.")
    st.stop()

model = load_model(MODEL_PATH)
st.info(f"Model loaded from {MODEL_PATH}")

# ---------------- Load Scaler ----------------
try:
    scaler = joblib.load(SCALER_PATH)
    st.success("Scaler loaded successfully ✅")
except Exception as e:
    st.warning(f"Scaler not available: {e}. Using fallback normalization.")
    scaler = None

# ---------------- Feature Extraction ----------------
def extract_mfcc_mean_from_path(path, n_mfcc=MFCC_N):
    y, sr = librosa.load(path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

def extract_features_from_filelike(filelike, n_mfcc=MFCC_N):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(filelike.read())
        tmp.flush()
        tmp.close()
        y, sr = librosa.load(tmp.name, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        display_params = {
            "Mean MFCC": float(np.mean(mfcc_mean)),
            "Mean Chroma": float(np.mean(chroma)),
            "Spectral Contrast": float(np.mean(spectral_contrast)),
            "Zero Crossing Rate": float(np.mean(zcr)),
            "Root Mean Square Energy": float(np.mean(rms))
        }
        return mfcc_mean.reshape(1, -1), y, sr, mfccs, display_params
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

# ---------------- PDF Report ----------------
def create_pdf_report(params, diagnosis, prob_healthy=None, prob_parkinson=None):
    pdf = FPDF()
    pdf.add_page()
    font_path = os.path.join(os.getcwd(), "DejaVuSans.ttf")
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", "", 14)
    else:
        pdf.set_font("Arial", "", 14)
    pdf.cell(0, 10, "Parkinson’s Voice Analysis Report", ln=True, align="C")
    pdf.cell(0, 10, "", ln=True)
    pdf.cell(0, 10, f"Diagnosis: {diagnosis}", ln=True)
    if prob_healthy is not None and prob_parkinson is not None:
        pdf.set_font("DejaVu", "", 12)
        pdf.cell(0, 8, f"Probability (Healthy): {prob_healthy*100:.2f}%", ln=True)
        pdf.cell(0, 8, f"Probability (Parkinson): {prob_parkinson*100:.2f}%", ln=True)
    pdf.cell(0, 10, "", ln=True)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, "Extracted Voice Parameters:", ln=True)
    pdf.set_font("DejaVu", "", 10)
    for k, v in params.items():
        try:
            pdf.cell(0, 8, f"{k}: {float(v):.4f}", ln=True)
        except Exception:
            pdf.cell(0, 8, f"{k}: {v}", ln=True)
    if os.path.exists("mfcc_plot.png"):
        pdf.image("mfcc_plot.png", x=40, y=pdf.get_y() + 5, w=130)
    pdf.set_y(-30)
    pdf.set_font("DejaVu", "", 10)
    pdf.cell(0, 10, "Please consult a trained medical professional in case of doubt.", align="R")
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

# ---------------- Main Logic ----------------
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.write("🔍 Extracting voice features...")

    try:
        features, y, sr, mfccs, display_params = extract_features_from_filelike(uploaded_file)

        if scaler is not None:
            try:
                features_scaled = scaler.transform(features)
            except Exception as e:
                st.warning(f"Scaler transform failed: {e} — falling back to per-sample z-score.")
                features_scaled = (features - np.mean(features)) / (np.std(features) + 1e-9)
        else:
            features_scaled = (features - np.mean(features)) / (np.std(features) + 1e-9)

        pred = model.predict(features_scaled)
        prob_parkinson = None
        prob_healthy = None
        if pred.ndim == 2 and pred.shape[1] == 1:
            prob_parkinson = float(pred[0][0])
            prob_healthy = 1.0 - prob_parkinson
        elif pred.ndim == 2 and pred.shape[1] == 2:
            prob_healthy = float(pred[0][0])
            prob_parkinson = float(pred[0][1])
        else:
            prob_parkinson = float(pred[0][0])
            prob_healthy = 1.0 - prob_parkinson

        if prob_parkinson > prob_healthy:
            diagnosis = "🩺 Parkinson’s Detected"
            st.error(f"{diagnosis} — please consult a neurologist.")
        else:
            diagnosis = "✅ Healthy Voice"
            st.success(f"{diagnosis} — No Parkinson’s signs found.")

        st.write(f"**Probability — Healthy:** {prob_healthy*100:.2f}%    **Parkinson:** {prob_parkinson*100:.2f}%")

        st.markdown("### 🎵 Extracted Voice Parameters")
        st.dataframe({k: [v] for k, v in display_params.items()})

        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.specshow(mfccs, x_axis="time", ax=ax)
        ax.set(title="MFCCs of the Voice Sample")
        plt.tight_layout()
        plt.savefig("mfcc_plot.png")
        st.pyplot(fig)

        if st.button("📄 Generate PDF Report"):
            pdf_path = create_pdf_report(display_params, diagnosis, prob_healthy, prob_parkinson)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Report",
                    data=f,
                    file_name="Parkinson_Report.pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.warning(f"⚠️ Error processing audio: {e}")

# ---------------- Footer ----------------
st.markdown("<div class='footer'>💻 Developed by <b>Shourya Ghosh</b></div>", unsafe_allow_html=True)
