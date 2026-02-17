import streamlit as st
import requests
from PIL import Image
import io
import base64
import json
import os

# ---- Config ----
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ---- Page Setup ----
st.set_page_config(
    page_title="Détection de Pneumonie",
    page_icon="🫁",
    layout="wide"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid;
        margin: 0.5rem 0;
    }
    .normal-result {
        background: #d4edda;
        border-color: #28a745;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
    }
    .pneumonia-result {
        background: #f8d7da;
        border-color: #dc3545;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="main-header">
    <h1>🫁 Détection de Pneumonie par Rayons X</h1>
    <p>Système d'aide au diagnostic basé sur le Deep Learning (EfficientNet-B0)</p>
    <p><em>Ahmed Ben Attia Khiari & Achref Ghorbel — Polytech International</em></p>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.header("ℹ️ À propos")
    st.markdown("""
    Ce système utilise un modèle **EfficientNet-B0** entraîné sur le dataset 
    **Chest X-Ray Images** pour détecter la pneumonie dans les radiographies thoraciques.

    **⚠️ Avertissement:** Ce système est un outil d'aide au diagnostic à usage académique 
    uniquement. Il ne remplace pas l'avis d'un professionnel de santé.
    """)

    st.header("📊 Métriques du Modèle")
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            m = data["metrics"]
            st.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
            st.metric("AUC-ROC", f"{m['auc_roc']:.4f}")
            st.metric("Recall", f"{m['recall']*100:.1f}%")
            st.metric("Specificity", f"{m['specificity']*100:.1f}%")
            st.metric("F1-Score", f"{m['f1_score']:.4f}")
        else:
            st.warning("API non disponible")
    except:
        st.warning("⚠️ API non connectée. Lancez: `uvicorn app:app --port 8000`")

# ---- Main Content ----
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Radiographie")
    uploaded_file = st.file_uploader(
        "Choisir une image radiographique (JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Uploadez une radiographie thoracique pour analyse"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image uploadée", use_container_width=True)

with col2:
    st.header("🔍 Résultat")

    if uploaded_file:
        with st.spinner("Analyse en cours..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]

                    # Result display
                    if prediction == "NORMAL":
                        st.markdown(f"""
                        <div class="normal-result">
                            <h2>✅ {prediction}</h2>
                            <h3>Confiance: {confidence}%</h3>
                            <p>Aucun signe de pneumonie détecté</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="pneumonia-result">
                            <h2>⚠️ {prediction}</h2>
                            <h3>Confiance: {confidence}%</h3>
                            <p>Signes de pneumonie détectés — Consultez un médecin</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Probabilities
                    st.subheader("📊 Probabilités")
                    probs = result["probabilities"]
                    col_a, col_b = st.columns(2)
                    col_a.metric("Normal", f"{probs['NORMAL']}%")
                    col_b.metric("Pneumonie", f"{probs['PNEUMONIA']}%")

                    # Progress bar
                    st.progress(probs["PNEUMONIA"] / 100)

                    # Grad-CAM
                    st.subheader("🔥 Carte d'attention (Grad-CAM)")
                    st.caption("Les zones rouges/jaunes indiquent où le modèle concentre son attention")
                    gradcam_bytes = base64.b64decode(result["gradcam_image"])
                    gradcam_img = Image.open(io.BytesIO(gradcam_bytes))
                    st.image(gradcam_img, caption="Grad-CAM Overlay", use_container_width=True)

                else:
                    st.error(f"Erreur API: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de se connecter à l'API. Lancez: `uvicorn app:app --port 8000`")
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    else:
        st.info("👈 Uploadez une radiographie pour commencer l'analyse")

# ---- Footer ----
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Projet Deep Learning — Détection de Pneumonie | Polytech International 2026</p>
    <p>Professeur: Haythem Ghazouani</p>
</div>

""", unsafe_allow_html=True)
