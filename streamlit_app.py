# streamlit_app.py - Version chatbot avec saisie audio en bas et CSS amélioré
import streamlit as st
import requests
import time
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"  # URL de l'API FastAPI

st.set_page_config(
    page_title="Assistant Éducatif Audio",
    page_icon="🎓",
    layout="wide"
)

# Initialisation de la session
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'theme' not in st.session_state:
    st.session_state.theme = "light"  # Ajout d'un thème par défaut

# Style CSS personnalisé et amélioré
st.markdown("""
<style>
    /* Variables de couleurs */
    :root {
        --primary-color: #4169E1;
        --secondary-color: #6A0DAD;
        --accent-color: #FF6347;
        --bg-primary: #f8f9fa;
        --bg-secondary: #ffffff;
        --text-primary: #212529;
        --text-secondary: #6c757d;
        --border-color: #e0e0e0;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --hover-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
        --gradient-bg: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    }

    /* Styles généraux */
    .stApp {
        background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
        background-color: var(--bg-primary);
    }

    /* Entête stylisée */
    .main-header {
        position: relative;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        background: var(--gradient-bg);
        color: white;
        border-radius: 15px;
        box-shadow: var(--shadow);
        overflow: hidden;
    }

    .main-header h1 {
        font-weight: 700;
        letter-spacing: 1px;
        position: relative;
        z-index: 2;
    }

    .main-header:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: repeating-linear-gradient(
            45deg,
            rgba(255, 255, 255, 0.05),
            rgba(255, 255, 255, 0.05) 10px,
            rgba(255, 255, 255, 0.1) 10px,
            rgba(255, 255, 255, 0.1) 20px
        );
        z-index: 1;
    }

    /* Conteneur de chat amélioré */
    .chat-container {
        height: 450px;
        overflow-y: auto;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border: none;
        border-radius: 20px;
        background-color: var(--bg-secondary);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        scrollbar-width: thin;
        scrollbar-color: var(--primary-color) var(--bg-primary);
    }

    .chat-container:hover {
        box-shadow: var(--hover-shadow);
    }

    /* Personnalisation des messages */
    .stChatMessage {
        margin-bottom: 1.2rem;
        padding: 0.5rem 0;
    }

    .stChatMessage .stChatMessageContent {
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        max-width: 85%;
    }

    .stChatMessage.user .stChatMessageContent {
        background-color: #E3F2FD;
        border-bottom-right-radius: 5px;
    }

    .stChatMessage.assistant .stChatMessageContent {
        background-color: #F1F8E9;
        border-bottom-left-radius: 5px;
    }

    /* Conteneur de saisie élégant */
    .input-container {
        padding: 1.5rem;
        background-color: var(--bg-secondary);
        border-radius: 20px;
        box-shadow: var(--shadow);
        margin-top: 1.5rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        position: relative;
    }

    .input-container:hover {
        border-color: var(--primary-color);
        box-shadow: var(--hover-shadow);
    }

    .input-container:before {
        content: "🎤 Parlez-moi!";
        position: absolute;
        top: -12px;
        left: 20px;
        background: var(--bg-secondary);
        padding: 0 10px;
        font-size: 0.9rem;
        color: var(--primary-color);
        font-weight: 600;
    }

    /* Indicateur de traitement animé */
    .processing-indicator {
        text-align: center;
        color: var(--primary-color);
        font-weight: 500;
        margin: 1.5rem 0;
        padding: 0.8rem;
        background: rgba(65, 105, 225, 0.1);
        border-radius: 10px;
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }

    /* Footer stylisé */
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
        padding: 1rem;
        border-top: 1px solid var(--border-color);
        background: rgba(255, 255, 255, 0.7);
        border-radius: 0 0 10px 10px;
    }

    /* Boutons améliorés */
    .stButton > button {
        background: var(--gradient-bg);
        border: none;
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 6px rgba(65, 105, 225, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 10px rgba(65, 105, 225, 0.4);
    }

    .stButton > button:active {
        transform: translateY(1px);
    }

    /* Animation pour les uploads */
    .stFileUploader {
        border: 2px dashed var(--primary-color);
        border-radius: 15px;
        padding: 1rem;
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: var(--accent-color);
        background-color: rgba(65, 105, 225, 0.05);
    }

    /* Audio player personnalisé */
    audio {
        width: 100%;
        border-radius: 50px;
        background: var(--bg-primary);
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }

    /* Animations d'entrée */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .chat-container, .input-container, .main-header {
        animation: fadeIn 0.6s ease-out forwards;
    }

    /* Badge de notification */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 50px;
        background-color: var(--accent-color);
        color: white;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Ajout de classe pour le thème sombre si activé
if st.session_state.theme == "dark":
    st.markdown('<div class="dark-theme">', unsafe_allow_html=True)

# Entête principal avec animation et design moderne
st.markdown("""
<div class='main-header'>
    <h1>🎓 Assistant Éducatif Audio <span class="badge">Pro</span></h1>
</div>
""", unsafe_allow_html=True)

# Boutons d'action améliorés
col1, col2 = st.columns([4, 1])
# PARTIE SUPÉRIEURE: AFFICHAGE DE LA CONVERSATION AVEC ANIMATION
for i, message in enumerate(st.session_state.messages):
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["content"])

# Affichage de l'indicateur de traitement si actif
if st.session_state.processing:
    st.markdown("<div class='processing-indicator'>L'assistant analyse votre requête...</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# PARTIE INFÉRIEURE: SAISIE AUDIO AMÉLIORÉE VISUELLEMENT
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Déposez votre enregistrement audio ici (MP3, WAV, etc.)", type=["mp3", "wav", "m4a", "flac"])

if uploaded_file:
    # Affichage des contrôles audio
    st.audio(uploaded_file, format="audio/wav")

    # Bouton de transcription amélioré
    transcribe_button = st.button("✨ Transcrire et Envoyer", use_container_width=True)

    if transcribe_button:
        st.session_state.processing = True

        try:
            # Mise à jour invisible du statut
            with st.spinner(""):
                # Envoi du fichier à l'API pour transcription
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                response = requests.post(f"{API_URL}/transcribe", files=files)

                if response.status_code == 200:
                    result = response.json()
                    transcription = result["text"]

                    # Envoi de la transcription à l'API de chat
                    chat_response = requests.post(
                        f"{API_URL}/message",
                        json={
                            "message": transcription,
                            "session_id": st.session_state.session_id
                        }
                    )

                    if chat_response.status_code == 200:
                        chat_result = chat_response.json()
                        assistant_response = chat_result["response"]

                        # Sauvegarde de l'ID de session si nouvelle session
                        if not st.session_state.session_id:
                            st.session_state.session_id = chat_result["session_id"]

                        # Ajout à l'historique
                        st.session_state.messages.append({"role": "user", "content": transcription})
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    else:
                        st.error(f"Erreur lors de la génération de réponse: {chat_response.text}")
                elif response.status_code == 404:
                    st.error("Erreur: L'endpoint de transcription n'existe pas. Vérifiez que l'API est correctement configurée.")
                else:
                    st.error(f"Erreur lors de la transcription (Code {response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Erreur de connexion: {str(e)}")
        finally:
            st.session_state.processing = False
            st.rerun()

# Bouton "Nouvelle conversation" déplacé ici
if st.button("🔄 Nouvelle conversation"):
    st.session_state.messages = []
    st.session_state.session_id = None
    st.success("Conversation réinitialisée!")
    time.sleep(1)
    st.rerun()

# Tips pour l'utilisateur avec des icônes
st.markdown("""
<div style="margin-top: 1rem; padding: 0.8rem; background-color: rgba(65, 105, 225, 0.1); border-radius: 10px;">
    <h4>💡 Conseils d'utilisation:</h4>
    <ul>
        <li>Parlez clairement et distinctement pour de meilleurs résultats</li>
        <li>Utilisez un microphone externe pour une qualité audio optimale</li>
        <li>Les questions courtes et précises fonctionnent mieux</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Pied de page amélioré
st.markdown(f"""
<div class='footer'>
    <p>© 2025 Assistant Éducatif Audio | Créé avec ❤️ pour apprendre | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)

# Fermeture de la div du thème sombre si activée
if st.session_state.theme == "dark":
    st.markdown('</div>', unsafe_allow_html=True)
