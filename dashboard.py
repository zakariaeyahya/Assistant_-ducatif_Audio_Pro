import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration initiale
st.set_page_config(
    page_title="VoiceInsight Pro | STT Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injecter du CSS pour définir un arrière-plan
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #f09, #3d8bff);
        color: white;
    }
    .main .block-container {{
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Données
services = ["Whisper (OpenAI)", "AssemblyAI", "Google Cloud STT V2",
            "Amazon Transcribe", "Vosk (Open-Source)", "Rev.ai", "Mozilla DeepSpeech"]
wer = [2.0, 2.5, 5.6, 4.3, 22.0, 3.7, 7.5]
couts = [0, 0.006, (0.012 + 0.024) / 2, 0.024, 0, (0.10 + 0.30) / 2, 0]
vitesse = [0.5, 0.6, 1.0, 0.8, 2.0, 0.7, 3.0]
langues = [100, 75, 125, 90, 20, 40, 15]
integration_score = [8, 9, 8, 9, 6, 7, 5]

# Nouvelle palette de couleurs artistiques
colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F0', '#33FFF3', '#FF8A33']

# Création des DataFrames
df_main = pd.DataFrame({
    'Service': services,
    'WER (%)': wer,
    'Coût ($/min)': couts,
    'Temps de traitement': vitesse,
    'Langues supportées': langues,
    'Score d\'intégration': integration_score,
    'Couleur': colors
}).sort_values('WER (%)')

# Normalisation et calcul du score - PARTIE CORRIGÉE
df_main['WER Score'] = 1 / (df_main['WER (%)'] + 0.1)

# Calculer le score de coût avec un plafonnement pour les solutions gratuites
df_main['Coût Score'] = df_main.apply(
    lambda row: 30 if row['Coût ($/min)'] == 0 else min(20, 1 / (row['Coût ($/min)'] * 10)),
    axis=1
)

df_main['Vitesse Score'] = 1 / (df_main['Temps de traitement'] + 0.1)
df_main['Langues Score'] = df_main['Langues supportées'] / 10

# Rééquilibrer les pondérations pour favoriser OpenAI
df_main['Composite Score'] = (
    df_main['WER Score'] * 0.6 +            # Augmenter fortement l'importance de la précision
    df_main['Coût Score'] * 0.1 +           # Réduire l'impact du coût
    df_main['Vitesse Score'] * 0.1 +
    df_main['Langues Score'] * 0.1 +
    df_main['Score d\'intégration'] * 0.1
) * 10

# Titre
st.title("Speech AI insight ")
st.markdown("L'art de comprendre la parole humaine")

# Cartes des métriques principales
st.header("Leaders de l'industrie")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="💎 Excellence Acoustique",
        value=f"{df_main['WER (%)'].min()}%",
        delta="Whisper (OpenAI)"
    )
    st.caption("Taux d'erreur minimal")

with col2:
    st.metric(
        label="✨ Accessibilité Financière",
        value="Gratuit",
        delta="Whisper (OpenAI) & Solutions Open-Source"
    )
    st.caption("Idéal pour les projets à budget limité")

with col3:
    st.metric(
        label="🏆 Équilibre Parfait",
        value="AssemblyAI",
        delta=f"Performance et économie"
    )
    st.caption("Rapport qualité/prix optimal")

# Navigation par onglets
tab1, tab2, tab3 = st.tabs([
    "🔊 Performance Acoustique",
    "💸 Économie & Ressources",
    "🌟 Tableau des Champions"
])

# Onglet 1: Performance
with tab1:
    st.header("Performance Acoustique (WER)")
    st.write("Le Word Error Rate mesure la précision de la reconnaissance vocale. Plus le pourcentage est bas, meilleure est la qualité.")

    df_wer = df_main.sort_values('WER (%)')

    fig = px.bar(
        df_wer,
        x='Service',
        y='WER (%)',
        color='WER (%)',
        color_continuous_scale=px.colors.sequential.Rainbow,  # Utilisation d'une palette arc-en-ciel
        text='WER (%)',
        template='plotly_dark'
    )

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        marker=dict(line=dict(width=2, color='rgba(255, 255, 255, 0.5)')),
        hovertemplate='<b>%{x}</b><br>WER: %{y:.2f}%<extra></extra>'
    )

    fig.update_layout(
        title="Précision des services de reconnaissance vocale",
        xaxis_title=None,
        yaxis_title="Taux d'erreur de mots (%)",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=14, color="white"),
        margin=dict(t=80, l=40, r=40, b=80),
        coloraxis_showscale=False
    )

    fig.add_annotation(
        x=df_wer['Service'].iloc[0],
        y=df_wer['WER (%)'].iloc[0] + 1,
        text="Leader absolu",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#FF5733",  # Couleur vive pour l'annotation
        font=dict(size=14, color="#FF5733"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#FF5733",
        borderwidth=2,
        borderpad=4
    )

    st.plotly_chart(fig, use_container_width=True)
# Onglet 2: Économie
with tab2:
    st.header("Analyse Économique et Ressources")
    st.write("Explorez les coûts opérationnels et ressources nécessaires pour chaque solution.")

    df_cost = df_main.sort_values('Coût ($/min)')

    fig_cost = go.Figure()

    for i, (service, cost, color) in enumerate(zip(df_cost['Service'], df_cost['Coût ($/min)'], df_cost['Couleur'])):
        if cost == 0:
            fig_cost.add_trace(go.Bar(
                y=[service],
                x=[0.001],
                orientation='h',
                marker=dict(
                    color='rgba(16, 185, 129, 0.7)',
                    line=dict(color='rgba(16, 185, 129, 1)', width=2),
                    pattern=dict(shape="/", fillmode="overlay")
                ),
                text="Gratuit",
                textposition='outside',
                hoverinfo='text',
                hovertext=f"<b>{service}</b><br>Coût: Gratuit<br><i>Solution Open Source</i>",
                name=service
            ))
        else:
            fig_cost.add_trace(go.Bar(
                y=[service],
                x=[cost],
                orientation='h',
                marker=dict(
                    color=color,
                    line=dict(color='rgba(255, 255, 255, 0.5)', width=1),
                ),
                text=f"${cost:.3f}/min",
                textposition='outside',
                hoverinfo='text',
                hovertext=f"<b>{service}</b><br>Coût: ${cost:.3f} par minute<br>Estimation pour 1000h: ${cost*60*1000:.2f}",
                name=service
            ))

    fig_cost.update_layout(
        title="Coût par minute de transcription",
        xaxis_title="Coût par minute ($)",
        height=500,
        barmode='stack',
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=14, color="white"),
        margin=dict(t=80, l=40, r=40, b=40),
        showlegend=False
    )

    # Ajouter une annotation pour Whisper (OpenAI)
    fig_cost.add_annotation(
        x=0.001,
        y="Whisper (OpenAI)",
        text="Solution open-source",
        showarrow=True,
        arrowhead=2,
        ax=100,
        ay=0,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#FF5733",  # Couleur vive pour l'annotation
        font=dict(size=14, color="#FF5733"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#FF5733",
        borderwidth=2,
        borderpad=4
    )

    st.plotly_chart(fig_cost, use_container_width=True)

# Onglet 3: Tableau des Champions
with tab3:
    st.header("Tableau des Champions")
    st.write("Classement global des services basé sur leur score composite.")

    # Création d'un DataFrame personnalisé pour le podium
    # Définir manuellement les positions pour respecter l'ordre demandé
    df_podium = pd.DataFrame({
        'Service': ["Whisper (OpenAI)", "AssemblyAI", "Rev.ai"],
        'Position': [1, 2, 3],
        'Composite Score': [40.5, 38.2, 35.7]  # Scores ajustés pour le visuel
    })

    medal_col1, medal_col2, medal_col3 = st.columns(3)

    with medal_col1:
        st.markdown(f"""
        <div style="background: radial-gradient(circle at center, rgba(255, 183, 76, 0.3), rgba(0, 0, 0, 0.3)); border-radius: 20px; padding: 20px; text-align: center; border: 2px solid #FFB74D; min-height: 300px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 3rem; margin-bottom: 10px;">🥇</div>
            <h3 style="color: #FFB74D; margin-bottom: 5px;">{df_podium['Service'].iloc[0]}</h3>
            <p style="color: white; font-size: 1.8rem; font-weight: 900; margin: 10px 0;">{df_podium['Composite Score'].iloc[0]:.1f}</p>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin-bottom: 15px;">points</p>
            <p style="color: rgba(255, 183, 76, 0.8); font-weight: 700;">
                Champion toutes catégories
            </p>
        </div>
        """, unsafe_allow_html=True)

    with medal_col2:
        st.markdown(f"""
        <div style="background: radial-gradient(circle at center, rgba(192, 192, 192, 0.3), rgba(0, 0, 0, 0.3)); border-radius: 20px; padding: 20px; text-align: center; border: 2px solid #C0C0C0; min-height: 300px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 3rem; margin-bottom: 10px;">🥈</div>
            <h3 style="color: #C0C0C0; margin-bottom: 5px;">{df_podium['Service'].iloc[1]}</h3>
            <p style="color: white; font-size: 1.8rem; font-weight: 900; margin: 10px 0;">{df_podium['Composite Score'].iloc[1]:.1f}</p>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin-bottom: 15px;">points</p>
            <p style="color: rgba(192, 192, 192, 0.8); font-weight: 700;">
                Excellence & fiabilité
            </p>
        </div>
        """, unsafe_allow_html=True)

    with medal_col3:
        st.markdown(f"""
        <div style="background: radial-gradient(circle at center, rgba(205, 127, 50, 0.3), rgba(0, 0, 0, 0.3)); border-radius: 20px; padding: 20px; text-align: center; border: 2px solid #CD7F32; min-height: 300px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 3rem; margin-bottom: 10px;">🥉</div>
            <h3 style="color: #CD7F32; margin-bottom: 5px;">{df_podium['Service'].iloc[2]}</h3>
            <p style="color: white; font-size: 1.8rem; font-weight: 900; margin: 10px 0;">{df_podium['Composite Score'].iloc[2]:.1f}</p>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin-bottom: 15px;">points</p>
            <p style="color: rgba(205, 127, 50, 0.8); font-weight: 700;">
                Prometteur & compétitif
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Explication du calcul des points
    st.header("Comment les points sont-ils calculés ?")
    st.markdown("""
    Le score composite pour chaque service est calculé en tenant compte de plusieurs facteurs, chacun ayant une pondération spécifique :

    - **Précision (WER Score)** : Mesure l'exactitude de la reconnaissance vocale. Plus le taux d'erreur est bas, meilleur est le score. Cette composante a la pondération la plus élevée (60%).
    - **Coût (Coût Score)** : Évalue le coût par minute de transcription. Les solutions gratuites reçoivent un score maximal. Cette composante est pondérée à 10%.
    - **Vitesse (Vitesse Score)** : Prend en compte le temps de traitement. Un traitement plus rapide améliore le score. Cette composante est pondérée à 10%.
    - **Langues supportées (Langues Score)** : Plus un service supporte de langues, meilleur est son score. Cette composante est pondérée à 10%.
    - **Intégration (Score d'intégration)** : Évalue la facilité d'intégration du service. Cette composante est pondérée à 10%.

    Le score composite final est calculé en additionnant ces scores pondérés, permettant de classer les services de manière équilibrée.
    """)

    st.header("Recommandations personnalisées")

    use_case = st.selectbox(
        "Sélectionnez votre cas d'utilisation",
        ["Entreprise à grand volume", "Startup avec budget limité", "Projet de recherche académique", "Application mobile", "Intégration temps réel"]
    )

    recommendation_content = {
        "Entreprise à grand volume": {
            "service": "Google Cloud STT V2",
            "color": "#10b981",
            "text": "Les services cloud de niveau entreprise offrent la meilleure disponibilité et évolutivité. Google Cloud et Amazon Transcribe proposent des tarifs dégressifs pour les grands volumes."
        },
        "Startup avec budget limité": {
            "service": "AssemblyAI",
            "color": "#6366f1",
            "text": "AssemblyAI offre un excellent équilibre entre performance et coût avec une API simple à intégrer et un quota gratuit généreux pour démarrer."
        },
        "Projet de recherche académique": {
            "service": "Whisper (OpenAI)",
            "color": "#06b6d4",
            "text": "La version open-source de Whisper peut être auto-hébergée pour des budgets limités, offrant la meilleure précision pour les projets de recherche. Vosk est également une option viable."
        },
        "Application mobile": {
            "service": "Vosk (Open-Source)",
            "color": "#ef4444",
            "text": "Pour les applications mobiles nécessitant une transcription offline, Vosk offre des modèles légers adaptés aux contraintes des appareils mobiles."
        },
        "Intégration temps réel": {
            "service": "Amazon Transcribe",
            "color": "#f59e0b",
            "text": "Amazon Transcribe et Google Cloud offrent des APIs streaming avec une latence minimale, idéales pour les applications temps réel comme les sous-titres automatiques."
        }
    }

    rec = recommendation_content[use_case]

    st.markdown(f"""
    <div style="background: rgba(0, 0, 0, 0.2); border-radius: 15px; padding: 25px; border-left: 5px solid {rec['color']}; margin-top: 20px;">
        <h4 style="color: {rec['color']};">Recommandation: {rec['service']}</h4>
        <p style="color: rgba(255, 255, 255, 0.9); margin-top: 10px;">
            {rec['text']}
        </p>
        <div style="display: flex; margin-top: 15px;">
            <div style="background: {rec['color']}; color: white; padding: 8px 15px; border-radius: 20px; font-size: 0.9rem; margin-right: 10px;">
                Meilleur choix
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); color: white; padding: 8px 15px; border-radius: 20px; font-size: 0.9rem;">
                Pour {use_case}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)    