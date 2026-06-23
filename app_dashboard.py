import streamlit as st
import time
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shared_state import state
from sim_runner import run_simulation
from sumo_rl.environment.attack_controller import AttackType, ATTACK_FAMILY_LABELS

# Source de verite unique pour la segmentation des familles (label -> Enum).
# Ordonne pour presenter les familles de la plus brutale a la plus subtile.
ATTACK_FAMILY_ORDER = [
    AttackType.NONE,
    AttackType.JAMMER,
    AttackType.FLOODING_DDOS,
    AttackType.SLOWLORIS_DDOS,
    AttackType.GHOST_VEHICLES,
    AttackType.POSITION_JITTER,
    AttackType.TEMPORAL_DOS,
    AttackType.DATA_POISONING,
]
LABEL_TO_ATTACK = {ATTACK_FAMILY_LABELS[a]: a for a in ATTACK_FAMILY_ORDER}
ATTACK_LABELS = [ATTACK_FAMILY_LABELS[a] for a in ATTACK_FAMILY_ORDER]

# 1. Configuration de la page
st.set_page_config(
    page_title="VANET Security Lab - Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé pour l'effet "SOC Dashboard" haut de gamme
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .badge-advanced {
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    /* Title and Headers Styling */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        color: #ffffff !important;
    }
    
    /* Custom Card Style */
    .metric-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: bold;
        margin-top: 10px;
    }
    
    /* Colors for metrics */
    .color-healthy { color: #2ea043; }
    .color-warning { color: #d29922; }
    .color-danger { color: #f85149; }
    .color-info { color: #58a6ff; }
    
    /* Log console styling */
    .log-box {
        background-color: #010409;
        border: 1px solid #30363d;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        height: 250px;
        overflow-y: scroll;
        padding: 15px;
        margin-top: 10px;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Sidebar - Configuration et Commandes
st.sidebar.markdown("# 🛡️ VANET Security Lab")
st.sidebar.markdown("### *Dashboard de Résilience Cyber-Physique*")
st.sidebar.divider()

# Contrôles de simulation
st.sidebar.subheader("🕹️ Contrôle Simulation")
use_gui = st.sidebar.toggle("Afficher SUMO-GUI (Fenêtre 3D)", value=False)

col_start, col_stop = st.sidebar.columns(2)
with col_start:
    if st.button("▶️ Démarrer", use_container_width=True, disabled=state.running):
        state.use_gui = use_gui
        t = threading.Thread(target=run_simulation)
        t.daemon = True
        t.start()
        time.sleep(0.8)  # Laisser le temps à SUMO d'initier TraCI
        st.rerun()

with col_stop:
    if st.button("⏹️ Arrêter", use_container_width=True, disabled=not state.running):
        state.should_stop = True
        time.sleep(0.5)
        st.rerun()

st.sidebar.divider()

# Sélection de la ville
st.sidebar.subheader("🌍 Environnement / Ville")
cities = ["Ville par défaut (Grille 4x4)", "Ville A (Grille 2x2)", "Ville B (Grille 3x2)"]
selected_city = st.sidebar.selectbox(
    "Charger la topologie :",
    cities,
    index=cities.index(state.city_map) if state.city_map in cities else 0,
    disabled=state.running
)
state.city_map = selected_city

st.sidebar.divider()

# Choix du mode d'opération
st.sidebar.subheader("🛡️ Mode Opérationnel")
mode_choice = st.sidebar.radio(
    "Sélectionnez le mode :",
    ("Défense Seule (Trafic Sain)", "Attaque Manuelle (Stress Test)", "Duel Adversarial (cGAN vs PPO)"),
    disabled=not state.running
)

# Traduction du choix du mode vers l'état partagé
if mode_choice == "Défense Seule (Trafic Sain)":
    state.mode = "defense_only"
elif mode_choice == "Attaque Manuelle (Stress Test)":
    state.mode = "manual_attack"
else:
    state.mode = "adversarial_gan"

# Attribution des cibles (Intersections)
if state.available_nodes:
    target_options = ["Toutes (GLOBAL)"] + state.available_nodes
    
    # Trouver l'index actuel, default = 0
    if state.target_node_id in target_options:
        idx = target_options.index(state.target_node_id)
    elif state.available_nodes and state.target_node_id in state.available_nodes:
        idx = target_options.index(state.target_node_id)
    else:
        idx = 0
        
    state.target_node_id = st.sidebar.selectbox(
        "🎯 Intersection Cible :",
        target_options,
        index=idx,
        disabled=not state.running
    )

# Configuration d'Attaque Manuelle (Stress Test) -- Segmentation 8 familles
if state.mode == "manual_attack":
    st.sidebar.subheader("⚔️ Injection d'Attaque (8 familles)")
    atk_type_str = st.sidebar.selectbox(
        "Famille d'attaque :",
        ATTACK_LABELS,
    )
    state.manual_attack_type = LABEL_TO_ATTACK.get(atk_type_str, AttackType.NONE)

    virulence = st.sidebar.slider(
        "Virulence de l'Attaque :",
        min_value=0, max_value=100, value=50, step=5,
        format="%d%%"
    )
    state.manual_virulence = virulence / 100.0

    # Aide contextuelle: effet physique attendu de la famille selectionnee.
    _physical_hint = {
        AttackType.GHOST_VEHICLES: "➡️ Injecte de VRAIS véhicules fantômes ROUGES (visibles en GUI).",
        AttackType.POSITION_JITTER: "➡️ Injecte des véhicules fantômes ROUGES + bruit cinématique.",
        AttackType.JAMMER: "➡️ Gèle la phase du feu (l'agent perd le contrôle physique).",
        AttackType.FLOODING_DDOS: "➡️ Gel intermittent de la phase + saturation latence.",
        AttackType.SLOWLORIS_DDOS: "➡️ Gel intermittent + lag persistant.",
        AttackType.TEMPORAL_DOS: "➡️ Gel intermittent + latence variable.",
        AttackType.DATA_POISONING: "➡️ Purement perceptuel: cache les files réelles.",
    }
    if state.manual_attack_type in _physical_hint:
        st.sidebar.caption(_physical_hint[state.manual_attack_type])
elif state.mode == "adversarial_gan":
    # En duel, on peut aussi contraindre la famille observée par l'orchestrateur.
    st.sidebar.subheader("⚔️ Duel Adversarial")
    st.sidebar.caption("L'attaquant surrogate choisit la famille; la cible reste réglable ci-dessus.")

st.sidebar.divider()
st.sidebar.markdown("**INF4258 - GROUPE 10**")
st.sidebar.markdown("""
<div style="text-align:center;">
    <h1 style="color:#58a6ff;margin-bottom:0px;">Bouclier MARL</h1>
    <p style="color:#8b949e;font-size:0.9rem;">Défense Proactive</p>
</div>
""", unsafe_allow_html=True)

# 3. Main Frame - Dashboard
st.markdown('# 📶 Supervision de Résilience VANET', unsafe_allow_html=True)
st.markdown("Auditez la robustesse du contrôleur de trafic IA (Multi-Agent) face aux attaques injectées en temps réel sur une grille urbaine 4x4.")

# Affichage des cartes de télémétrie en temps réel
col1, col2, col3, col4 = st.columns(4)

# Évaluation du statut de sécurité
with state.lock:
    atk_name = state.active_attack_name
    atk_intensity = state.active_attack_intensity
    curr_reward = state.current_reward
    target_reward = state.target_reward
    sim_step = state.step
    running_status = state.running
    total_ghosts = state.total_ghosts_spawned
    phase_frozen = state.phase_frozen
    target_comm = state.target_comm_ok
    n_atk = state.n_intersections_attacked

# Carte 1: Statut de la Simulation
with col1:
    if running_status:
        st.markdown(f'<div class="metric-card"><div>Statut Moteur</div><div class="metric-val color-healthy">● RUNNING</div><div style="font-size:0.85rem;color:#8b949e;">Step: {sim_step}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><div>Statut Moteur</div><div class="metric-val" style="color:#8b949e;">○ STOPPED</div><div style="font-size:0.85rem;color:#8b949e;">Aucun processus</div></div>', unsafe_allow_html=True)

# Carte 2: Type d'Attaque active
with col2:
    if atk_name == "Aucune":
        st.markdown('<div class="metric-card"><div>Sécurité Réseau</div><div class="metric-val color-healthy">✅ Sain</div><div style="font-size:0.85rem;color:#8b949e;">Aucune menace décelée</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="metric-card" style="border-color:#f85149;"><div>⚠️ Sécurité Réseau</div><div class="metric-val color-danger" style="font-size:1.5rem;">{atk_name}</div><div style="font-size:0.85rem;color:#f85149;">🔴 ATTAQUE EN COURS</div></div>', unsafe_allow_html=True)
# Carte 3: Canal V2X (comm_ok)
with col3:
    if target_comm:
        st.markdown('<div class="metric-card"><div>Canal V2X (Cible)</div><div class="metric-val color-healthy">📶 CONNECTÉ</div><div style="font-size:0.85rem;color:#8b949e;">Communication intègre</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card" style="border-color:#f85149;"><div>Canal V2X (Cible)</div><div class="metric-val color-danger">📵 COUPÉ</div><div style="font-size:0.85rem;color:#f85149;">Fail-Safe activé</div></div>', unsafe_allow_html=True)

# Carte 4: Résilience (Récompense)
with col4:
    if not running_status:
        st.markdown('<div class="metric-card"><div>Score de Résilience</div><div class="metric-val" style="color:#8b949e;">-</div><div style="font-size:0.85rem;color:#8b949e;">Attente données</div></div>', unsafe_allow_html=True)
    else:
        if curr_reward >= -1.5:
            color_class = "color-healthy"
            status_desc = "Trafic Fluide"
        elif curr_reward >= -5.0:
            color_class = "color-warning"
            status_desc = "Mode Dégradé"
        else:
            color_class = "color-danger"
            status_desc = "Fail-Safe Actif"
            
        st.markdown(f'<div class="metric-card"><div>Résilience (Moy / Cible)</div><div class="metric-val {color_class}" style="font-size:1.8rem;">{curr_reward:+.1f} / {target_reward:+.1f}</div><div style="font-size:0.85rem;color:#8b949e;">{status_desc}</div></div>', unsafe_allow_html=True)

# Carte 5 (pleine largeur): Impact Physique reel du canal TraCI (Lot E)
if running_status and atk_name != "Aucune":
    freeze_txt = "🔒 Phase GELÉE" if phase_frozen else "Phase libre"
    st.markdown(
        f'<div class="metric-card" style="margin-top:10px;">'
        f'<div>Impact Physique sur SUMO (canal TraCI)</div>'
        f'<div class="metric-val color-danger">🚗 {total_ghosts} fantômes ROUGES injectés</div>'
        f'<div style="font-size:0.85rem;color:#8b949e;">{freeze_txt} · effet réel sur le trafic</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.write("")

# 4. Visualisation Graphique (Charts)
if running_status:
    # Récupérer l'historique
    with state.lock:
        rewards = list(state.reward_history)
        target_rewards = list(state.target_reward_history)
        waiting_times = list(state.waiting_time_history)
        latencies = list(state.latency_history)
        queues = list(state.queue_history)

    # Création du DataFrame pour l'affichage des courbes
    df_metrics = pd.DataFrame({
        "Pas": np.arange(len(rewards)),
        "Récompense Moyenne (16 agents)": rewards,
        "Récompense Cible (intersection)": target_rewards[:len(rewards)],
        "Temps d'attente système (s)": waiting_times,
        "Latence V2X (ms)": latencies,
        "Véhicules en attente (Queues)": queues
    })

    tab_graf, tab_sys = st.tabs(["📡 Télémétrie Graphique", "⚔️ Analyse du Duel cGAN"])
    
    with tab_graf:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("### Performance & Sécurité Routière")
            st.line_chart(df_metrics, x="Pas", y=["Temps d'attente système (s)", "Véhicules en attente (Queues)"], color=["#58a6ff", "#f85149"])
            
        with col_g2:
            st.markdown("### Récompense & Santé Télécoms")
            st.line_chart(df_metrics, x="Pas", y=["Récompense Moyenne (16 agents)", "Récompense Cible (intersection)", "Latence V2X (ms)"], color=["#2ea043", "#bc8cff", "#d29922"])

    with tab_sys:
        st.markdown("### 🕸️ Threat Surface (Radar SOC)")
        
        # Fake values based on current attack for the radar chart
        radar_values = [0.1, 0.1, 0.1, 0.1, 0.1]
        with state.lock:
            atk_type = state.active_attack_name
            intensity = state.active_attack_intensity
            
        if "Jammer" in atk_type or "Brouilleur" in atk_type:
            radar_values = [0.9, 0.1, 0.9, 0.8, 0.2]
        elif "DoS" in atk_type or "Flooding" in atk_type:
            radar_values = [0.8, 0.2, 0.5, 0.9, 0.1]
        elif "Sybil" in atk_type or "Ghost" in atk_type or "Jitter" in atk_type:
            radar_values = [0.2, 0.9, 0.4, 0.3, 0.8]
        elif "Poisoning" in atk_type:
            radar_values = [0.1, 0.8, 0.8, 0.1, 0.5]
        elif "cGAN" in atk_type or "Multi-Agents" in atk_type or "Adversarial" in atk_type:
            import random
            radar_values = [random.uniform(0.6, 1.0) for _ in range(5)]
            
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=['Latence V2X', 'Falsification Capteurs', 'Disponibilité (DoS)', 'Saturation TraCI', 'Anomalie Cinématique'],
            fill='toself',
            name='Empreinte de la Menace',
            line_color='#f85149',
            fillcolor='rgba(248, 81, 73, 0.3)'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='#30363d', tickfont=dict(color='#8b949e')),
                angularaxis=dict(gridcolor='#30363d', tickfont=dict(color='#c9d1d9'))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if "cGAN" in atk_type or "Multi-Agents" in atk_type or "Adversarial" in atk_type:
            with st.expander("ℹ️ Comprendre l'attaque cGAN (Curriculum GAN)", expanded=True):
                st.markdown("""
                **Que fait le cGAN ?**
                Le cGAN est un réseau de neurones attaquant entraîné pour vaincre notre IA de contrôle (PPO). Contrairement aux autres attaques (fixes), le cGAN observe en temps réel la distribution du trafic et la politique de l'agent.
                
                **Ses attaques :**
                Il génère un vecteur d'attaque dynamique (bruit malveillant) injecté directement dans l'espace d'observation du contrôleur. Son objectif est de tromper l'IA pour qu'elle prenne les pires décisions possibles (ex: donner le feu vert à une voie vide tout en bloquant une voie surchargée).
                
                **Réaction de la défense :**
                Notre agent de contrôle a subi un **Curriculum Adversarial Training**. Il a appris à reconnaître ces perturbations (vecteurs bruités) et à les filtrer implicitement pour rétablir une matrice de décision saine, évitant ainsi le crash de l'infrastructure et maintenant le trafic fluide.
                """)
        
# 5. Zone de Logs (SIEM / EDR)
st.markdown("### 📜 Logs d'Intrusion (SIEM)")
with state.lock:
    logs_list = list(state.logs)

log_text = ""
for l in reversed(logs_list):
    prefix = "[INFO] "
    if l["type"] == "warning":
        prefix = "⚠️ [ALERT] "
    elif l["type"] == "error":
        prefix = "❌ [ERROR] "
    log_text += f"Pas {l['step']:03d} | {prefix}{l['text']}\n"

if not log_text:
    log_text = "En attente du lancement de la simulation..."

st.markdown(f'<div class="log-box">{log_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

# 6. Auto-Refresh de Streamlit
if running_status:
    time.sleep(0.5)
    st.rerun()
