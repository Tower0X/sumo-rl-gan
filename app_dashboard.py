import streamlit as st
import time
import threading
import numpy as np
import pandas as pd
from shared_state import state
from sim_runner import run_simulation
from sumo_rl.environment.attack_controller import AttackType

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
    state.target_node_id = st.sidebar.selectbox(
        "🎯 Intersection Cible :",
        state.available_nodes,
        index=state.available_nodes.index(state.target_node_id) if state.target_node_id in state.available_nodes else 0,
        disabled=not state.running
    )

# Configuration d'Attaque Manuelle (Stress Test)
if state.mode == "manual_attack":
    st.sidebar.subheader("⚔️ Injection d'Attaque")
    atk_type_str = st.sidebar.selectbox(
        "Type de perturbation :",
        (
            "Aucune", 
            "Brouilleur (Coupure Totale)",
            "DoS - Flooding (Saturation)", 
            "DoS - Slowloris (Lag Persistant)",
            "Sybil - Ghost Vehicles (Densité)",
            "Sybil - Position Jitter (Cinématique)",
            "Temporal DoS (Latence Variable)",
            "Data Poisoning (Capteurs)"
        )
    )
    
    # Mapping des noms UI vers l'Enum AttackType
    mapping = {
        "Aucune": AttackType.NONE,
        "Brouilleur (Coupure Totale)": AttackType.JAMMER,
        "DoS - Flooding (Saturation)": AttackType.FLOODING_DDOS,
        "DoS - Slowloris (Lag Persistant)": AttackType.SLOWLORIS_DDOS,
        "Sybil - Ghost Vehicles (Densité)": AttackType.GHOST_VEHICLES,
        "Sybil - Position Jitter (Cinématique)": AttackType.POSITION_JITTER,
        "Temporal DoS (Latence Variable)": AttackType.TEMPORAL_DOS,
        "Data Poisoning (Capteurs)": AttackType.DATA_POISONING
    }
    state.manual_attack_type = mapping.get(atk_type_str, AttackType.NONE)
        
    virulence = st.sidebar.slider(
        "Virulence de l'Attaque :",
        min_value=0, max_value=100, value=50, step=5,
        format="%d%%"
    )
    state.manual_virulence = virulence / 100.0

st.sidebar.divider()
st.sidebar.markdown("**INF4258 - GROUPE 10**")
st.sidebar.caption("Université de Yaoundé I")

# 3. Main Frame - Dashboard
st.markdown('# 📶 Supervision de Résilience VANET <div class="badge-advanced">Advanced Temporal Intelligence (LSTM)</div>', unsafe_allow_html=True)
st.markdown("Auditez la robustesse du contrôleur de trafic IA (Multi-Agent) face aux attaques injectées en temps réel sur une grille urbaine 4x4.")

# Affichage des cartes de télémétrie en temps réel
col1, col2, col3, col4 = st.columns(4)

# Évaluation du statut de sécurité
with state.lock:
    atk_name = state.active_attack_name
    atk_intensity = state.active_attack_intensity
    curr_reward = state.current_reward
    sim_step = state.step
    running_status = state.running

# Carte 1: Statut de la Simulation
with col1:
    if running_status:
        st.markdown(f'<div class="metric-card"><div>Statut Moteur</div><div class="metric-val color-healthy">● RUNNING</div><div style="font-size:0.85rem;color:#8b949e;">Step: {sim_step}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><div>Statut Moteur</div><div class="metric-val" style="color:#8b949e;">○ STOPPED</div><div style="font-size:0.85rem;color:#8b949e;">Aucun processus</div></div>', unsafe_allow_html=True)

# Carte 2: Type d'Attaque active
with col2:
    if atk_name == "Aucune":
        st.markdown('<div class="metric-card"><div>Sécurité Réseau</div><div class="metric-val color-healthy">Sain</div><div style="font-size:0.85rem;color:#8b949e;">Aucune menace décelée</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="metric-card"><div>Sécurité Réseau</div><div class="metric-val color-danger">{atk_name}</div><div style="font-size:0.85rem;color:#8b949e;">Attaque active</div></div>', unsafe_allow_html=True)

# Carte 3: Virulence de l'Attaque
with col3:
    if atk_name == "Aucune":
        st.markdown('<div class="metric-card"><div>Force Perturbation</div><div class="metric-val" style="color:#8b949e;">0%</div><div style="font-size:0.85rem;color:#8b949e;">Canal intègre</div></div>', unsafe_allow_html=True)
    else:
        color_class = "color-warning" if atk_intensity < 0.6 else "color-danger"
        st.markdown(f'<div class="metric-card"><div>Force Perturbation</div><div class="metric-val {color_class}">{atk_intensity*100:.0f}%</div><div style="font-size:0.85rem;color:#8b949e;">Virulence réseau</div></div>', unsafe_allow_html=True)

# Carte 4: Résilience (Récompense)
with col4:
    if not running_status:
        st.markdown('<div class="metric-card"><div>Score de Résilience</div><div class="metric-val" style="color:#8b949e;">-</div><div style="font-size:0.85rem;color:#8b949e;">Attente données</div></div>', unsafe_allow_html=True)
    else:
        if curr_reward >= -1.5:
            color_class = "color-healthy"
            status_desc = "Optimal (Pas d'urgence)"
        elif curr_reward >= -5.0:
            color_class = "color-warning"
            status_desc = "Mode Dégradé / Ralentissement"
        else:
            color_class = "color-danger"
            status_desc = "Fail-Safe Actif / Freinage Urgent"
            
        st.markdown(f'<div class="metric-card"><div>Score de Résilience</div><div class="metric-val {color_class}">{curr_reward:+.2f}</div><div style="font-size:0.85rem;color:#8b949e;">{status_desc}</div></div>', unsafe_allow_html=True)

st.write("")

# 4. Visualisation Graphique (Charts)
if running_status:
    # Récupérer l'historique
    with state.lock:
        rewards = list(state.reward_history)
        waiting_times = list(state.waiting_time_history)
        latencies = list(state.latency_history)
        queues = list(state.queue_history)

    # Création du DataFrame pour l'affichage des courbes
    df_metrics = pd.DataFrame({
        "Pas": np.arange(len(rewards)),
        "Récompense PPO (Stabilité)": rewards,
        "Temps d'attente système (s)": waiting_times,
        "Latence V2X (ms)": latencies,
        "Véhicules en attente (Queues)": queues
    })

    tab_graf, tab_sys = st.tabs(["📡 Télémétrie Graphique", "⚔️ Analyse du Duel cGAN"])
    
    with tab_graf:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("### Performance & Sécurité Routière")
            # Graphique combiné Waiting Time et File d'attente
            st.line_chart(df_metrics, x="Pas", y=["Temps d'attente système (s)", "Véhicules en attente (Queues)"], color=["#58a6ff", "#f85149"])
            
        with col_g2:
            st.markdown("### Récompense & Santé Télécoms")
            # Graphique combiné Récompense et Latence
            st.line_chart(df_metrics, x="Pas", y=["Récompense PPO (Stabilité)", "Latence V2X (ms)"], color=["#2ea043", "#d29922"])

    with tab_sys:
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("### Cartographie du Comportement du GAN")
            st.markdown("""
            En mode **Duel Adversarial**, le GAN génère des attaques stochastiques pour identifier les faiblesses du PPO :
            *   Le **DoS Temporel** cherche à rendre l'agent aveugle sur les temps de changement.
            *   Le **Data Poisoning** dissimule la taille réelle des files d'attente.
            *   L'injection de **Véhicules Fantômes** provoque de fausses ouvertures de voies.
            """)
            if len(rewards) > 0:
                st.info(f"Dégâts moyens infligés par le GAN au cours de ce run : {-np.mean(rewards):.2f}")
                
        with col_s2:
            st.markdown("### Réponse de l'Agent PPO")
            st.markdown("""
            Face à une panne ou à une latence réseau supérieure à 80%, la récompense de l'agent RL se reconfigure en mode **Fail-Safe** :
            *   Transition immédiate d'une logique de fluidité vers une logique de sécurité.
            *   Déstockage régulier des voies pour éviter le blocage.
            """)
            if len(queues) > 0:
                st.success(f"Nombre maximum de voitures bloquées simultanément : {np.max(queues)}")

# 5. Zone de Logs
st.markdown("### 📜 Journal des Événements")
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
