import sumo_rl
import gymnasium as gym

def run_sumo_rl_demo():
    print("[*] Lancement d'une simulation classique via SUMO-RL...")
    
    # On utilise un environnement pré-défini de sumo-rl (TSC : Traffic Signal Control)
    env = sumo_rl.SumoEnvironment(
        net_file='sumo_rl/nets/2way-single-intersection/single-intersection.net.xml',
        route_file='sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
        use_gui=True,
        num_seconds=1000,
        delta_time=5, # Temps entre chaque décision RL (5 secondes)
        single_agent=True
    )

    obs, info = env.reset()
    done = False
    
    print("[*] Simulation en cours... Observez les feux de circulation changer via l'IA brute.")
    
    step = 0
    while not done and step < 50:
        # Ici l'IA choisit quelle phase passer au vert (action aléatoire pour la démo)
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            print(f"Pas {step} | Récompense (Réduction de l'attente): {reward:.2f}")
            
        done = terminated or truncated
        step += 1

    env.close()
    print("[*] Démonstration terminée.")

if __name__ == "__main__":
    run_sumo_rl_demo()
