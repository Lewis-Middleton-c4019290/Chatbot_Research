import os
import sys
import numpy as np
import pickle
import time
import pygame

current_dir = os.path.dirname(os.path.abspath(__file__))  
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir)) 

try:
    from gym_torcs import TorcsEnv
except ImportError:
    from gym_environment import TorcsEnv

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Torcs Controler")

def get_human_input():
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    
    steer = 0.0
    throttle = 0.0
    brake = 0.0
    
    # Steering: A/Left = Positive, D/Right = Negative
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        steer = 0.5
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        steer = -0.5
    
    # Accelerate: W or Up
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        throttle = 1.0
        
    # Brake: S or Down
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        brake = 1.0 
        
    return steer, throttle, brake

def save_data(observations, actions, path):
    """Saves everything needed for Imitation or RL Pre-training."""
    if len(observations) > 0:
        with open(path, "wb") as f:
            pickle.dump({
                "obs": np.array(observations, dtype=np.float32), 
                "acts": np.array(actions, dtype=np.float32)
            }, f)
        print(f"\n>>> SAVED TO {os.path.basename(path)} | Total Steps: {len(observations)}")

def run_collection():

    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    observations, actions = [], []
    save_path = os.path.join(current_dir, "pro_driving.pkl")
    
    try:
        obs = env.reset()
        print("\n" + "="*40)
        print("COLLECTING DRIVING DATA")
        print(f"File: {os.path.basename(save_path)}")
        print("CONTROLS: WASD or ARROWS | QUIT: Q or ESC")
        print("="*40 + "\n")
        
        running = True
        while running:
            # 1. Capture ALL THREE values from keyboard
            steer, throttle, brake = get_human_input()
            
            # 2. Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:  # Quit on Q or ESC and save data
                        running = False

            # 3. Create the 3-value action array: [Steer, Throttle, Brake]
            act = np.array([steer, throttle, brake], dtype=np.float32)
            
            # 4. Step the environment
            next_obs, reward, done, info = env.step(act)
            
            # 5. Record observation and the 3-part action
            observations.append(obs)
            actions.append(act)
            
            obs = next_obs

            # Auto-save every 2000 steps
            if len(observations) % 2000 == 0:
                save_data(observations, actions, save_path)
            
            if len(observations) % 100 == 0:
                print(f"Steps Captured: {len(observations)} | Speed: {obs[31]:.1f} km/h", end="\r")

            if done:
                print("\nEnvironment signaled 'Done' (Crash or Track End).")
                running = False

        save_data(observations, actions, save_path)

    except Exception as e:
        print(f"\nError during collection: {e}")
    finally:
        env.end()
        pygame.quit()

if __name__ == "__main__":
    os.system("pkill -9 torcs")
    time.sleep(1)
    run_collection()