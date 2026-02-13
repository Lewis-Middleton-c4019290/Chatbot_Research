import os
import numpy as np
import ollama
import pygame
# ... (Keep your existing TorcsEnv imports here)

# 1. THE DATA TRANSLATOR
def get_engineer_briefing(obs):
    """
    Translates raw TORCS numbers into a sentence for the AI.
    TORCS obs indices (standard gym_torcs):
    - obs[0]: Focus (distance from track center)
    - obs[1-19]: Track sensors
    - obs[31]: Speed (km/h)
    - obs[33]: Fuel level
    """
    speed = obs[31]
    fuel = obs[33]
    # Simple logic to check 'tire health' or 'track position'
    # In gym_torcs, you can derive tire state from damage or distance
    
    briefing = (
        f"DATA REPORT: Current Speed: {speed:.1f} km/h. "
        f"Fuel Remaining: {fuel:.1f}. "
        "Engine temperature and tires look stable for now."
    )
    return briefing

# 2. UPDATED CHAT FUNCTION
def ask_engineer(user_query, current_obs):
    # Get the plain-English version of the numbers
    telemetry_context = get_engineer_briefing(current_obs)
    
    messages = [
        {'role': 'system', 'content': 'You are a professional F1 Race Engineer. Use telemetry to give short, tactical advice.'},
        {'role': 'user', 'content': f"{telemetry_context}\n\nDRIVER QUESTION: {user_query}"}
    ]

    response = ollama.chat(model='granite3.1-dense:8b', messages=messages)
    return response['message']['content']

# 3. THE MAIN LOOP (Combined)
def run_integrated_session():
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    obs = env.reset()
    
    print("\nEngineer: Radio Check. Driver, do you copy?")
    
    running = True
    while running:
        # Standard Driving Logic
        steer, throttle, brake = get_human_input()
        act = np.array([steer, throttle, brake], dtype=np.float32)
        obs, reward, done, info = env.step(act)

        # CHECK FOR CHAT TRIGGER
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # Press 'C' to talk to the Engineer
                if event.key == pygame.K_c:
                    # Pause the game (optional) or just ask
                    query = input("Ask Engineer: ")
                    reply = ask_engineer(query, obs)
                    print(f"\nENGINEER: {reply}\n")
                
                if event.key == pygame.K_q:
                    running = False
        
        if done: break

    env.end()