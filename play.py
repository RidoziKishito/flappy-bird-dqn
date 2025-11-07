from env import FlappyBirdEnv
from utils import ReplayBuffer
import os
import numpy as np
import pygame
import time
import matplotlib.pyplot as plt
from agent import Agent
from config import CHECKPOINT_PATH


def play_model(num_episodes=1, dif="normal", render=True, target_score=1000):
    env = FlappyBirdEnv(difficulty=dif, render_mode=render)
    agent = Agent()

    # Load model
    if os.path.exists(CHECKPOINT_PATH):
        agent.load(CHECKPOINT_PATH)
        print("Loaded model for play.")
    else:
        print("No checkpoint found - play with random policy.")

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            # Handle quit / escape
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    env.close()
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    env.close()
                    return

            # Greedy action
            action = agent.act(state, epsilon=0.0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            steps += 1

            if render:
                env.render()

            # Early stop if high score reached
            if env.score >= target_score:
                print(f"[EP {ep}] Reached {target_score} at step {steps}, stopping early.")
                done = True

        print(f"[EP {ep}] Steps: {steps} | Score: {env.score} | Reward: {ep_reward:.2f}")

        # short delay before next episode
        if render:
            time.sleep(1.0)

    env.close()
    print("\nAll episodes finished.")


def play_model_no_render(num_episodes=10, target_score=1000, dif="normal"):
    env = FlappyBirdEnv(difficulty=dif, render_mode=False)
    agent = Agent()

    # Load model
    if os.path.exists(CHECKPOINT_PATH):
        agent.load(CHECKPOINT_PATH)
        print("Loaded model for play (headless).")
    else:
        print("No checkpoint found - play with random policy.")

    scores = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action = agent.act(state, epsilon=0.0)  # greedy policy
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            steps += 1

            # Stop if reached high score
            if env.score >= target_score:
                print(f"[EP {ep}] Reached {target_score} at step {steps}, stopping early.")
                done = True

        scores.append(env.score)
        print(f"[EP {ep}] Steps: {steps} | Score: {env.score} | Reward: {ep_reward:.2f}")

    env.close()

    # --- Plot results ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_episodes + 1), scores, marker='o', label='Score per Episode')
    plt.axhline(y=sum(scores)/len(scores), color='r', linestyle='--', label=f'Average ({sum(scores)/len(scores):.2f})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Flappy Bird — Evaluation Results (No Render)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print summary
    print("\nSummary:")
    print(f"Scores: {scores}")
    print(f"Average score: {sum(scores)/len(scores):.2f}")