import os
import time
import numpy as np
import pygame
from env import FlappyBirdEnv
from agent import Agent
from utils import ReplayBuffer
from config import EPI_NUMS, CHECKPOINT_PATH


def train_loop(num_episodes=EPI_NUMS, render=False, resume=False, difficulty="normal"):
    """
    Main training loop for DQN agent in Flappy Bird environment.
    Parameters:
        num_episodes (int): Number of episodes to train. Default: EPI_NUMS from config.py
        render (bool): Whether to render the environment. Default: False
        resume (bool): Whether to resume training from checkpoint. Default: False
        difficulty (str): Difficulty level of the game ('easy', 'normal', 'hard', 'extreme'). Default: 'normal'
    """

    # === Initialize environment, agent, and replay buffer ===
    env = FlappyBirdEnv(difficulty=difficulty, render_mode=render)
    agent = Agent()
    buffer = ReplayBuffer(50000)

    if resume and os.path.exists(CHECKPOINT_PATH):
        print("Loading checkpoint...")
        agent.load(CHECKPOINT_PATH)
        print("Loaded.")

    epsilon_start = 1.0
    epsilon_final = 0.02
    epsilon_decay = 15000  # steps

    total_steps = 0
    losses = []
    all_scores = []

    # --- WARMUP PHASE --- 
    warmup_steps = 5000
    print(f"Collecting {warmup_steps} random transitions for warmup...")
    state = env.reset()
    for step in range(warmup_steps):
        action = np.random.randint(0, agent.n_actions)  # random 0 hoặc 1
        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, float(done))

        if done:
            state = env.reset()
        else:
            state = next_state

    print(f"Warmup finished. Replay buffer size = {len(buffer)}")

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False

        while not done:
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * max(0, (1 - total_steps / epsilon_decay))
            action = agent.act(state, epsilon)

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, float(done))

            loss = agent.update(buffer, batch_size=64, target_update=1000)
            if loss is not None:
                losses.append(loss)

            state = next_state
            ep_reward += reward
            total_steps += 1
            steps += 1

            if render:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        env.close()
                        return
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                        env.close()
                        return
                env.render()

        all_scores.append(env.score)
        if ep % 10 == 0:
            avg_score = np.mean(all_scores[-50:]) if len(all_scores) >= 1 else env.score
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 1 else 0.0
            print(f"Ep {ep:4d} | Steps {total_steps:6d} | Score {env.score:3d} | EpReward {ep_reward:.2f} | "
                  f"Epsilon {epsilon:.3f} | AvgScore50 {avg_score:.2f} | AvgLoss100 {avg_loss:.4f}")
            # save checkpoint
            agent.save(CHECKPOINT_PATH)

    # final save
    agent.save(CHECKPOINT_PATH)
    env.close()
    print("Training finished. Model saved to", CHECKPOINT_PATH)