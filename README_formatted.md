# 🕹️ Flappy Bird — DQN / Double-DQN (Project Overview)

**This repository implements a compact Flappy Bird environment, DQN / Double-DQN agents, training & evaluation utilities, and a simple GUI / console front-end to run experiments and play the trained agent.**

🔗 Live demo / usage: see `main.py`, `console_main.py`, and `play.py` for running and evaluation.

---

## ✨ Features

- Compact Gym-like Flappy Bird environment (`env.py`) with `reset()`, `step()`, `render()`, `close()`.
- DQN and Double-DQN agents (`agent.py`, `agent_ddqn.py`) with policy & target networks, epsilon-greedy selection, and update routines.
- Replay buffer and utilities in `utils.py` for storing and sampling transitions.
- Training loop in `train.py` with warmup, epsilon schedule, logging, and checkpoint saving.
- Play / evaluation helpers in `play.py` with both rendered and headless modes.
- Simple Tk GUI (`main.py`) and CLI (`console_main.py`) front-ends.
- Notebooks for experimentation: `flappy-bird-dqn.ipynb`, `double_dqn_train.ipynb`.

---

## 🗂️ Project Flow

1. Configure parameters in `config.py` — screen, physics, rewards, checkpoint path, training episodes.
2. Use `env.py` to run the environment and collect transitions.
3. Train with `train.py` (or notebooks) using `agent.py` (DQN) or `agent_ddqn.py` (Double-DQN).
4. Monitor and visualize with `plot.py`, and run/play trained agents with `play.py` or `main.py` GUI.

---

## 📁 Files & Responsibilities

- `config.py` — central configuration (screen size, physics, rewards, checkpoint paths, episodes).
- `env.py` — FlappyBirdEnv implementation and rendering fallback if assets are missing.
- `agent.py` — DQN agent (policy & target nets, updates, save/load).
- `agent_ddqn.py` — Double-DQN variant (policy selects next action, target evaluates).
- `utils.py` — ReplayBuffer and helper utilities.
- `train.py` — main training loop: `train_loop(num_episodes=..., render=False, resume=False, difficulty="normal")`.
- `play.py` — play/evaluate functions (`play_model`, `play_model_no_render`).
- `main.py` — Tkinter GUI for Train / Load+Train / Play.
- `console_main.py` — CLI menu for headless training.
- `plot.py` — simple plotting utilities.
- `flappy-bird-dqn.ipynb`, `double_dqn_train.ipynb` — notebook-based experiments.
- `assets/` — images for background, bird, pipes, base, etc. If missing, renderer falls back to shapes.

---

## ⚙️ Installation & Quick Start

This is a lightweight Python project. Typical steps to run locally:

1. Create (optional) and activate a Python virtual environment.
2. Install dependencies (torch, pygame, numpy, matplotlib, etc.) — adjust versions as needed for your environment.
3. Configure parameters in `config.py`.
4. For headless training, run `console_main.py`. For GUI, run `main.py`.

(Exact dependency pins are not included in the repository; use your preferred Python environment manager and install packages required by the code.)

---

## 🖼️ Assets

Place image assets in the `assets/` directory for full visuals. If assets are missing, rendering uses simple shapes so training and headless runs are still possible.

---

## 💾 Checkpoints

The default checkpoint path is defined in `config.py` as `CHECKPOINT_PATH`. Training saves model state (policy + target). GUI and play scripts load automatically from that location.

---

## 💡 Tips

- Use `console_main.py` or notebooks for headless training (useful on servers).
- Tweak rewards and physics in `config.py` to adjust agent behavior.
- Switch to Double-DQN (`agent_ddqn.py`) for more stable learning.
- Watch `best.pth` and training logs to monitor progress.

---

## 📬 Contact / Next Steps

- Tweak `config.py` (e.g., `EPI_NUMS`, learning rate) to experiment.
- Run `console_main.py` to start training and monitor checkpoints.
- Launch `main.py` for GUI visualization and interactive play.

---

## ⚖️ License / Attribution

This is a student / educational project for learning purposes. You may reuse or reference it with proper attribution.

---

✨ “Flap, learn, repeat.”
