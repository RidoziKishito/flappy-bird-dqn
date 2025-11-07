# 🕹️ **Flappy Bird — DQN / Double-DQN (Project Overview)**

---

## 📘 **Summary**
This repository implements a compact **Flappy Bird** environment, **DQN / Double-DQN agents**, training & evaluation utilities, and a simple **GUI / console front-end** to run experiments and play the trained agent.

---

## 🔁 **Project Flow**

1. **Configure parameters** in `config.py` — screen, physics, rewards, checkpoint path, training episodes.  
2. **Environment (`env.py`)** — concise Gym-like API:  
   `reset() -> state`, `step(action) -> (state, reward, done, info)`.  
3. **Agents (`agent.py`, `agent_ddqn.py`)** — define networks, epsilon-greedy action selection, loss computation, and updates.  
4. **Replay Buffer (`utils.py`)** — stores transitions and provides random batches for learning.  
5. **Training Loop (`train.py`)** — handles warmup, epsilon schedule, environment interaction, updates, logging, and checkpoint saves.  
6. **Evaluation / Play (`play.py`)** — runs greedy evaluation episodes with or without rendering (pygame / headless).  
7. **Front-ends:**
   - `main.py` — **Tk GUI** for Train / Load+Train / Play (Render / No Render).  
   - `console_main.py` — **CLI menu** for headless training (great for servers).  
8. **Notebooks** (`flappy-bird-dqn.ipynb`, `double_dqn_train.ipynb`) — alternate training scripts for experimentation or use on Jupyter/Kaggle.

---

## 📁 **Files and Responsibilities**

### 🧾 `config.py`
- Central configuration: screen size, physics constants, reward shaping, checkpoint paths, and training episodes.  
- Edit this file to tune environment and training defaults.

### 🐤 `env.py`
- Implements **FlappyBirdEnv** with deterministic logic.  
- API: `reset()`, `step()`, `render()`, `close()`.  
- Handles reward shaping, collisions, and fallback to simple shapes if assets are missing.

### 🧠 `agent.py`
- Standard **DQN agent**: policy + target networks, epsilon-greedy selection, `compute_loss()`, `update()`, `save()`, `load()`.  
- Uses **soft target updates**. Primary implementation for training.

### 🧩 `agent_ddqn.py`
- **Double-DQN variant** — uses policy net for next-state action selection and target net for evaluation.  
- Drop-in replacement for more stable learning.

### 📦 `utils.py`
- **ReplayBuffer**: deque-based, with `push()` and `sample(batch_size)` methods.  
- Stores transitions as `Transition` namedtuples.

### 🏋️ `train.py`
- Core **training loop**:  
  `train_loop(num_episodes=..., render=False, resume=False, difficulty="normal")`  
- Handles warmup, epsilon decay, training updates, checkpointing, and optional rendering.

### 🎮 `play.py`
- Helpers:
  - `play_model(num_episodes=1, dif="normal", render=True, target_score=1000)` — interactive play.  
  - `play_model_no_render(num_episodes=10, target_score=1000, dif="normal")` — headless evaluation + plotting.  
- Loads checkpoint automatically from `CHECKPOINT_PATH`.

### 🪟 `main.py`
- **Tkinter GUI** exposing: Train / Load+Train / Play (Render / No Render).  
- Redirects `stdout` to GUI console for easy log monitoring.

### 💻 `console_main.py`
- **CLI interface** for headless servers.  
- Ideal for remote or lightweight training.

### 📊 `plot.py`
- Quick plotting utility to visualize and compare sample scores.

### 🧠 `agent_ddqn.py`, `double_dqn_train.ipynb`, `flappy-bird-dqn.ipynb`
- Alternate training recipes and experiments for DQN / Double-DQN.

---

## 🖼️ **Assets**
- Directory: `assets/`  
  - Contains images for background, bird, pipes, base, etc.  
  - If missing → falls back to minimalistic shape-based rendering.  
  - Place image assets next to code for full visual experience.

---

## 💾 **Checkpoints**
- Default checkpoint path: defined in `config.py` → `CHECKPOINT_PATH`.  
- Training saves both **policy** and **target** network states.  
- GUI and play scripts load automatically from this location.

---

## 💡 **Tips**
- **Headless training:** use `console_main.py` or notebooks with dummy SDL drivers.  
- **Fine-tune behavior:** adjust rewards / physics in `config.py`.  
- **Stabilize learning:** switch to Double-DQN by importing `agent_ddqn.py`.  
- **Monitor performance:** watch `best.pth` and training logs.

---

## 📬 **Contact / Next Steps**
- To experiment quickly:
  1. Tweak parameters in `config.py` (e.g. `EPI_NUMS`, learning rate).  
  2. Run `console_main.py` to start training.  
  3. Monitor checkpoints and scores.  
- For visualization, launch `main.py` and interact via GUI.

---

## ⚖️ **License / Attribution**
> This is a **student / educational project** for learning purposes.  
> You may reuse or reference it with proper attribution.

---
✨ *“Flap, learn, repeat.”*
