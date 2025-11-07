import matplotlib.pyplot as plt

episodes = list(range(1, 11))

double_score = [500, 500, 500, 500, 19, 243, 216, 142, 25, 285]
dqn_score = [163, 28, 142, 28, 44, 54, 500, 88, 29, 109]

plt.figure(figsize=(8, 5))
plt.plot(episodes, double_score, marker='o', label='Double DQN', linewidth=2)
plt.plot(episodes, dqn_score, marker='s', label='DQN', linewidth=2)
plt.title("Score per Episode", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
