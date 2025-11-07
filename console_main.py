from train import train_loop
from config import EPI_NUMS
from play import play_model, play_model_no_render

def main():
    """Main menu interface for Flappy Bird DQN."""
    print("=" * 40)
    print("Flappy DQN")
    print("=" * 40)
    print("1: Train new model (headless)")
    print("2: Continue training from checkpoint (headless)")
    print("3: Play with trained model (render)")
    print("4: Play with trained model (NO render)")
    print("5: Quit")
    print("\nNote: Press ESC to quit during render mode.")
    
    choice = input("Your choice (1-5): ").strip()
    
    if choice == "1":
        train_loop(num_episodes=EPI_NUMS, render=False, resume=False, difficulty="normal")
    elif choice == "2":
        train_loop(num_episodes=EPI_NUMS, render=False, resume=True)
    elif choice == "3":
        play_model(render=True, dif="normal") # hidden difficulty level
    elif choice == "4":
        play_model_no_render(target_score=1000, num_episodes=1, dif="normal") # hidden difficulty level
    elif choice == "5":
        print("Exiting...")
    else:
        print("Unknown choice. Exiting.")

if __name__ == "__main__":
    main()