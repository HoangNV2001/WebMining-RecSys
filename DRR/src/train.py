import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

from envs import OfflineEnv
from recommender import DRRAgent

# --- CONFIG ---
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m')
STATE_SIZE = 10
MAX_EPISODE_NUM = 10000 # Số episode mỗi strategy

# GPU Check
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available:", gpus)


def smooth_curve(data, window_size=50):
    """Làm mượt dữ liệu để vẽ biểu đồ đẹp"""
    if len(data) < window_size: return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')

def plot_comparison(all_logs, save_path):
    """Vẽ biểu đồ so sánh tất cả các strategy"""
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    WINDOW = 50

    metrics = [
        ("precision", "Precision (%)", axs[0, 0]),
        ("reward", "Total Reward", axs[0, 1]),
        ("mean_action", "Mean Action", axs[1, 0]),
        ("q_loss", "Q Loss", axs[1, 1])
    ]

    for key, title, ax in metrics:
        for strategy, logs in all_logs.items():
            smoothed = smooth_curve(logs[key], WINDOW)
            x = logs["step"][len(logs["step"]) - len(smoothed):]
            ax.plot(x, smoothed, label=strategy, linewidth=1.5)
        
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    print('>>> Loading Data...')
    # Load Ratings
    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
    ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    ratings_df = ratings_df.astype(np.uint32)
    
    # Load Movies
    movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'), encoding='latin-1').readlines()]
    movies_df = pd.DataFrame(movies_list, columns=['MovieID', 'Title', 'Genres'])
    movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}

    # Load User History (Pre-processed)
    users_dict = np.load('./data/user_dict.npy', allow_pickle=True)
    users_history_lens = np.load('./data/users_histroy_len.npy')

    users_num = max(ratings_df["UserID"]) + 1
    items_num = max(ratings_df["MovieID"]) + 1

    # Split Train Set (80%)
    train_users_num = int(users_num * 0.8)
    train_users_dict = {k: users_dict.item().get(k) for k in range(1, train_users_num+1)}
    train_users_history_lens = users_history_lens[:train_users_num]
    
    print('>>> Data Ready!')
    time.sleep(1)

    # --- DEFINING STRATEGIES ---
    strategies = {
        "Best_Case (PER + Std=1.0)": {
            "std": 1.0, 
            "use_per": True
        },
        "Baseline (No PER + Std=1.0)": {
            "std": 1.0, 
            "use_per": False
        },
        "High_Exploration (PER + Std=1.5)": {
            "std": 1.5,
            "use_per": True
        }
    }

    all_logs = {} # Lưu kết quả của tất cả các lần chạy

    # --- TRAINING LOOP ---
    for name, config in strategies.items():
        print(f"\n{'='*50}")
        print(f"STARTING STRATEGY: {name}")
        print(f"Config: {config}")
        print(f"{'='*50}")

        # Reset Environment for each strategy
        env = OfflineEnv(train_users_dict, train_users_history_lens, movies_id_to_movies, STATE_SIZE)
        
        # Init Agent
        recommender = DRRAgent(
            env, users_num, items_num, STATE_SIZE, 
            use_wandb=False,
            std=config["std"],
            use_per=config["use_per"]
        )
        
        recommender.actor.build_networks()
        recommender.critic.build_networks()
        
        # Train & Collect Logs
        logs = recommender.train(MAX_EPISODE_NUM, load_model=False)
        
        all_logs[name] = logs
        print(f"Finished Strategy: {name}")

    # --- FINAL PLOTTING ---
    print("\n>>> Generating Comparison Plots...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    save_path = f"./save_model/FINAL_COMPARISON_{timestamp}.png"
    
    plot_comparison(all_logs, save_path)
    print("ALL DONE.")