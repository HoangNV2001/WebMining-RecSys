import pandas as pd
import numpy as np
import tensorflow as tf
import os
import glob
import re

from envs import OfflineEnv
from recommender import DRRAgent

# --- CẤU HÌNH ---
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m')
STATE_SIZE = 10
EMBEDDING_DIM = 100
SAVE_MODEL_DIR = "./save_model" 

def calculate_metrics(recommended_items, true_items, k=10):
    """
    Tính Precision@K và NDCG@K cho một user
    """
    # Lấy Top-K items
    top_k_items = recommended_items[:k]
    
    # --- Precision@K ---
    # Số item gợi ý đúng (có trong tập true_items) chia cho K
    hit_count = len(set(top_k_items) & set(true_items))
    precision = hit_count / k
    
    # --- NDCG@K ---
    dcg = 0.0
    for i, item in enumerate(top_k_items):
        if item in true_items:
            # log2(i+2) vì rank bắt đầu từ 0 (công thức là log2(i+1) với rank từ 1)
            dcg += 1.0 / np.log2(i + 2)
            
    idcg = 0.0
    # IDCG là trường hợp lý tưởng: đưa tất cả item đúng lên đầu
    num_relevant = min(len(true_items), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)
        
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return precision, ndcg

def find_latest_weights(base_dir):
    """
    Tìm file weight actor và critic mới nhất trong toàn bộ các thư mục con
    """
    # Tìm tất cả file actor_*.weights.h5
    actor_files = glob.glob(os.path.join(base_dir, "**", "actor_*.weights.h5"), recursive=True)
    
    if not actor_files:
        raise FileNotFoundError("Không tìm thấy file weight nào trong " + base_dir)

    # Sắp xếp theo thời gian sửa đổi (mới nhất cuối cùng)
    actor_files.sort(key=os.path.getmtime)
    latest_actor = actor_files[-1]
    
    folder_path = os.path.dirname(latest_actor)
    filename = os.path.basename(latest_actor)
    
    # Lấy số episode từ tên file (ví dụ: actor_5000.weights.h5)
    match = re.search(r'actor_(\d+)', filename)
    if match:
        ep_num = match.group(1)
        latest_critic = os.path.join(folder_path, f"critic_{ep_num}.weights.h5")
    else:
        latest_critic = glob.glob(os.path.join(folder_path, "critic_*.weights.h5"))
        if latest_critic:
            latest_critic.sort(key=os.path.getmtime)
            latest_critic = latest_critic[-1]
        else:
            raise FileNotFoundError("Tìm thấy actor nhưng không thấy critic tương ứng.")
            
    print(f"\n>>> FOUND LATEST MODEL:")
    print(f"    Actor:  {latest_actor}")
    print(f"    Critic: {latest_critic}")
    
    return latest_actor, latest_critic

# --- MAIN EVALUATION ---
if __name__ == "__main__":
    print('>>> Loading Data for Evaluation...')
    
    # 1. Load Data 
    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
    ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    ratings_df = ratings_df.astype(np.uint32)
    
    movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'), encoding='latin-1').readlines()]
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}

    users_dict = np.load('./data/user_dict.npy', allow_pickle=True)
    users_history_lens = np.load('./data/users_histroy_len.npy')

    users_num = max(ratings_df["UserID"]) + 1
    items_num = max(ratings_df["MovieID"]) + 1

    # Chia dữ liệu Train/Test (80/20)
    train_users_num = int(users_num * 0.8)
    
    # LẤY TEST SET (20% user cuối)
    test_users_dict = {k: users_dict.item().get(k) for k in range(train_users_num + 1, users_num)}
    test_users_history_lens = users_history_lens[train_users_num:]
    
    print(f">>> Test Users: {len(test_users_dict)}")
    
    # 2. Khởi tạo Môi trường Test
    env = OfflineEnv(test_users_dict, test_users_history_lens, movies_id_to_movies, STATE_SIZE)

    # 3. Khởi tạo Agent (is_test=True để tắt exploration)
    agent = DRRAgent(env, users_num, items_num, STATE_SIZE, is_test=True, use_wandb=False)
    
    # Build networks trước khi load weights
    agent.actor.build_networks()
    agent.critic.build_networks()
    
    # 4. Load Model Mới Nhất
    try:
        actor_path, critic_path = find_latest_weights(SAVE_MODEL_DIR)
        agent.load_model(actor_path, critic_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check your save path.")
        exit()

    # 5. Bắt đầu đánh giá
    print("\n>>> STARTING EVALUATION (Precision@K, NDCG@K)...")
    
    p_5_list, p_10_list = [], []
    n_5_list, n_10_list = [], []
    
    TEST_EPISODES = len(test_users_dict) # Test hết các user
    
    for i in range(TEST_EPISODES):
        user_id, items_ids, done = env.reset()
        
        user_full_history = test_users_dict[user_id]
        # Lấy các item tương lai (những item chưa có trong state hiện tại - 10 item cuối làm state)
        # Giả sử items_ids hiện tại là state.
        
        user_eb = agent.embedding_network.get_layer('user_embedding')(np.array(user_id))
        items_eb = agent.embedding_network.get_layer('movie_embedding')(np.array(items_ids))
        state = agent.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
        
        action = agent.actor.network(state)
        
        # Lấy Top 10 items gợi ý (loại trừ các item đã xem trong state)
        recommended_items = agent.recommend_item(action, items_ids, top_k=10)
        
        # GROUND TRUTH: Là tất cả các item user đã xem/thích mà KHÔNG nằm trong history input (items_ids)
        ground_truth = [item[0] for item in user_full_history if item[0] not in items_ids]
        
        if len(ground_truth) == 0:
            continue # Bỏ qua nếu user không còn item nào để test
            
        # Tính toán Metrics
        p_5, n_5 = calculate_metrics(recommended_items, ground_truth, k=5)
        p_10, n_10 = calculate_metrics(recommended_items, ground_truth, k=10)
        
        p_5_list.append(p_5)
        p_10_list.append(p_10)
        n_5_list.append(n_5)
        n_10_list.append(n_10)
        
        if (i+1) % 100 == 0:
            print(f"Evaluated {i+1}/{TEST_EPISODES} users...")

    # 6. Kết quả cuối cùng
    print(f"\n{'='*30}")
    print(f"FINAL EVALUATION RESULTS ({len(p_5_list)} users)")
    print(f"{'='*30}")
    print(f"Precision@5 : {np.mean(p_5_list)*100:.4f} %")
    print(f"Precision@10: {np.mean(p_10_list)*100:.4f} %")
    print(f"NDCG@5      : {np.mean(n_5_list):.4f}")
    print(f"NDCG@10     : {np.mean(n_10_list):.4f}")
    print(f"{'='*30}")