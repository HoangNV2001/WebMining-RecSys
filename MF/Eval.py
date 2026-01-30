# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from collections import defaultdict
import time
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD & PREPROCESS DATA
# ==========================================
print(">>> [1/5] Loading Data...")
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
file_path = r'ml-1m/ratings.dat'

try:
    ratings_df = pd.read_csv(file_path, sep='::', names=r_cols, engine='python', encoding='latin-1')
except FileNotFoundError:
    print(f"File not found at: {file_path}")
    print("Creating dummy data for testing...")
    users = np.random.randint(1, 6041, 100000)
    items = np.random.randint(1, 3707, 100000)
    ratings = np.random.randint(1, 6, 100000)
    ratings_df = pd.DataFrame({'user_id': users, 'movie_id': items, 'rating': ratings})

## Encode rating 0-1 (Thích >= 4)
ratings_df['rating_bin'] = np.where(ratings_df['rating'] >= 4, 1, 0)
ratings_df = ratings_df.dropna(subset=["user_id", "movie_id", "rating"])

# Map ID về 0-index
u = (ratings_df["user_id"].to_numpy(dtype=np.int32) - 1)
i = (ratings_df["movie_id"].to_numpy(dtype=np.int32) - 1)
r = ratings_df["rating_bin"].to_numpy(dtype=np.int8)

n_users = int(u.max()) + 1
n_items = int(i.max()) + 1

Y_data = np.stack([u, i, r], axis=1) # shape (N, 3)
print(f"Users={n_users}, Items={n_items}, Interactions={len(Y_data)}")

# ==========================================
# 2. HELPER FUNCTIONS (METRICS)
# ==========================================
def precision_at_k(ranked_items, relevant_items, k):
    if k <= 0: return 0.0
    hit = 0
    for item in ranked_items[:k]:
        if item in relevant_items:
            hit += 1
    return hit / k

def ndcg_at_k(ranked_items, relevant_items, k):
    dcg = 0.0
    for idx, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            dcg += 1.0 / np.log2(idx + 2)
    ideal_hits = min(len(relevant_items), k)
    if ideal_hits == 0: return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg

def evaluate_fold(predict_func, user_pos_test, user_seen, k_list=(5, 10)):
    """Đánh giá model trên 1 fold"""
    results = {f"P@{k}": [] for k in k_list}
    results.update({f"NDCG@{k}": [] for k in k_list})

    # Chỉ đánh giá những user có trong tập test (ground truth)
    test_users = list(user_pos_test.keys())
    
    for uu in test_users:
        rel_items = user_pos_test[uu]
        
        # 1. Dự đoán điểm
        scores = predict_func(uu) 
        
        # 2. Mask items đã xem trong train
        seen = user_seen.get(uu, set())
        if seen:
            scores[list(seen)] = -np.inf
        
        # 3. Lấy Top-K (tối ưu tốc độ bằng argpartition)
        max_k = max(k_list)
        if max_k >= len(scores):
            ranked_top = np.argsort(scores)[::-1]
        else:
            idx_part = np.argpartition(scores, -max_k)[-max_k:]
            ranked_top = idx_part[np.argsort(scores[idx_part])[::-1]]

        # 4. Tính metrics
        for k in k_list:
            results[f"P@{k}"].append(precision_at_k(ranked_top, rel_items, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(ranked_top, rel_items, k))
            
    # Trả về trung bình của fold
    return {m: np.mean(v) for m, v in results.items()}

# ==========================================
# 3. K-FOLD CROSS VALIDATION LOOP
# ==========================================
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

# Dictionary để lưu kết quả của từng fold cho từng model
final_results = defaultdict(lambda: defaultdict(list))

print(f"\n>>> [2/5] Starting {K}-Fold Cross Validation...")

fold_idx = 1
for train_index, test_index in kf.split(Y_data):
    print(f"\n--- FOLD {fold_idx}/{K} ---")
    
    # 1. Split Data
    Y_train_fold = Y_data[train_index]
    Y_test_fold = Y_data[test_index]
    
    # 2. Build Sparse Matrix (Train)
    R_train = csr_matrix(
        (Y_train_fold[:, 2].astype(np.float32), (Y_train_fold[:, 0], Y_train_fold[:, 1])),
        shape=(n_users, n_items), dtype=np.float32
    )
    
    # 3. Prepare Ground Truth (Test) & Seen Items (Train)
    user_pos_test = defaultdict(set)
    for uu, ii, rr in Y_test_fold:
        if int(rr) == 1: user_pos_test[int(uu)].add(int(ii))
            
    user_seen = defaultdict(set)
    for uu, ii, rr in Y_train_fold:
        user_seen[int(uu)].add(int(ii))

    # ==========================
    # MODEL A: SVD
    # ==========================
    # print("   Training SVD...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    U_svd = svd.fit_transform(R_train)
    Vt_svd = svd.components_
    
    def pred_svd(u): return U_svd[u] @ Vt_svd
    
    res_svd = evaluate_fold(pred_svd, user_pos_test, user_seen)
    for k, v in res_svd.items(): final_results['SVD'][k].append(v)
    
    # ==========================
    # MODEL B: ITEM-BASED CF
    # ==========================
    # print("   Training Item-Based...")
    sim_item = cosine_similarity(R_train.T, dense_output=True)
    
    def pred_item(u): return R_train[u].toarray().flatten() @ sim_item
    
    res_item = evaluate_fold(pred_item, user_pos_test, user_seen)
    for k, v in res_item.items(): final_results['Item-Base'][k].append(v)

    # ==========================
    # MODEL C: USER-BASED CF
    # ==========================
    # print("   Training User-Based...")
    sim_user = cosine_similarity(R_train, dense_output=True)
    
    def pred_user(u): return sim_user[u] @ R_train
    
    res_user = evaluate_fold(pred_user, user_pos_test, user_seen)
    for k, v in res_user.items(): final_results['User-Base'][k].append(v)
    
    print(f"   Done Fold {fold_idx}. SVD P@10: {res_svd['P@10']:.4f} | Item P@10: {res_item['P@10']:.4f} | User P@10: {res_user['P@10']:.4f}")
    fold_idx += 1

# ==========================================
# 4. FINAL REPORT (MEAN +/- STD)
# ==========================================
print("\n" + "="*60)
print(f"FINAL REPORT ({K}-FOLD CROSS VALIDATION)")
print("="*60)

# Định dạng output đẹp
metrics_order = ['P@5', 'NDCG@5', 'P@10', 'NDCG@10']

for model_name, metrics in final_results.items():
    print(f"\n MODEL: {model_name}")
    for m in metrics_order:
        values = metrics[m]
        mean_val = np.mean(values)
        std_val = np.std(values)
        # In ra định dạng: Metric: Mean +- Std
        print(f"   {m:<8}: {mean_val:.4f} ± {std_val:.4f}")

print("\n>>> ALL DONE.")