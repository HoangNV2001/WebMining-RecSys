import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from envs import OfflineEnv
from recommender import DRRAgent

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m')
STATE_SIZE = 10
TOP_K = 10

SAVED_ACTOR_PATH = './save_model/trail-2026-01-21-11/actor_10_fixed.weights.h5' 
SAVED_CRITIC_PATH = './save_model/trail-2026-01-21-11/critic_10_fixed.weights.h5'

if __name__ == "__main__":
    print('Data loading for Evaluation...')

    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
    users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
    movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'),encoding='latin-1').readlines()]

    ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'])
    ratings_df = ratings_df.astype(np.uint32)

    movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
    movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

    print("Data preprocessing...")
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}

    users_dict = {user : [] for user in set(ratings_df["UserID"])}
    ratings_df = ratings_df.sort_values(by='Timestamp', ascending=True)

    ratings_df_gen = ratings_df.iterrows()
    users_dict_for_history_len = {user : [] for user in set(ratings_df["UserID"])}
    
    for data in ratings_df_gen:
        users_dict[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))
        if data[1]['Rating'] >= 4:
            users_dict_for_history_len[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))

    users_history_lens = [len(users_dict_for_history_len[u]) for u in set(ratings_df["UserID"])]
    
    users_num = max(ratings_df["UserID"])+1
    items_num = max(ratings_df["MovieID"])+1

    eval_users_num = int(users_num * 0.2)
    eval_users_dict = {k:users_dict[k] for k in range(users_num-eval_users_num, users_num)}
    eval_users_history_lens = users_history_lens[-eval_users_num:]

    print(f"Evaluation Users: {len(eval_users_dict)}")

    def calculate_ndcg(rel, irel):
        dcg = 0
        idcg = 0
        rel = [1 if r>0 else 0 for r in rel]
        for i, (r, ir) in enumerate(zip(rel, irel)):
            dcg += (r)/np.log2(i+2)
            idcg += (ir)/np.log2(i+2)
        return dcg, idcg

    def evaluate(recommender, env, check_movies=False, top_k=10):
        episode_reward = 0
        steps = 0
        mean_precision = 0
        mean_ndcg = 0
        
        user_id, items_ids, done = env.reset()
        
        if check_movies:
            print(f'User_id : {user_id}, History length: {len(env.user_items)}')

        while not done:
            # Embedding
            user_eb = recommender.embedding_network.get_layer('user_embedding')(np.array(user_id))
            items_eb = recommender.embedding_network.get_layer('movie_embedding')(np.array(items_ids))
            
            # State
            state = recommender.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
            
            # Action
            action = recommender.actor.network(state)
            
            # Recommend
            recommended_item = recommender.recommend_item(action, env.recommended_items, top_k=top_k)
            
            # Step
            next_items_ids, reward, done, _ = env.step(recommended_item, top_k=top_k)
            
            if top_k:
                correct_list = [1 if r > 0 else 0 for r in reward]
                
                # NDCG
                dcg, idcg = calculate_ndcg(correct_list, [1 for _ in range(len(reward))])
                mean_ndcg += dcg/idcg if idcg > 0 else 0
                
                # Precision
                correct_num = correct_list.count(1)
                mean_precision += correct_num/top_k
            
            reward_sum = np.sum(reward)
            items_ids = next_items_ids
            episode_reward += reward_sum
            steps += 1
            
            if check_movies:
                print(f'Recommended: {recommended_item}')
                print(f'Precision: {correct_num/top_k:.3f}, NDCG: {(dcg/idcg if idcg>0 else 0):.3f}')
            
            break
        
        return mean_precision, mean_ndcg

    sum_precision = 0
    sum_ndcg = 0
    
    end_evaluation = 10 
    count_eval = 0

    print("Starting evaluation loop...")

    for i, user_id in enumerate(eval_users_dict.keys()):
        env = OfflineEnv(eval_users_dict, users_history_lens, movies_id_to_movies, STATE_SIZE, fix_user_id=user_id)
        
        # Khởi tạo Agent
        recommender = DRRAgent(env, users_num, items_num, STATE_SIZE)
        
        # Build networks
        recommender.actor.build_networks()
        recommender.critic.build_networks()
        
        # Load weights đã train
        try:
            recommender.load_model(SAVED_ACTOR_PATH, SAVED_CRITIC_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please check the path to .weights.h5 files")
            break

        # Đánh giá
        precision, ndcg = evaluate(recommender, env, check_movies=True, top_k=TOP_K)
        
        sum_precision += precision
        sum_ndcg += ndcg
        count_eval += 1
        
        print(f"User {i}: Precision@{TOP_K}={precision:.4f}, NDCG@{TOP_K}={ndcg:.4f}")

        if i >= end_evaluation:
            break
    
    # Tính trung bình
    if count_eval > 0:
        print(f'\n=== FINAL RESULT over {count_eval} users ===')
        print(f'Average Precision@{TOP_K} : {sum_precision/count_eval:.4f}')
        print(f'Average NDCG@{TOP_K}      : {sum_ndcg/count_eval:.4f}')
    else:
        print("No evaluation performed.")